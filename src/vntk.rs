// src/vntk.rs

use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::types::{TransitionMatrix, VntkOutput};

// ──────────────────────────────────────────────────────────────────────────────
// Public result type
// ──────────────────────────────────────────────────────────────────────────────

/// Output of a single VNTK call covering all beams at one decoding step.
///
/// Index arithmetic:
/// - `tokens[i * branch_size + j]`      — j-th token candidate for beam i
/// - `next_nodes[i * branch_size + j]`  — trie node reached by that token
/// - `valid[i * branch_size + j]`       — whether slot j is a real child
/// - `dense_masks[i * vocab_size + tok]`— O(1) membership test for beam i
#[derive(Debug, Clone)]
pub struct VntkResult {
    /// Token IDs: shape [n × branch_size], invalid slots hold 0.
    pub tokens: Vec<u32>,
    /// Next-node IDs: shape [n × branch_size], invalid slots hold 0.
    pub next_nodes: Vec<u32>,
    /// Validity flags: shape [n × branch_size].
    pub valid: Vec<bool>,
    /// Dense boolean mask: shape [n × vocab_size].
    pub dense_masks: Vec<bool>,
    /// B_t: the padded branch-factor used at this level.
    pub branch_size: usize,
}

impl VntkResult {
    /// Returns the valid (token, next_node) pairs for beam `i`.
    #[inline]
    pub fn children_for(&self, i: usize) -> impl Iterator<Item = (u32, u32)> + '_ {
        let base = i * self.branch_size;
        (0..self.branch_size).filter_map(move |j| {
            if self.valid[base + j] {
                Some((self.tokens[base + j], self.next_nodes[base + j]))
            } else {
                None
            }
        })
    }

    /// Returns the dense mask slice for beam `i` (length = vocab_size).
    #[inline]
    pub fn mask_for(&self, i: usize, vocab_size: usize) -> &[bool] {
        let base = i * vocab_size;
        &self.dense_masks[base..base + vocab_size]
    }

    /// Collapses all per-beam dense masks into a single OR-reduced mask of
    /// length `vocab_size`.  Used when all beams in a batch share one logit
    /// vector (single-query inference).
    pub fn global_mask(&self, vocab_size: usize) -> Vec<bool> {
        let n = self.dense_masks.len() / vocab_size;
        let mut out = vec![false; vocab_size];
        for i in 0..n {
            let base = i * vocab_size;
            for (o, &m) in out
                .iter_mut()
                .zip(&self.dense_masks[base..base + vocab_size])
            {
                *o |= m;
            }
        }
        out
    }

    /// Converts the dense bool mask for beam `i` into a packed `Vec<u64>`
    /// (same layout as `DenseMask::bits`) for cheap bitwise AND with
    /// the model's top-k mask.
    pub fn packed_mask_for(&self, i: usize, vocab_size: usize) -> Vec<u64> {
        let slice = self.mask_for(i, vocab_size);
        let words = vocab_size.div_ceil(64);
        let mut out = vec![0u64; words];
        for (idx, &set) in slice.iter().enumerate() {
            if set {
                out[idx / 64] |= 1u64 << (idx % 64);
            }
        }
        out
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// VNTK implementation
// ──────────────────────────────────────────────────────────────────────────────

impl TransitionMatrix {
    /// **Vectorized Node Transition Kernel** — Algorithm 2 from the paper.
    ///
    /// For each of the `n = batch_size × beam_width` active beams, reads the
    /// CSR row for that beam's current trie node and writes up to `B_t`
    /// (token, next-node) pairs into pre-allocated output buffers.
    ///
    /// # Layout
    /// All output buffers are flat and strided by `branch_size` (= B_t).
    ///
    /// # Parallelism
    /// The per-beam inner loop is embarrassingly parallel and runs via Rayon.
    /// Writes to disjoint buffer slices avoid any synchronisation overhead.
    ///
    /// # Arguments
    /// - `current_nodes` — flat slice of length `n`, one node ID per beam
    /// - `level`         — current decoding step (0-indexed); selects `B_t`
    ///
    /// # Panics
    /// Panics if `level >= sid_length` or any node ID ≥ `num_nodes`.
    pub fn vntk(&self, current_nodes: &[u32], level: usize) -> VntkResult {
        assert!(
            level < self.sid_length as usize,
            "level {level} out of range (sid_length={})",
            self.sid_length
        );

        let b_t = self.max_branches[level] as usize;
        let n = current_nodes.len();
        let v = self.vocab_size as usize;

        // Allocate output buffers up-front; rayon writes into disjoint slices.
        let mut tokens: Vec<u32> = vec![0u32; n * b_t];
        let mut next_nodes: Vec<u32> = vec![0u32; n * b_t];
        let mut valid: Vec<bool> = vec![false; n * b_t];
        let mut dense_masks: Vec<bool> = vec![false; n * v];

        // Split each output buffer into n contiguous chunks, one per beam,
        // then zip them together so each rayon task owns exactly its slice.
        let tok_chunks: Vec<&mut [u32]> = tokens.chunks_mut(b_t).collect();
        let next_chunks: Vec<&mut [u32]> = next_nodes.chunks_mut(b_t).collect();
        let valid_chunks: Vec<&mut [bool]> = valid.chunks_mut(b_t).collect();
        let mask_chunks: Vec<&mut [bool]> = dense_masks.chunks_mut(v).collect();

        // Bundle into a single Vec of mutable tuple-slices for rayon.
        tok_chunks
            .into_par_iter()
            .zip(next_chunks)
            .zip(valid_chunks)
            .zip(mask_chunks)
            .zip(current_nodes.par_iter())
            .for_each(|((((tok_s, next_s), valid_s), mask_s), &node)| {
                debug_assert!(
                    node < self.num_nodes,
                    "node {node} ≥ num_nodes {}",
                    self.num_nodes
                );

                // ── Phase 1: CSR boundary lookup ─────────────────────────────
                let row_start = self.row_pointers[node as usize] as usize;
                let row_end = self.row_pointers[node as usize + 1] as usize;
                let n_child = row_end - row_start;

                // ── Phase 2: Speculative copy into padded B_t slots ──────────
                // Slots beyond n_child remain zeroed (implicit padding).
                let fill = n_child.min(b_t);
                for j in 0..fill {
                    let entry = self.data[row_start + j];
                    tok_s[j] = entry[0];
                    next_s[j] = entry[1];
                    valid_s[j] = true;
                }

                // ── Phase 3: Scatter into dense vocab mask ───────────────────
                // Only `fill` entries are real; token IDs are already sorted.
                for j in 0..fill {
                    mask_s[self.data[row_start + j][0] as usize] = true;
                }
            });

        VntkResult {
            tokens,
            next_nodes,
            valid,
            dense_masks,
            branch_size: b_t,
        }
    }

    /// Thin wrapper that converts a `VntkResult` into the simpler `VntkOutput`
    /// expected by the test module (`next_nodes` flat vec + single bool mask).
    ///
    /// Only meaningful when `current_nodes` contains a single beam; for
    /// multi-beam callers use `VntkResult` directly.
    pub fn vntk_single(&self, node: u32, level: usize) -> VntkOutput {
        let result = self.vntk(&[node], level);
        VntkOutput {
            next_nodes: result.children_for(0).map(|(_, n)| n).collect(),
            mask: result.dense_masks[..self.vocab_size as usize].to_vec(),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Standalone function form (matches the test module's call convention)
// ──────────────────────────────────────────────────────────────────────────────

/// Calls `TransitionMatrix::vntk` and returns a `VntkOutput` shaped for the
/// test module:
/// - `next_nodes`: flat list of valid next-node IDs across all beams
/// - `mask`:       OR-reduced dense bool mask of length `vocab_size`
pub fn vntk(
    current_nodes: &[u32],
    matrix: &TransitionMatrix,
    level: usize,
    vocab_size: usize,
) -> VntkOutput {
    debug_assert_eq!(
        vocab_size, matrix.vocab_size as usize,
        "vocab_size mismatch"
    );
    let result = matrix.vntk(current_nodes, level);

    // Collect all valid next-node IDs in beam × child order.
    let next_nodes: Vec<u32> = (0..current_nodes.len())
        .flat_map(|i| result.children_for(i).map(|(_, n)| n))
        .collect();

    let mask = result.global_mask(vocab_size);

    VntkOutput { next_nodes, mask }
}
