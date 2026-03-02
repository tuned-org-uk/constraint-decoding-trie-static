// src/types.rs

use std::fmt;

// ──────────────────────────────────────────────────────────────────────────────
// TransitionMatrix
// ──────────────────────────────────────────────────────────────────────────────

/// Stacked CSR transition matrix: interleaves [col_idx, next_state] for
/// coalesced reads. Row i corresponds to trie node i; values are
/// (token_id, next_node_id) pairs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransitionMatrix {
    /// CSR row pointers: length = num_nodes + 1.
    /// `row_pointers[i]..row_pointers[i+1]` is the slice in `data` for node i.
    pub row_pointers: Vec<u32>,

    /// Interleaved pairs: [(token_id, next_node_id), …]
    pub data: Vec<[u32; 2]>,

    /// Max observed branch factor at each trie depth: length = sid_length.
    pub max_branches: Vec<u32>,

    /// Total number of trie nodes (states).
    pub num_nodes: u32,

    /// Vocabulary size |V|.
    pub vocab_size: u32,

    /// Semantic ID length L.
    pub sid_length: u32,
}

impl TransitionMatrix {
    /// Construct an empty matrix for `num_nodes` nodes.
    pub fn new(num_nodes: u32, vocab_size: u32, sid_length: u32) -> Self {
        Self {
            row_pointers: vec![0u32; num_nodes as usize + 1],
            data: Vec::new(),
            max_branches: vec![0u32; sid_length as usize],
            num_nodes,
            vocab_size,
            sid_length,
        }
    }

    /// Returns the children slice for `node` as `&[[token_id, next_node]; _]`.
    ///
    /// # Panics
    /// Panics if `node >= num_nodes`.
    #[inline]
    pub fn children(&self, node: u32) -> &[[u32; 2]] {
        assert!(
            node < self.num_nodes,
            "node {node} out of range (num_nodes={})",
            self.num_nodes
        );
        let start = self.row_pointers[node as usize] as usize;
        let end = self.row_pointers[node as usize + 1] as usize;
        &self.data[start..end]
    }

    /// Looks up the next node reached from `node` by emitting `token`.
    /// Returns `None` if the transition does not exist (invalid / masked).
    #[inline]
    pub fn next_node(&self, node: u32, token: u32) -> Option<u32> {
        self.children(node)
            .iter()
            .find(|&&[t, _]| t == token)
            .map(|&[_, n]| n)
    }

    /// Returns `true` if `node` has no outgoing transitions (i.e. is a leaf).
    #[inline]
    pub fn is_leaf(&self, node: u32) -> bool {
        self.children(node).is_empty()
    }

    /// Number of outgoing transitions (branches) from `node`.
    #[inline]
    pub fn degree(&self, node: u32) -> u32 {
        self.children(node).len() as u32
    }

    /// Validates internal invariants; useful inside `debug_assert!`.
    pub fn check_invariants(&self) -> Result<(), String> {
        if self.row_pointers.len() != self.num_nodes as usize + 1 {
            return Err(format!(
                "row_pointers length {} ≠ num_nodes+1 {}",
                self.row_pointers.len(),
                self.num_nodes + 1
            ));
        }
        let last = *self.row_pointers.last().unwrap() as usize;
        if last != self.data.len() {
            return Err(format!(
                "row_pointers tail {last} ≠ data.len() {}",
                self.data.len()
            ));
        }
        // Rows must be non-decreasing
        for w in self.row_pointers.windows(2) {
            if w[0] > w[1] {
                return Err(format!("row_pointers not monotone: {} > {}", w[0], w[1]));
            }
        }
        // All token ids must be in [0, vocab_size)
        for &[tok, nxt] in &self.data {
            if tok >= self.vocab_size {
                return Err(format!("token {tok} ≥ vocab_size {}", self.vocab_size));
            }
            if nxt >= self.num_nodes {
                return Err(format!("next_node {nxt} ≥ num_nodes {}", self.num_nodes));
            }
        }
        Ok(())
    }
}

impl fmt::Display for TransitionMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TransitionMatrix(nodes={}, edges={}, |V|={}, L={})",
            self.num_nodes,
            self.data.len(),
            self.vocab_size,
            self.sid_length,
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// DenseMask
// ──────────────────────────────────────────────────────────────────────────────

/// Bit-packed dense mask for the first `depth` trie layers.
///
/// For `depth = 2` and `vocab_size = V` this is a V × V bit matrix stored as
/// packed `u64` words (row-major).  A set bit at linear index
/// `i * vocab_size + j` means the 2-token prefix `[i, j]` exists in C.
///
/// `states[i * vocab_size + j]` is the trie node reached after emitting
/// `[i, j]`; 0 is used as a sentinel for *invalid* entries (the root is
/// never a valid post-prefix destination in practice).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseMask {
    /// Packed u64 bits.  Length = ceil(vocab_size^depth / 64).
    pub bits: Vec<u64>,

    /// Flat node-ID lookup: length = vocab_size ^ depth.
    /// Entry is 0 (sentinel) when the corresponding prefix is absent.
    pub states: Vec<u32>,

    /// Number of dense layers `d`.
    pub depth: u32,

    /// Vocabulary size |V|.
    pub vocab_size: u32,
}

impl DenseMask {
    /// Allocate a zeroed mask for `vocab_size^depth` entries.
    pub fn new(vocab_size: u32, depth: u32) -> Self {
        let total = (vocab_size as usize).pow(depth);
        let words = total.div_ceil(64);
        Self {
            bits: vec![0u64; words],
            states: vec![0u32; total],
            depth,
            vocab_size,
        }
    }

    /// Converts a token sequence of length `depth` to a flat index.
    ///
    /// # Panics
    /// Panics in debug builds if `tokens.len() != depth`.
    #[inline]
    pub fn flat_index(&self, tokens: &[u32]) -> usize {
        debug_assert_eq!(tokens.len(), self.depth as usize);
        tokens.iter().fold(0usize, |acc, &t| {
            acc * self.vocab_size as usize + t as usize
        })
    }

    /// Sets the bit and stores the destination `node_id` for prefix `tokens`.
    pub fn insert(&mut self, tokens: &[u32], node_id: u32) {
        let idx = self.flat_index(tokens);
        let word = idx / 64;
        let bit = idx % 64;
        self.bits[word] |= 1u64 << bit;
        self.states[idx] = node_id;
    }

    /// Returns `true` if the prefix encoded by `tokens` is marked valid.
    #[inline]
    pub fn contains(&self, tokens: &[u32]) -> bool {
        let idx = self.flat_index(tokens);
        let word = idx / 64;
        let bit = idx % 64;
        (self.bits[word] >> bit) & 1 == 1
    }

    /// Two-token shorthand used pervasively in tests and the decoder.
    /// Equivalent to `contains(&[v1, v2])` when `depth == 2`.
    #[inline]
    pub fn get(&self, v1: u32, v2: u32) -> bool {
        debug_assert_eq!(self.depth, 2, "get(v1,v2) requires depth == 2");
        self.contains(&[v1, v2])
    }

    /// Returns the trie node reached after the valid prefix `tokens`,
    /// or `None` if the prefix is absent.
    #[inline]
    pub fn state_for(&self, tokens: &[u32]) -> Option<u32> {
        if self.contains(tokens) {
            Some(self.states[self.flat_index(tokens)])
        } else {
            None
        }
    }

    /// Iterates over all valid prefixes as `(tokens, node_id)`.
    pub fn iter_valid(&self) -> impl Iterator<Item = (Vec<u32>, u32)> + '_ {
        let d = self.depth as usize;
        let v = self.vocab_size as usize;
        let total = v.pow(d as u32);
        (0..total).filter_map(move |idx| {
            let word = idx / 64;
            let bit = idx % 64;
            if (self.bits[word] >> bit) & 1 == 0 {
                return None;
            }
            // decode flat index back into token sequence
            let mut rem = idx;
            let mut toks = vec![0u32; d];
            for pos in (0..d).rev() {
                toks[pos] = (rem % v) as u32;
                rem /= v;
            }
            Some((toks, self.states[idx]))
        })
    }
}

impl fmt::Display for DenseMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let valid = self.bits.iter().map(|w| w.count_ones()).sum::<u32>();
        write!(
            f,
            "DenseMask(depth={}, |V|={}, valid_prefixes={valid})",
            self.depth, self.vocab_size
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// StaticIndex
// ──────────────────────────────────────────────────────────────────────────────

/// Combined STATIC index: dense mask for the first `d` layers,
/// sparse CSR matrix for all deeper layers.
#[derive(Debug, Clone)]
pub struct StaticIndex {
    /// Bit-packed dense mask covering the first `d` trie levels.
    pub dense: DenseMask,

    /// CSR transition matrix for levels `d..L`.
    pub sparse: TransitionMatrix,

    /// Total number of constraints |C|.
    pub num_constraints: usize,
}

impl StaticIndex {
    pub fn new(dense: DenseMask, sparse: TransitionMatrix, num_constraints: usize) -> Self {
        Self {
            dense,
            sparse,
            num_constraints,
        }
    }

    /// Quick sanity check delegating to both sub-structures.
    pub fn check_invariants(&self) -> Result<(), String> {
        self.sparse.check_invariants()
    }
}

impl fmt::Display for StaticIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "StaticIndex(|C|={}, {}, {})",
            self.num_constraints, self.dense, self.sparse
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// BeamState
// ──────────────────────────────────────────────────────────────────────────────

/// Live decoding state for a single batch of beam searches.
///
/// Shapes:
/// - `nodes`  : `[batch_size][beam_width]`  — current trie node per beam
/// - `scores` : `[batch_size][beam_width]`  — accumulated log-probability
/// - `tokens` : `[batch_size][beam_width][step]` — partial SID decoded so far
#[derive(Debug, Clone)]
pub struct BeamState {
    pub nodes: Vec<Vec<u32>>,
    pub scores: Vec<Vec<f64>>,
    pub tokens: Vec<Vec<Vec<u32>>>,
}

impl BeamState {
    /// Creates a blank state for `batch_size` queries, each with `beam_width`
    /// beams, all starting at the trie root (node 0) with log-prob 0.0.
    pub fn new(batch_size: usize, beam_width: usize) -> Self {
        Self {
            nodes: vec![vec![0u32; beam_width]; batch_size],
            scores: vec![vec![0.0f64; beam_width]; batch_size],
            tokens: vec![vec![Vec::new(); beam_width]; batch_size],
        }
    }

    pub fn batch_size(&self) -> usize {
        self.nodes.len()
    }
    pub fn beam_width(&self) -> usize {
        self.nodes.first().map_or(0, Vec::len)
    }

    /// Returns the current decoding step (number of tokens emitted so far).
    /// Assumes all beams in batch 0 are at the same step.
    pub fn step(&self) -> usize {
        self.tokens
            .first()
            .and_then(|b| b.first())
            .map_or(0, Vec::len)
    }

    /// Flattens `nodes` into a single `Vec<u32>` of length
    /// `batch_size * beam_width` for bulk VNTK calls.
    pub fn flat_nodes(&self) -> Vec<u32> {
        self.nodes
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect()
    }

    /// Replaces the state from a flat representation produced by the decoder.
    /// `flat_nodes` and `flat_scores` must have length `batch_size * beam_width`.
    pub fn update_from_flat(
        &mut self,
        flat_nodes: &[u32],
        flat_scores: &[f64],
        flat_tokens: &[Vec<u32>],
    ) {
        let bw = self.beam_width();
        for (b, row) in self.nodes.iter_mut().enumerate() {
            row.copy_from_slice(&flat_nodes[b * bw..(b + 1) * bw]);
        }
        for (b, row) in self.scores.iter_mut().enumerate() {
            row.copy_from_slice(&flat_scores[b * bw..(b + 1) * bw]);
        }
        for (b, row) in self.tokens.iter_mut().enumerate() {
            for (w, toks) in row.iter_mut().enumerate() {
                *toks = flat_tokens[b * bw + w].clone();
            }
        }
    }

    /// Returns completed sequences (those whose `tokens` length == `sid_length`).
    pub fn completed(&self, sid_length: usize) -> Vec<Vec<u32>> {
        self.tokens
            .iter()
            .flat_map(|batch| batch.iter())
            .filter(|seq| seq.len() == sid_length)
            .cloned()
            .collect()
    }
}

impl fmt::Display for BeamState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BeamState(batch={}, beams={}, step={})",
            self.batch_size(),
            self.beam_width(),
            self.step(),
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// VntkOutput  (consumed by decoder.rs and the test module)
// ──────────────────────────────────────────────────────────────────────────────

/// Output of a single VNTK call for one decoding step.
#[derive(Debug, Clone)]
pub struct VntkOutput {
    /// Next trie-node IDs, one per valid (beam, child) slot.
    /// Length = `beam_width * max_branches_at_level` (padded with sentinel 0).
    pub next_nodes: Vec<u32>,

    /// Dense boolean mask over the vocabulary: `mask[t]` is `true` iff token
    /// `t` is a valid next token for *at least one* active beam.
    /// Length = `vocab_size`.
    pub mask: Vec<bool>,
}
