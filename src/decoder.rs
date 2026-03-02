// src/decoder.rs

use rayon::prelude::*;

use crate::types::{BeamState, StaticIndex, VntkOutput};
use crate::vntk::VntkResult;

// ──────────────────────────────────────────────────────────────────────────────
// Top-level decoder struct
// ──────────────────────────────────────────────────────────────────────────────

pub struct ConstrainedDecoder {
    pub index: StaticIndex,
    pub beam_width: usize, // M
    pub batch_size: usize, // B
}

impl ConstrainedDecoder {
    pub fn new(index: StaticIndex, beam_width: usize, batch_size: usize) -> Self {
        Self {
            index,
            beam_width,
            batch_size,
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Public: single decoding step  (Algorithm 1, one iteration)
    // ──────────────────────────────────────────────────────────────────────────

    /// Execute one constrained decoding step.
    ///
    /// # Arguments
    /// - `logits` — raw model outputs, shape \[B × M × |V|\]
    /// - `state`  — mutable beam state (nodes, scores, partial tokens)
    /// - `step`   — 0-indexed decoding step t
    pub fn step(
        &self,
        logits: &[Vec<Vec<f64>>], // [B][M][|V|]
        state: &mut BeamState,
        step: usize,
    ) {
        let vocab = self.index.sparse.vocab_size as usize;
        let b = self.batch_size;
        let m = self.beam_width;

        debug_assert_eq!(logits.len(), b);
        debug_assert!(logits.iter().all(|q| q.len() == m));
        debug_assert!(logits.iter().all(|q| q.iter().all(|bm| bm.len() == vocab)));

        // ── Phase 1: LogSoftmax ───────────────────────────────────────────────
        let log_probs = log_softmax_3d(logits);

        // ── Phase 2: Constraint masking ───────────────────────────────────────
        // Returns:
        //   masks      : [B][M][|V|]  — true = token is valid
        //   next_nodes : [B][M][B_t]  — trie nodes after each valid token slot
        let (masks, next_nodes) = if step < self.index.dense.depth as usize {
            self.dense_lookup(state, step)
        } else {
            self.sparse_lookup(state, step)
        };

        // ── Phase 3: Apply mask → NEG_INF for invalid tokens ─────────────────
        let masked = apply_mask(&log_probs, &masks);

        // ── Phase 4: Beam search selection ───────────────────────────────────
        // new_tokens  : [B][M]      — chosen token per surviving beam
        // new_scores  : [B][M]      — updated cumulative log-prob
        // src_beams   : [B][M]      — which old beam each new beam came from
        let (new_tokens, new_scores, src_beams) = beam_search(&masked, &state.scores, m);

        // ── Phase 5: State gather ─────────────────────────────────────────────
        self.gather_state(
            state,
            &new_tokens,
            &new_scores,
            &src_beams,
            &next_nodes,
            step,
        );
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Public: full decoding loop  (Algorithm 1 complete)
    // ──────────────────────────────────────────────────────────────────────────

    /// Run the full constrained beam-search loop for `sid_length` steps.
    ///
    /// `logit_fn` is called once per step; it receives the current `BeamState`
    /// and must return logits of shape `[B × M × |V|]`.
    ///
    /// Returns the top-`beam_width` decoded SIDs for every query in the batch.
    pub fn decode<F>(&self, logit_fn: F, sid_length: usize) -> Vec<Vec<Vec<u32>>>
    // [B][M][L]
    where
        F: Fn(&BeamState, usize) -> Vec<Vec<Vec<f64>>>,
    {
        let mut state = BeamState::new(self.batch_size, self.beam_width);

        for step in 0..sid_length {
            let logits = logit_fn(&state, step);
            self.step(&logits, &mut state, step);
        }

        // Return the token sequences accumulated in state.
        state.tokens.clone()
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Phase 2a: dense lookup  (steps 0 .. dense_depth−1)
    // ──────────────────────────────────────────────────────────────────────────

    /// For steps covered by the bit-packed dense mask, look up validity in O(1)
    /// per token without touching the CSR matrix.
    ///
    /// Returns `(masks, next_nodes)` shaped `[B][M][|V|]` and `[B][M][1]`
    /// respectively (one "next node" per beam; the trie node reached after the
    /// chosen token is resolved lazily in `gather_state` from the dense mask's
    /// `states` array).
    pub fn dense_lookup(
        &self,
        state: &BeamState,
        step: usize,
    ) -> (Vec<Vec<Vec<bool>>>, Vec<Vec<Vec<u32>>>) {
        let vocab = self.index.sparse.vocab_size as usize;
        let depth = self.index.dense.depth as usize;
        let b = self.batch_size;
        let m = self.beam_width;

        debug_assert!(step < depth, "dense_lookup called outside dense range");
        debug_assert!(depth >= 1);

        // masks[b][m][v] = token validity at this step for each beam
        let mut masks: Vec<Vec<Vec<bool>>> = vec![vec![vec![false; vocab]; m]; b];

        // next_nodes is not used for dense steps in our gather logic; keep shape stable.
        let next_nodes: Vec<Vec<Vec<u32>>> = vec![vec![vec![0u32; 1]; m]; b];

        for bi in 0..b {
            for mi in 0..m {
                let prev = &state.tokens[bi][mi];
                debug_assert_eq!(prev.len(), step);

                if step == 0 {
                    // Step 0: allow tokens that start at least one valid dense prefix.
                    for tok in 0..vocab as u32 {
                        if self.index.dense.first_token_valid(tok) {
                            masks[bi][mi][tok as usize] = true;
                        }
                    }
                    continue;
                }

                // step >= 1: extend the prefix by one candidate token and test
                for tok in 0..vocab as u32 {
                    let mut candidate = prev.clone();
                    candidate.push(tok);

                    let valid = if candidate.len() == depth {
                        // Boundary case: full dense prefix, must be exact membership
                        self.index.dense.contains(&candidate)
                    } else {
                        // Proper partial prefix
                        self.index.dense.partial_prefix_has_extension(&candidate)
                    };

                    if valid {
                        masks[bi][mi][tok as usize] = true;
                    }
                }
            }
        }

        (masks, next_nodes)
    }

    /// Returns true if *any* full-depth dense entry starts with `tok`.
    #[inline]
    pub fn dense_first_token_valid(&self, tok: u32) -> bool {
        self.index.dense.first_token_valid(tok)
    }

    /// Returns true if `partial_prefix` (length < depth) can be extended to a
    /// valid full-depth prefix.
    fn dense_prefix_has_extension(&self, partial_prefix: &[u32]) -> bool {
        let vocab = self.index.sparse.vocab_size as usize;
        let depth = self.index.dense.depth as usize;
        let len = partial_prefix.len();
        debug_assert!(len < depth);

        // Flat index of the first entry in the block covered by partial_prefix.
        let block_start: usize = partial_prefix
            .iter()
            .fold(0usize, |acc, &t| acc * vocab + t as usize);
        let stride = vocab.pow((depth - len) as u32);
        let base = block_start * stride;
        let end = base + stride;
        let ws = base / 64;
        let we = end.div_ceil(64).min(self.index.dense.bits.len());
        self.index.dense.bits[ws..we].iter().any(|&w| w != 0)
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Phase 2b: sparse lookup  (steps dense_depth .. L−1)
    // ──────────────────────────────────────────────────────────────────────────

    /// For deeper steps, call VNTK on the CSR transition matrix.
    ///
    /// Returns `(masks, next_nodes)` shaped `[B][M][|V|]` and `[B][M][B_t]`.
    pub fn sparse_lookup(
        &self,
        state: &BeamState,
        step: usize,
    ) -> (Vec<Vec<Vec<bool>>>, Vec<Vec<Vec<u32>>>) {
        let vocab = self.index.sparse.vocab_size as usize;
        let b = self.batch_size;
        let m = self.beam_width;
        let b_t = self.index.sparse.max_branches[step] as usize;

        // Flatten [B][M] nodes into a single slice for a single VNTK call.
        let flat_nodes: Vec<u32> = state.nodes.iter().flatten().copied().collect();

        let result = self.index.sparse.vntk(&flat_nodes, step);

        // Reshape VntkResult back to [B][M][…]
        let mut masks: Vec<Vec<Vec<bool>>> = vec![vec![vec![false; vocab]; m]; b];
        let mut next_nodes: Vec<Vec<Vec<u32>>> = vec![vec![vec![0u32; b_t]; m]; b];

        for bi in 0..b {
            for mi in 0..m {
                let flat_i = bi * m + mi;

                // Dense mask slice → masks[bi][mi][*]
                let mask_slice = result.mask_for(flat_i, vocab);
                masks[bi][mi].copy_from_slice(mask_slice);

                // Next-node slots → next_nodes[bi][mi][*]
                let base = flat_i * b_t;
                next_nodes[bi][mi].copy_from_slice(&result.next_nodes[base..base + b_t]);
            }
        }

        (masks, next_nodes)
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Phase 5: state gather
    // ──────────────────────────────────────────────────────────────────────────

    /// Applies the beam-search selection to the live `BeamState`.
    ///
    /// For each surviving beam in each batch entry:
    /// 1. Copy the partial token sequence from the *source* beam.
    /// 2. Append the newly chosen token.
    /// 3. Advance the trie node pointer using `next_nodes`.
    /// Applies the beam-search selection to the live `BeamState`.
    ///
    /// This implementation handles the transition from dense "prefix-only"
    /// tracking to sparse "trie-node" tracking once the prefix length
    /// matches `index.dense.depth`.
    fn gather_state(
        &self,
        state: &mut BeamState,
        new_tokens: &[Vec<u32>],      // [B][M]
        new_scores: &[Vec<f64>],      // [B][M]
        src_beams: &[Vec<usize>],     // [B][M] — source beam for each new beam
        next_nodes: &[Vec<Vec<u32>>], // [B][M][B_t] — from VNTK
        step: usize,
    ) {
        let b = self.batch_size;
        let m = self.beam_width;
        let depth = self.index.dense.depth as usize;

        // Snapshot current state to avoid reading partially updated sequences.
        let old_tokens: Vec<Vec<Vec<u32>>> = state.tokens.clone();
        let old_nodes: Vec<Vec<u32>> = state.nodes.clone();

        for bi in 0..b {
            for mi in 0..m {
                let src_idx = src_beams[bi][mi];
                let chosen_token = new_tokens[bi][mi];

                // 1. Update cumulative score
                state.scores[bi][mi] = new_scores[bi][mi];

                // 2. Extend the sequence (copy-on-write from source beam)
                let mut seq = old_tokens[bi][src_idx].clone();
                seq.push(chosen_token);
                state.tokens[bi][mi] = seq;

                // 3. Advance the trie node
                // step 0 creates a 1-token prefix; step (depth-1) creates a depth-token prefix.
                let current_len = step + 1;

                state.nodes[bi][mi] = if current_len < depth {
                    // Phase A: Still in dense marginalization territory.
                    // We don't have enough tokens to look up a specific trie node yet.
                    0
                } else if current_len == depth {
                    // Phase B: Boundary reached.
                    // Use the bit-packed DenseMask to find the trie node starting the sparse layer.
                    let prefix = &state.tokens[bi][mi];
                    self.index.dense.state_for(prefix).unwrap_or_else(|| {
                        debug_assert!(false, "Prefix {:?} missing in dense mask", prefix);
                        0
                    })
                } else {
                    // Phase C: Deep sparse layer traversal using VNTK.
                    self.resolve_next_node(
                        old_nodes[bi][src_idx],
                        chosen_token,
                        &next_nodes[bi][src_idx],
                        step,
                    )
                };
            }
        }
    }

    /// Resolves the next trie node for a beam that chose `token` at `step`,
    /// given the pre-computed `next_node_slots` from VNTK.
    ///
    /// VNTK returns slots sorted by token ID, so we binary-search rather than
    /// doing a linear scan or a second CSR lookup.
    pub fn resolve_next_node(
        &self,
        current_node: u32,
        token: u32,
        next_node_slots: &[u32], // length B_t, parallel to sorted children
        step: usize,
    ) -> u32 {
        // Children are sorted by token ID; binary-search for `token`.
        let children = self.index.sparse.children(current_node);
        match children.binary_search_by_key(&token, |&[t, _]| t) {
            Ok(pos) if pos < next_node_slots.len() => next_node_slots[pos],
            // Fallback: direct CSR lookup (should not happen in correct usage).
            Ok(pos) => children[pos][1],
            Err(_) => {
                debug_assert!(
                    false,
                    "token {token} not found in children of node {current_node}"
                );
                0
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Pure helper functions
// ──────────────────────────────────────────────────────────────────────────────

/// Numerically stable log-softmax over the last axis.
/// Input / output shape: `[B][M][|V|]`.
pub fn log_softmax_3d(logits: &[Vec<Vec<f64>>]) -> Vec<Vec<Vec<f64>>> {
    logits
        .par_iter()
        .map(|query| query.iter().map(|beam| log_softmax_1d(beam)).collect())
        .collect()
}

/// Numerically stable log-softmax over a single 1-D slice.
pub fn log_softmax_1d(x: &[f64]) -> Vec<f64> {
    let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let log_sum_exp = x.iter().map(|&v| (v - max).exp()).sum::<f64>().ln();
    x.iter().map(|&v| v - max - log_sum_exp).collect()
}

/// Applies a boolean constraint mask to log-probabilities.
/// Invalid tokens (mask == false) are set to `f64::NEG_INFINITY`.
/// Input / output shape: `[B][M][|V|]`.
pub fn apply_mask(log_probs: &[Vec<Vec<f64>>], masks: &[Vec<Vec<bool>>]) -> Vec<Vec<Vec<f64>>> {
    log_probs
        .par_iter()
        .zip(masks.par_iter())
        .map(|(q_lp, q_mask)| {
            q_lp.iter()
                .zip(q_mask.iter())
                .map(|(beam_lp, beam_mask)| {
                    beam_lp
                        .iter()
                        .zip(beam_mask.iter())
                        .map(|(&lp, &valid)| if valid { lp } else { f64::NEG_INFINITY })
                        .collect()
                })
                .collect()
        })
        .collect()
}

/// Beam search selection over masked log-probabilities.
///
/// Scores are accumulated as `parent_score + log_prob(token)`.
///
/// Returns `(new_tokens, new_scores, src_beams)`, all shaped `[B][M]`.
pub fn beam_search(
    masked_log_probs: &[Vec<Vec<f64>>], // [B][M][|V|]
    parent_scores: &[Vec<f64>],         // [B][M]
    beam_width: usize,
) -> (Vec<Vec<u32>>, Vec<Vec<f64>>, Vec<Vec<usize>>) {
    let b = masked_log_probs.len();

    // Process each query in the batch independently and in parallel.
    let results: Vec<_> = (0..b)
        .into_par_iter()
        .map(|bi| {
            let lp = &masked_log_probs[bi]; // [M][|V|]
            let par = &parent_scores[bi]; // [M]
            let vocab = lp[0].len();
            let m = lp.len();

            // Enumerate all (beam, token) candidates and score them.
            let mut candidates: Vec<(f64, usize, u32)> = // (score, src_beam, token)
                (0..m)
                    .flat_map(|mi| {
                        (0..vocab).filter_map(move |v| {
                            let lp_val = lp[mi][v];
                            if lp_val.is_finite() {
                                Some((par[mi] + lp_val, mi, v as u32))
                            } else {
                                None
                            }
                        })
                    })
                    .collect();

            // Partial-sort: keep top `beam_width` by descending score.
            candidates.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            candidates.truncate(beam_width);

            // Separate into parallel vecs.
            let new_scores: Vec<f64> = candidates.iter().map(|c| c.0).collect();
            let src_beams: Vec<usize> = candidates.iter().map(|c| c.1).collect();
            let new_tokens: Vec<u32> = candidates.iter().map(|c| c.2).collect();

            (new_tokens, new_scores, src_beams)
        })
        .collect();

    let new_tokens: Vec<Vec<u32>> = results.iter().map(|r| r.0.clone()).collect();
    let new_scores: Vec<Vec<f64>> = results.iter().map(|r| r.1.clone()).collect();
    let src_beams: Vec<Vec<usize>> = results.iter().map(|r| r.2.clone()).collect();

    (new_tokens, new_scores, src_beams)
}

// ──────────────────────────────────────────────────────────────────────────────
// Public convenience: full decode from flat uniform logits (used by tests)
// ──────────────────────────────────────────────────────────────────────────────

/// Runs the full decode loop using a *static* flat logit vector (same logits
/// repeated for every batch entry, beam, and step).  Useful for unit tests
/// where the model is not available.
pub fn constrained_beam_decode(
    index: &StaticIndex,
    flat_logits: &[f32], // length = vocab_size
    sid_length: usize,
    beam_width: usize,
) -> Vec<Vec<u32>> {
    let vocab = index.sparse.vocab_size as usize;
    let logits_f64: Vec<f64> = flat_logits.iter().map(|&v| v as f64).collect();
    // Shape: [1][beam_width][vocab_size]
    let logits_3d = vec![vec![logits_f64; beam_width]];

    let decoder = ConstrainedDecoder::new(index.clone(), beam_width, 1);
    let sequences = decoder.decode(|_state, _step| logits_3d.clone(), sid_length);

    sequences.into_iter().next().unwrap_or_default()
}
