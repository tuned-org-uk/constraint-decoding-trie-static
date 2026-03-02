// src/dense_mask.rs

use crate::types::DenseMask;
use rayon::prelude::*;

// ──────────────────────────────────────────────────────────────────────────────
// Construction
// ──────────────────────────────────────────────────────────────────────────────

impl DenseMask {
    // ------------------------------------------------------------------
    // Bulk construction from a constraint set
    // ------------------------------------------------------------------

    /// Build a `DenseMask` directly from a full constraint set.
    ///
    /// This is the canonical constructor used by `build_static_index`.
    /// It is equivalent to calling `DenseMask::new` followed by repeated
    /// `insert` calls, but avoids recomputing `flat_index` twice per entry.
    ///
    /// # Arguments
    /// - `constraints` — every sequence must have length ≥ `depth`
    /// - `vocab_size`  — |V|
    /// - `depth`       — number of dense layers d (typically 2)
    /// - `node_ids`    — parallel slice: `node_ids[i]` is the trie node reached
    ///                   after the first `depth` tokens of `constraints[i]`.
    ///                   Pass an all-zeros slice when node IDs are not yet known
    ///                   and will be back-filled by `transition.rs`.
    pub fn from_constraints(
        constraints: &[Vec<u32>],
        vocab_size: u32,
        depth: u32,
        node_ids: &[u32],
    ) -> Self {
        debug_assert_eq!(
            constraints.len(),
            node_ids.len(),
            "constraints and node_ids must have equal length"
        );

        let mut mask = DenseMask::new(vocab_size, depth);

        for (seq, &nid) in constraints.iter().zip(node_ids.iter()) {
            debug_assert!(
                seq.len() >= depth as usize,
                "sequence too short for dense depth {depth}"
            );
            mask.insert(&seq[..depth as usize], nid);
        }

        mask
    }

    // ------------------------------------------------------------------
    // Prefix validity — O(word) scan over packed bits
    // ------------------------------------------------------------------

    /// Returns `true` if **any** full-depth prefix starts with `first_token`.
    ///
    /// Operates on packed `u64` words without deserialising individual bits.
    /// Used by `decoder.rs` at step 0 to expose the valid first-token set.
    ///
    /// # Complexity
    /// O(|V|^(depth-1) / 64)  ≈  O(1) for small depth and typical |V|.
    pub fn first_token_valid(&self, first_token: u32) -> bool {
        let (base, end) = self.token_block_range(first_token, 0);
        self.any_bit_set_in(base, end)
    }

    /// Returns `true` if `partial` (length < `depth`) can be extended to a
    /// valid full-depth prefix in the constraint set.
    ///
    /// # Panics (debug)
    /// Panics if `partial.len() >= depth`.
    pub fn partial_prefix_has_extension(&self, partial: &[u32]) -> bool {
        debug_assert!(
            partial.len() < self.depth as usize,
            "partial prefix length {} must be < depth {}",
            partial.len(),
            self.depth
        );
        let flat_base: usize = partial.iter().fold(0usize, |acc, &t| {
            acc * self.vocab_size as usize + t as usize
        });
        let stride = (self.vocab_size as usize).pow((self.depth as usize - partial.len()) as u32);
        let base = flat_base * stride;
        let end = base + stride;
        self.any_bit_set_in(base, end)
    }

    // ------------------------------------------------------------------
    // Bit-parallel intersection
    // ------------------------------------------------------------------

    /// Returns a new `DenseMask` that is the intersection of `self` and `other`.
    ///
    /// Two masks can be intersected to find the set of prefixes that appear in
    /// **both** constraint sets — useful for multi-constraint filtering.
    ///
    /// # Panics
    /// Panics if `self` and `other` have different `vocab_size` or `depth`.
    pub fn intersect(&self, other: &DenseMask) -> DenseMask {
        assert_eq!(
            self.vocab_size, other.vocab_size,
            "vocab_size mismatch in intersect"
        );
        assert_eq!(self.depth, other.depth, "depth mismatch in intersect");

        let bits: Vec<u64> = self
            .bits
            .iter()
            .zip(other.bits.iter())
            .map(|(&a, &b)| a & b)
            .collect();

        // Zero out states entries whose bit was cleared by the intersection.
        let total = (self.vocab_size as usize).pow(self.depth);
        let mut states = vec![0u32; total];
        for idx in 0..total {
            if (bits[idx / 64] >> (idx % 64)) & 1 == 1 {
                states[idx] = self.states[idx];
            }
        }

        DenseMask {
            bits,
            states,
            depth: self.depth,
            vocab_size: self.vocab_size,
        }
    }

    /// Returns a new `DenseMask` that is the union of `self` and `other`.
    ///
    /// Used when merging two separately-built index shards.
    /// Where both masks have a valid entry, `self`'s node ID takes precedence.
    ///
    /// # Panics
    /// Panics if `vocab_size` or `depth` differ.
    pub fn union(&self, other: &DenseMask) -> DenseMask {
        assert_eq!(self.vocab_size, other.vocab_size);
        assert_eq!(self.depth, other.depth);

        let bits: Vec<u64> = self
            .bits
            .iter()
            .zip(other.bits.iter())
            .map(|(&a, &b)| a | b)
            .collect();

        let total = (self.vocab_size as usize).pow(self.depth);
        let mut states = other.states.clone(); // start with other's node IDs
        for idx in 0..total {
            // Self takes priority where self has the bit set.
            if (self.bits[idx / 64] >> (idx % 64)) & 1 == 1 {
                states[idx] = self.states[idx];
            }
        }

        DenseMask {
            bits,
            states,
            depth: self.depth,
            vocab_size: self.vocab_size,
        }
    }

    // ------------------------------------------------------------------
    // Packed-bit mask extraction (for logit gating)
    // ------------------------------------------------------------------

    /// Returns the first-token marginal as a packed `Vec<u64>` of length
    /// `ceil(vocab_size / 64)`.
    ///
    /// Bit `t` is set iff token `t` is a valid first token in the constraint
    /// set.  This vec can be ANDed directly with the model's top-k bitmask.
    pub fn first_token_packed_mask(&self) -> Vec<u64> {
        let v = self.vocab_size as usize;
        let words = v.div_ceil(64);
        let mut out = vec![0u64; words];
        for tok in 0..v as u32 {
            if self.first_token_valid(tok) {
                let idx = tok as usize;
                out[idx / 64] |= 1u64 << (idx % 64);
            }
        }
        out
    }

    /// Returns the second-token marginal **given** that `first_token` was chosen,
    /// packed as a `Vec<u64>` of length `ceil(vocab_size / 64)`.
    ///
    /// Only defined for `depth >= 2`.
    ///
    /// # Panics (debug)
    /// Panics if `depth < 2`.
    pub fn second_token_packed_mask(&self, first_token: u32) -> Vec<u64> {
        debug_assert!(
            self.depth >= 2,
            "second_token_packed_mask requires depth >= 2"
        );
        let v = self.vocab_size as usize;
        let words = v.div_ceil(64);
        let mut out = vec![0u64; words];
        for tok2 in 0..v as u32 {
            if self.get(first_token, tok2) {
                let idx = tok2 as usize;
                out[idx / 64] |= 1u64 << (idx % 64);
            }
        }
        out
    }

    // ------------------------------------------------------------------
    // Count helpers
    // ------------------------------------------------------------------

    /// Returns the total number of valid prefixes stored in the mask.
    pub fn count_valid(&self) -> u64 {
        self.bits.iter().map(|w| w.count_ones() as u64).sum()
    }

    /// Returns the number of distinct valid first tokens.
    pub fn count_valid_first_tokens(&self) -> u32 {
        (0..self.vocab_size)
            .filter(|&t| self.first_token_valid(t))
            .count() as u32
    }

    // ------------------------------------------------------------------
    // Serialisation helpers (used by persistence tests)
    // ------------------------------------------------------------------

    /// Serialises the mask into a flat byte buffer.
    ///
    /// Layout (little-endian):
    /// ```text
    /// [u32 vocab_size][u32 depth]
    /// [u32 bits_len][u64 * bits_len]
    /// [u32 states_len][u32 * states_len]
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.vocab_size.to_le_bytes());
        out.extend_from_slice(&self.depth.to_le_bytes());
        out.extend_from_slice(&(self.bits.len() as u32).to_le_bytes());
        for &w in &self.bits {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out.extend_from_slice(&(self.states.len() as u32).to_le_bytes());
        for &s in &self.states {
            out.extend_from_slice(&s.to_le_bytes());
        }
        out
    }

    /// Deserialises a `DenseMask` from the byte layout produced by `to_bytes`.
    ///
    /// Returns `None` if the buffer is malformed.
    pub fn from_bytes(buf: &[u8]) -> Option<Self> {
        let mut cur = 0usize;

        let read_u32 = |buf: &[u8], pos: &mut usize| -> Option<u32> {
            let bytes = buf.get(*pos..*pos + 4)?;
            *pos += 4;
            Some(u32::from_le_bytes(bytes.try_into().ok()?))
        };
        let read_u64 = |buf: &[u8], pos: &mut usize| -> Option<u64> {
            let bytes = buf.get(*pos..*pos + 8)?;
            *pos += 8;
            Some(u64::from_le_bytes(bytes.try_into().ok()?))
        };

        let vocab_size = read_u32(buf, &mut cur)?;
        let depth = read_u32(buf, &mut cur)?;
        let bits_len = read_u32(buf, &mut cur)? as usize;

        let mut bits = Vec::with_capacity(bits_len);
        for _ in 0..bits_len {
            bits.push(read_u64(buf, &mut cur)?);
        }

        let states_len = read_u32(buf, &mut cur)? as usize;
        let mut states = Vec::with_capacity(states_len);
        for _ in 0..states_len {
            states.push(read_u32(buf, &mut cur)?);
        }

        Some(DenseMask {
            bits,
            states,
            depth,
            vocab_size,
        })
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Returns the `[base, end)` range of flat indices covered by the block
    /// rooted at `token` appearing at position `pos` in the prefix.
    fn token_block_range(&self, token: u32, pos: usize) -> (usize, usize) {
        let v = self.vocab_size as usize;
        let d = self.depth as usize;
        let stride = v.pow((d - pos - 1) as u32);
        let base = token as usize * stride;
        (base, base + stride)
    }

    /// Returns `true` if any bit in flat-index range `[base, end)` is set.
    // In dense_mask.rs
    /// Returns `true` if any bit in flat-index range `[base, end)` is set.
    ///
    /// This implementation correctly handles ranges that span multiple 64-bit
    /// words as well as ranges contained within a single word.
    #[inline]
    fn any_bit_set_in(&self, base: usize, end: usize) -> bool {
        if base >= end {
            return false;
        }

        let w_start = base / 64;
        let w_end = (end - 1) / 64; // index of the last word touched

        // Safety bounds check
        if w_start >= self.bits.len() {
            return false;
        }
        let actual_w_end = w_end.min(self.bits.len() - 1);

        for w_idx in w_start..=actual_w_end {
            let mut val = self.bits[w_idx];

            // 1. Mask out bits BEFORE the range in the start word
            if w_idx == w_start {
                let shift = base % 64;
                val &= !0u64 << shift;
            }

            // 2. Mask out bits AFTER the range in the end word
            // This is applied independently so it works even if w_start == w_end.
            if w_idx == w_end {
                let limit = end % 64;
                if limit != 0 {
                    // mask has bits 0..limit-1 set
                    let mask = (1u64 << limit) - 1;
                    val &= mask;
                }
            }

            if val != 0 {
                return true;
            }
        }
        false
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Parallel bulk validation  (used by transition.rs integration tests)
// ──────────────────────────────────────────────────────────────────────────────

/// Validates a batch of token prefixes against the mask in parallel.
///
/// Returns a `Vec<bool>` of length `prefixes.len()` where `true` means the
/// prefix is present in the constraint set.
pub fn validate_prefixes(mask: &DenseMask, prefixes: &[Vec<u32>]) -> Vec<bool> {
    prefixes.par_iter().map(|p| mask.contains(p)).collect()
}

/// Converts a `DenseMask` into a flat `Vec<u64>` token-level marginal mask
/// for the given prefix position and preceding token sequence.
///
/// Returns a packed bitmask of length `ceil(vocab_size / 64)` whose bit `t`
/// is set iff appending token `t` to `prefix_so_far` yields a valid (partial
/// or complete) prefix in the mask.
pub fn marginal_mask_at(mask: &DenseMask, prefix_so_far: &[u32]) -> Vec<u64> {
    let len = prefix_so_far.len();
    let v = mask.vocab_size as usize;
    let depth = mask.depth as usize;
    let words = v.div_ceil(64);

    assert!(
        len < depth,
        "prefix_so_far length {len} must be < depth {depth}"
    );

    let mut out = vec![0u64; words];
    for tok in 0..v as u32 {
        let mut candidate = prefix_so_far.to_vec();
        candidate.push(tok);
        let valid = if candidate.len() == depth {
            mask.contains(&candidate)
        } else {
            mask.partial_prefix_has_extension(&candidate)
        };
        if valid {
            out[tok as usize / 64] |= 1u64 << (tok as usize % 64);
        }
    }
    out
}
