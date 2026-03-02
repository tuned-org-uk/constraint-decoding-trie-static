use crate::dense_mask::{marginal_mask_at, validate_prefixes};
use crate::transition::build_static_index;
use crate::types::DenseMask;

fn paper_mask() -> DenseMask {
    build_static_index(&[vec![1u32, 2, 1], vec![3, 1, 2], vec![3, 1, 3]], 4, 3, 2).dense
}

// ── Construction ─────────────────────────────────────────────────────────

#[test]
fn from_constraints_matches_transition_builder() {
    let constraints = vec![vec![1u32, 2, 1], vec![3, 1, 2], vec![3, 1, 3]];
    let from_tc = paper_mask();
    // Build with dummy node IDs — just check bit coverage matches.
    let from_fc = DenseMask::from_constraints(&constraints, 4, 2, &[0u32; 3]);
    assert_eq!(
        from_tc.bits, from_fc.bits,
        "bit patterns must match regardless of node IDs"
    );
}

#[test]
fn count_valid_equals_two() {
    // Only two valid 2-prefixes: (1,2) and (3,1)
    assert_eq!(paper_mask().count_valid(), 2);
}

#[test]
fn count_valid_first_tokens_equals_two() {
    // Tokens 1 and 3 are valid first tokens
    assert_eq!(paper_mask().count_valid_first_tokens(), 2);
}

// ── first_token_valid ─────────────────────────────────────────────────────

#[test]
fn first_token_valid_paper_example() {
    let m = paper_mask();
    assert!(!m.first_token_valid(0));
    assert!(m.first_token_valid(1));
    assert!(!m.first_token_valid(2));
    assert!(m.first_token_valid(3));
}

// ── partial_prefix_has_extension ─────────────────────────────────────────

#[test]
fn partial_prefix_depth1_matches_first_token_valid() {
    let m = paper_mask();
    for tok in 0..4u32 {
        assert_eq!(
            m.partial_prefix_has_extension(&[tok]),
            m.first_token_valid(tok),
            "single-token partial must match first_token_valid for tok={tok}"
        );
    }
}

// ── packed mask extraction ────────────────────────────────────────────────

#[test]
fn first_token_packed_mask_bits_1_and_3_set() {
    let packed = paper_mask().first_token_packed_mask();
    assert_eq!(packed[0] & (1 << 1), 1 << 1, "bit 1 must be set");
    assert_eq!(packed[0] & (1 << 3), 1 << 3, "bit 3 must be set");
    assert_eq!(packed[0] & (1 << 0), 0, "bit 0 must be clear");
    assert_eq!(packed[0] & (1 << 2), 0, "bit 2 must be clear");
}

#[test]
fn second_token_packed_mask_given_token1() {
    // After token 1, only token 2 is valid → bit 2 set, rest clear
    let packed = paper_mask().second_token_packed_mask(1);
    assert_eq!(packed[0] & (1 << 2), 1 << 2);
    assert_eq!(packed[0] & !((1u64) << 2), 0);
}

#[test]
fn second_token_packed_mask_given_token3() {
    // After token 3, only token 1 is valid → bit 1 set
    let packed = paper_mask().second_token_packed_mask(3);
    assert_eq!(packed[0] & (1 << 1), 1 << 1);
    assert_eq!(packed[0] & !((1u64) << 1), 0);
}

// ── intersect / union ─────────────────────────────────────────────────────

#[test]
fn intersect_with_self_is_identity() {
    let m = paper_mask();
    let out = m.intersect(&m);
    assert_eq!(out.bits, m.bits);
    assert_eq!(out.states, m.states);
}

#[test]
fn intersect_with_empty_is_empty() {
    let m = paper_mask();
    let empty = DenseMask::new(4, 2);
    let out = m.intersect(&empty);
    assert!(out.bits.iter().all(|&w| w == 0));
    assert_eq!(out.count_valid(), 0);
}

#[test]
fn union_with_empty_is_identity() {
    let m = paper_mask();
    let empty = DenseMask::new(4, 2);
    let out = m.union(&empty);
    assert_eq!(out.bits, m.bits);
}

#[test]
fn union_of_disjoint_masks_has_sum_of_counts() {
    // mask_a: only (1,2);  mask_b: only (3,1)
    let mut a = DenseMask::new(4, 2);
    a.insert(&[1, 2], 3);
    let mut b = DenseMask::new(4, 2);
    b.insert(&[3, 1], 4);
    let u = a.union(&b);
    assert_eq!(u.count_valid(), 2);
    assert!(u.get(1, 2));
    assert!(u.get(3, 1));
}

#[test]
fn intersect_of_disjoint_masks_is_empty() {
    let mut a = DenseMask::new(4, 2);
    a.insert(&[1, 2], 3);
    let mut b = DenseMask::new(4, 2);
    b.insert(&[3, 1], 4);
    assert_eq!(a.intersect(&b).count_valid(), 0);
}

// ── serialisation round-trip ──────────────────────────────────────────────

#[test]
fn to_bytes_from_bytes_round_trip() {
    let original = paper_mask();
    let buf = original.to_bytes();
    let restored = DenseMask::from_bytes(&buf).expect("deserialisation must succeed");
    assert_eq!(original.vocab_size, restored.vocab_size);
    assert_eq!(original.depth, restored.depth);
    assert_eq!(original.bits, restored.bits);
    assert_eq!(original.states, restored.states);
}

#[test]
fn from_bytes_truncated_returns_none() {
    let buf = paper_mask().to_bytes();
    assert!(
        DenseMask::from_bytes(&buf[..4]).is_none(),
        "truncated buffer must return None"
    );
}

// ── validate_prefixes ─────────────────────────────────────────────────────

#[test]
fn validate_prefixes_batch() {
    let m = paper_mask();
    let prefixes = vec![vec![1u32, 2], vec![3u32, 1], vec![0u32, 0], vec![2u32, 3]];
    let results = validate_prefixes(&m, &prefixes);
    assert_eq!(results, vec![true, true, false, false]);
}

// ── marginal_mask_at ──────────────────────────────────────────────────────

#[test]
fn marginal_mask_at_step0_matches_first_token_packed() {
    let m = paper_mask();
    let packed = marginal_mask_at(&m, &[]);
    let direct = m.first_token_packed_mask();
    assert_eq!(
        packed, direct,
        "marginal at step 0 must match first_token_packed_mask"
    );
}

#[test]
fn marginal_mask_at_step1_given_token1() {
    let m = paper_mask();
    let packed = marginal_mask_at(&m, &[1]);
    // Only token 2 valid → bit 2 set in word 0
    assert_eq!(packed[0] & (1 << 2), 1 << 2);
    assert_eq!(packed[0] & !((1u64) << 2), 0);
}

// ── any_bit_set_in edge cases ─────────────────────────────────────────────

#[test]
fn single_word_range_correct() {
    // vocab=8, depth=2 → 64 entries, 1 word. Insert [7,7] at flat idx 63.
    let mut m = DenseMask::new(8, 2);
    m.insert(&[7, 7], 1);
    assert!(m.first_token_valid(7));
    assert!(!m.first_token_valid(0));
}

#[test]
fn cross_word_boundary_range_correct() {
    // vocab=8, depth=2 → flat idx for [1,*] spans indices 8..15, fits in word 0.
    // For [7,*] indices 56..63 → word 0 bits 56-63.
    // For a vocab that pushes the block across two words use vocab=16, depth=2:
    //   flat size = 256, word count = 4.
    //   token 1 block = indices 16..31 (word 0 bits 16-31).
    let mut m = DenseMask::new(16, 2);
    m.insert(&[1, 0], 5);
    m.insert(&[1, 15], 6);
    assert!(m.first_token_valid(1));
    assert!(!m.first_token_valid(0));
    assert!(!m.first_token_valid(2));
}
