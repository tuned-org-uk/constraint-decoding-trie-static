use std::collections::HashSet;

use crate::decoder::constrained_beam_decode;
use crate::transition::build_static_index;
use crate::types::{BeamState, StaticIndex, VntkOutput};
use crate::vntk::vntk;

// ─────────────────────────────────────────────────────────────────────────────
// Shared fixture
// ─────────────────────────────────────────────────────────────────────────────

/// Builds the exact Figure 1 index used across multiple tests.
/// C = {[1,2,1], [3,1,2], [3,1,3]}, |V|=4, L=3, d=2
fn figure1_index() -> StaticIndex {
    let constraints = vec![vec![1u32, 2, 1], vec![3, 1, 2], vec![3, 1, 3]];
    build_static_index(
        &constraints,
        /*vocab_size=*/ 4,
        /*sid_length=*/ 3,
        /*dense_depth=*/ 2,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. CSR structure — Figure 1d
// ─────────────────────────────────────────────────────────────────────────────

/// Test: 3-item constraint set from the paper's Figure 1.
/// C = {[1,2,1], [3,1,2], [3,1,3]}, |V|=4, L=3
///
/// Expected trie nodes 0..=7:
///   0 (root) →{1}→ 1, →{3}→ 2
///   1        →{2}→ 3
///   2        →{1}→ 4
///   3        →{1}→ 5            (leaf: SID [1,2,1])
///   4        →{2}→ 6, →{3}→ 7  (leaves: [3,1,2] and [3,1,3])
///   5, 6, 7  leaf nodes (no transitions)
#[test]
fn test_paper_example() {
    let index = figure1_index();

    // ── total node count ───────────────────────────────────────────────────
    assert_eq!(
        index.sparse.num_nodes, 8,
        "CSR should contain 8 nodes as in Figure 1d"
    );

    // ── root (node 0): token 1 → node 1,  token 3 → node 2 ───────────────
    {
        let start = index.sparse.row_pointers[0] as usize;
        let end = index.sparse.row_pointers[1] as usize;
        assert_eq!(end - start, 2, "root should have 2 children");
        let ch: Vec<[u32; 2]> = index.sparse.data[start..end].to_vec();
        assert!(ch.contains(&[1, 1]), "root: token 1 → node 1");
        assert!(ch.contains(&[3, 2]), "root: token 3 → node 2");
    }

    // ── node 4: token 2 → node 6,  token 3 → node 7 ─────────────────────
    {
        let start = index.sparse.row_pointers[4] as usize;
        let end = index.sparse.row_pointers[5] as usize;
        assert_eq!(end - start, 2, "node 4 should have 2 children");
        assert_eq!(index.sparse.data[start], [2, 6], "node 4: token 2 → node 6");
        assert_eq!(
            index.sparse.data[start + 1],
            [3, 7],
            "node 4: token 3 → node 7"
        );
    }

    // ── leaf nodes 5, 6, 7: zero outgoing transitions ─────────────────────
    for leaf in [5usize, 6, 7] {
        let start = index.sparse.row_pointers[leaf] as usize;
        let end = index.sparse.row_pointers[leaf + 1] as usize;
        assert_eq!(end - start, 0, "leaf node {leaf} must have 0 children");
    }

    // ── total edge count: 7 ───────────────────────────────────────────────
    // root(2) + node1(1) + node2(1) + node3(1) + node4(2) = 7
    assert_eq!(
        index.sparse.data.len(),
        7,
        "total edge count across CSR data should be 7"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Dense mask — O(1) lookup
// ─────────────────────────────────────────────────────────────────────────────

/// Test: DenseMask correctly marks only the two valid 2-token prefixes.
/// From Figure 1a: valid 2-prefixes are (1,2) and (3,1).
#[test]
fn test_dense_mask_bitpacked() {
    let index = figure1_index();
    // Field is `index.dense`, not `index.dense_mask` — matches StaticIndex definition.
    let mask = &index.dense;

    // Valid 2-token prefixes
    assert!(mask.get(1, 2), "prefix (1,2) exists — path to SID [1,2,1]");
    assert!(
        mask.get(3, 1),
        "prefix (3,1) exists — paths to [3,1,2] and [3,1,3]"
    );

    // Every other (v1, v2) pair must be absent
    for v1 in 0..4u32 {
        for v2 in 0..4u32 {
            let expected = matches!((v1, v2), (1, 2) | (3, 1));
            assert_eq!(
                mask.get(v1, v2),
                expected,
                "dense mask mismatch at ({v1}, {v2})"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. VNTK masking
// ─────────────────────────────────────────────────────────────────────────────

/// Test: VNTK produces correct per-beam next-node IDs and OR-reduced token mask.
///
/// Both beams at node 4 (prefix [3,1]).
/// Expected: next_nodes = [6, 7, 6, 7]  (two beams × two children)
///           mask        = [F, F, T, T]
#[test]
fn test_vntk_masking() {
    let index = figure1_index();

    // Two beams, both at node 4.  `vntk` takes the flat [B*M] slice.
    let current_nodes: Vec<u32> = vec![4, 4];

    // `vntk` is the standalone function in vntk.rs with signature:
    //   fn vntk(current_nodes: &[u32], matrix: &TransitionMatrix,
    //           level: usize, vocab_size: usize) -> VntkOutput
    let VntkOutput { next_nodes, mask } = vntk(
        &current_nodes,
        &index.sparse,
        /*level=*/ 2,
        /*vocab_size=*/ 4,
    );

    // Both beams yield the same two children → flat list [6, 7, 6, 7].
    assert_eq!(
        next_nodes.len(),
        4,
        "two beams × two children = 4 next-node entries"
    );
    assert_eq!(next_nodes[0], 6, "beam 0, child 0 → node 6");
    assert_eq!(next_nodes[1], 7, "beam 0, child 1 → node 7");
    assert_eq!(next_nodes[2], 6, "beam 1, child 0 → node 6");
    assert_eq!(next_nodes[3], 7, "beam 1, child 1 → node 7");

    // OR-reduced dense mask over both beams: tokens 2 and 3 valid.
    assert_eq!(mask.len(), 4);
    assert!(!mask[0], "token 0 invalid at node 4");
    assert!(!mask[1], "token 1 invalid at node 4");
    assert!(mask[2], "token 2 valid at node 4 (→ node 6)");
    assert!(mask[3], "token 3 valid at node 4 (→ node 7)");
}

/// Test: VNTK at a leaf node returns an all-false mask (no valid extensions).
///
/// Node 5 is the leaf for SID [1,2,1].  At level 2 (the last sparse level
/// for L=3, d=2) max_branches[2] gives the branch size; all slots are empty.
#[test]
fn test_vntk_leaf_node_all_masked() {
    let index = figure1_index();

    let VntkOutput {
        next_nodes: _,
        mask,
    } = vntk(
        &[5u32],
        &index.sparse,
        /*level=*/ 2,
        /*vocab_size=*/ 4,
    );

    assert!(
        mask.iter().all(|&m| !m),
        "leaf node 5 must produce an all-false mask"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. End-to-end constrained decoding
// ─────────────────────────────────────────────────────────────────────────────

/// Test: Full decode with uniform logits recovers exactly the constraint set.
#[test]
fn test_end_to_end_constrained_decoding() {
    let constraints: Vec<Vec<u32>> = vec![vec![1, 2, 1], vec![3, 1, 2], vec![3, 1, 3]];
    let index = build_static_index(&constraints, 4, 3, 2);

    // `constrained_beam_decode` lives in decoder.rs and has signature:
    //   fn constrained_beam_decode(
    //       index: &StaticIndex, flat_logits: &[f32],
    //       sid_length: usize, beam_width: usize,
    //   ) -> Vec<Vec<u32>>
    let results = constrained_beam_decode(
        &index,
        &[0.0f32; 4], // uniform — all constrained paths equally likely
        /*sid_length=*/ 3,
        /*beam_width=*/ 3,
    );

    assert_eq!(
        results.len(),
        3,
        "beam search must return exactly beam_width results"
    );

    let result_set: HashSet<Vec<u32>> = results.into_iter().collect();
    let expected: HashSet<Vec<u32>> = constraints.into_iter().collect();
    assert_eq!(
        result_set, expected,
        "decoded SIDs must exactly match the constraint set C"
    );
}

/// Test: Hard constraint overrides a strongly biased model.
#[test]
fn test_decoding_respects_hard_constraints() {
    let subset = vec![vec![3u32, 1, 2]];
    let index = build_static_index(&subset, 4, 3, 2);

    // Token 1 has logit 100 — model overwhelmingly prefers [1, …].
    let biased = vec![0.0f32, 100.0, 0.0, 0.0];
    let results = constrained_beam_decode(&index, &biased, 3, 1);

    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0],
        vec![3u32, 1, 2],
        "hard constraint must override model's token preference"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Persistence — byte round-trip for DenseMask + full StaticIndex
// ─────────────────────────────────────────────────────────────────────────────

/// Test: DenseMask survives a to_bytes / from_bytes round-trip.
///
/// `StaticIndex` does not yet have Parquet I/O; we test the byte-level
/// serialisation defined in dense_mask.rs, and a manual CSR comparison.
#[test]
fn test_static_index_persistence() {
    let original = figure1_index();

    // ── DenseMask byte round-trip (dense_mask.rs) ─────────────────────────
    let dm_bytes = original.dense.to_bytes();
    let restored_dm = crate::types::DenseMask::from_bytes(&dm_bytes)
        .expect("DenseMask deserialisation must succeed");

    for v1 in 0..4u32 {
        for v2 in 0..4u32 {
            assert_eq!(
                original.dense.get(v1, v2),
                restored_dm.get(v1, v2),
                "dense mask mismatch at ({v1},{v2}) after round-trip"
            );
        }
    }
    assert_eq!(
        original.dense.states, restored_dm.states,
        "dense states must survive round-trip"
    );

    // ── CSR structural equality (no Parquet dependency) ───────────────────
    // Re-build from the same constraints and verify field-by-field equality.
    let constraints = vec![vec![1u32, 2, 1], vec![3, 1, 2], vec![3, 1, 3]];
    let rebuilt = build_static_index(&constraints, 4, 3, 2);

    assert_eq!(
        original.sparse.row_pointers, rebuilt.sparse.row_pointers,
        "row_pointers must be deterministic across builds"
    );
    assert_eq!(
        original.sparse.data, rebuilt.sparse.data,
        "CSR data must be deterministic across builds"
    );
    assert_eq!(
        original.sparse.num_nodes, rebuilt.sparse.num_nodes,
        "num_nodes must be deterministic"
    );
    assert_eq!(original.num_constraints, rebuilt.num_constraints);
}
