use crate::transition::build_static_index;
use crate::types::TransitionMatrix;
use crate::vntk::vntk;

fn paper_matrix() -> TransitionMatrix {
    build_static_index(&[vec![1u32, 2, 1], vec![3, 1, 2], vec![3, 1, 3]], 4, 3, 2).sparse
}

// ── Single-beam correctness ───────────────────────────────────────────────

#[test]
fn root_level0_two_children() {
    let tm = paper_matrix();
    let res = tm.vntk(&[0], 0); // root, level 0, B_t=2
    assert_eq!(res.branch_size, 2);

    let children: Vec<_> = res.children_for(0).collect();
    assert_eq!(children.len(), 2);
    // Tokens must be sorted: 1 then 3
    assert_eq!(children[0].0, 1);
    assert_eq!(children[1].0, 3);
}

#[test]
fn node4_level2_matches_figure1d() {
    let tm = paper_matrix();
    let res = tm.vntk(&[4], 2); // node 4, level 2

    let children: Vec<_> = res.children_for(0).collect();
    assert_eq!(children.len(), 2);
    assert_eq!(children[0], (2, 6), "token 2 → node 6");
    assert_eq!(children[1], (3, 7), "token 3 → node 7");
}

#[test]
fn dense_mask_set_for_node4() {
    let tm = paper_matrix();
    let res = tm.vntk(&[4], 2);
    let mask = res.mask_for(0, 4);
    assert!(!mask[0]);
    assert!(!mask[1]);
    assert!(mask[2], "token 2 valid at node 4");
    assert!(mask[3], "token 3 valid at node 4");
}

#[test]
fn leaf_node_all_invalid() {
    let tm = paper_matrix();
    let res = tm.vntk(&[5], 2); // node 5 is a leaf
    assert!(
        res.valid.iter().all(|&v| !v),
        "leaf must produce no valid slots"
    );
    assert!(
        res.dense_masks.iter().all(|&m| !m),
        "leaf must produce all-false dense mask"
    );
}

// ── Multi-beam correctness ────────────────────────────────────────────────

#[test]
fn two_beams_same_node_disjoint_output() {
    let tm = paper_matrix();
    // Both beams at node 4
    let res = tm.vntk(&[4, 4], 2);
    assert_eq!(res.tokens.len(), 2 * res.branch_size);
    assert_eq!(res.next_nodes.len(), 2 * res.branch_size);
    assert_eq!(res.dense_masks.len(), 2 * 4); // n * vocab_size

    // Both beams must have identical outputs
    let b0: Vec<_> = res.children_for(0).collect();
    let b1: Vec<_> = res.children_for(1).collect();
    assert_eq!(b0, b1);
}

#[test]
fn two_beams_different_nodes() {
    let tm = paper_matrix();
    // Beam 0 at root (2 children), beam 1 at a leaf (0 children)
    let res = tm.vntk(&[0, 5], 0);

    let b0: Vec<_> = res.children_for(0).collect();
    let b1: Vec<_> = res.children_for(1).collect();
    assert_eq!(b0.len(), 2, "root has 2 children");
    assert_eq!(b1.len(), 0, "leaf has 0 children");

    // Masks must not bleed between beams
    let mask0 = res.mask_for(0, 4);
    let mask1 = res.mask_for(1, 4);
    assert!(mask0[1] || mask0[3], "root mask non-empty");
    assert!(mask1.iter().all(|&m| !m), "leaf mask all-false");
}

// ── Global mask reduction ─────────────────────────────────────────────────

#[test]
fn global_mask_is_or_of_per_beam_masks() {
    let tm = paper_matrix();
    // Beam 0 at root (tokens 1,3 valid), beam 1 at node 1 (token 2 valid)
    let res = tm.vntk(&[0, 1], 0);
    let global = res.global_mask(4);
    assert!(!global[0]);
    assert!(global[1], "token 1 valid in beam 0");
    assert!(global[2], "token 2 valid in beam 1");
    assert!(global[3], "token 3 valid in beam 0");
}

// ── Packed mask ───────────────────────────────────────────────────────────

#[test]
fn packed_mask_matches_bool_mask() {
    let tm = paper_matrix();
    let res = tm.vntk(&[4], 2);
    let pack = res.packed_mask_for(0, 4);

    // token 2 → bit 2 of word 0; token 3 → bit 3 of word 0
    assert_eq!(pack[0] & (1 << 2), 1 << 2, "bit 2 set");
    assert_eq!(pack[0] & (1 << 3), 1 << 3, "bit 3 set");
    assert_eq!(pack[0] & (1 << 0), 0, "bit 0 clear");
    assert_eq!(pack[0] & (1 << 1), 0, "bit 1 clear");
}

// ── vntk_single convenience wrapper ──────────────────────────────────────

#[test]
fn vntk_single_matches_vntk_slice() {
    let tm = paper_matrix();
    let out = tm.vntk_single(4, 2);
    assert_eq!(out.next_nodes, vec![6, 7]);
    assert_eq!(out.mask, vec![false, false, true, true]);
}

// ── Standalone function form ──────────────────────────────────────────────

#[test]
fn standalone_vntk_function() {
    let tm = paper_matrix();
    let out = vntk(&[4, 4], &tm, 2, 4);
    // global mask: tokens 2 and 3 valid
    assert!(!out.mask[0]);
    assert!(!out.mask[1]);
    assert!(out.mask[2]);
    assert!(out.mask[3]);
    // next_nodes: two beams, each with children [6, 7]
    assert_eq!(out.next_nodes, vec![6, 7, 6, 7]);
}

// ── Padding behaviour ─────────────────────────────────────────────────────

#[test]
fn padding_slots_are_zero_and_invalid() {
    let tm = paper_matrix();
    // Node 1 has degree 1 but B_t at level 1 may be 1 or 2 depending on
    // the trie; check that all slots beyond n_child are zeroed.
    let res = tm.vntk(&[1], 1);
    let n_valid = res.valid.iter().filter(|&&v| v).count();
    // The rest must be zero-padded
    for j in n_valid..res.branch_size {
        assert_eq!(res.tokens[j], 0, "pad token must be 0");
        assert_eq!(res.next_nodes[j], 0, "pad next_node must be 0");
        assert!(!res.valid[j], "pad slot must be invalid");
    }
}

// ── branch_size cap ──────────────────────────────────────────────────────

#[test]
fn branch_size_never_exceeds_max_branches() {
    let tm = paper_matrix();
    for level in 0..tm.sid_length as usize {
        let res = tm.vntk(&[0], level);
        assert_eq!(
            res.branch_size, tm.max_branches[level] as usize,
            "branch_size must equal max_branches[{level}]"
        );
    }
}
