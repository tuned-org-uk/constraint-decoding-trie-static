use crate::types::{BeamState, DenseMask, TransitionMatrix};

// ── TransitionMatrix ────────────────────────────────────────────────────

#[test]
fn transition_matrix_children_and_next_node() {
    let mut tm = TransitionMatrix::new(3, 4, 2);
    // Manually wire: node 0 → token 1 → node 1, token 3 → node 2
    tm.row_pointers = vec![0, 2, 2, 2];
    tm.data = vec![[1, 1], [3, 2]];

    let ch = tm.children(0);
    assert_eq!(ch.len(), 2);
    assert_eq!(tm.next_node(0, 1), Some(1));
    assert_eq!(tm.next_node(0, 3), Some(2));
    assert_eq!(tm.next_node(0, 0), None);

    assert!(tm.is_leaf(1));
    assert!(tm.is_leaf(2));
    assert!(!tm.is_leaf(0));
    assert_eq!(tm.degree(0), 2);
}

#[test]
fn transition_matrix_check_invariants_ok() {
    let mut tm = TransitionMatrix::new(2, 4, 1);
    tm.row_pointers = vec![0, 1, 1];
    tm.data = vec![[2, 1]];
    assert!(tm.check_invariants().is_ok());
}

#[test]
fn transition_matrix_check_invariants_bad_tail() {
    let mut tm = TransitionMatrix::new(1, 4, 1);
    tm.row_pointers = vec![0, 99]; // tail doesn't match data.len()
    tm.data = vec![[0, 0]]; // only 1 entry
    assert!(tm.check_invariants().is_err());
}

// ── DenseMask ───────────────────────────────────────────────────────────

#[test]
fn dense_mask_insert_and_contains() {
    let mut dm = DenseMask::new(4, 2);

    dm.insert(&[1, 2], 3);
    dm.insert(&[3, 1], 4);

    assert!(dm.contains(&[1, 2]));
    assert!(dm.contains(&[3, 1]));
    assert!(!dm.contains(&[0, 0]));
    assert!(!dm.contains(&[2, 2]));

    assert_eq!(dm.state_for(&[1, 2]), Some(3));
    assert_eq!(dm.state_for(&[3, 1]), Some(4));
    assert_eq!(dm.state_for(&[0, 0]), None);
}

#[test]
fn dense_mask_get_shorthand() {
    let mut dm = DenseMask::new(4, 2);
    dm.insert(&[1, 2], 5);

    assert!(dm.get(1, 2));
    assert!(!dm.get(1, 1));
}

#[test]
fn dense_mask_iter_valid() {
    let mut dm = DenseMask::new(4, 2);
    dm.insert(&[1, 2], 3);
    dm.insert(&[3, 1], 4);

    let valid: Vec<_> = dm.iter_valid().collect();
    assert_eq!(valid.len(), 2);
    assert!(valid.contains(&(vec![1, 2], 3)));
    assert!(valid.contains(&(vec![3, 1], 4)));
}

#[test]
fn dense_mask_bit_packing_boundary() {
    // Force a prefix whose flat index falls exactly on a u64 boundary.
    // vocab=8, depth=2 → 64 entries → single u64 word; index 63 = [7, 7]
    let mut dm = DenseMask::new(8, 2);
    dm.insert(&[7, 7], 42);
    assert!(dm.get(7, 7));
    assert!(!dm.get(7, 6));
    assert_eq!(dm.state_for(&[7, 7]), Some(42));
}

// ── BeamState ───────────────────────────────────────────────────────────

#[test]
fn beam_state_new_and_accessors() {
    let bs = BeamState::new(2, 3);
    assert_eq!(bs.batch_size(), 2);
    assert_eq!(bs.beam_width(), 3);
    assert_eq!(bs.step(), 0);
    assert!(bs.flat_nodes().iter().all(|&n| n == 0));
    assert!(bs.completed(3).is_empty());
}

#[test]
fn beam_state_flat_nodes_roundtrip() {
    let mut bs = BeamState::new(1, 2);
    bs.nodes[0] = vec![4, 7];
    let flat = bs.flat_nodes();
    assert_eq!(flat, vec![4, 7]);
}

#[test]
fn beam_state_completed() {
    let mut bs = BeamState::new(1, 2);
    bs.tokens[0][0] = vec![1, 2, 1]; // complete (L=3)
    bs.tokens[0][1] = vec![3, 1]; // incomplete
    let done = bs.completed(3);
    assert_eq!(done, vec![vec![1u32, 2, 1]]);
}
