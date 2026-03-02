use crate::transition::build_static_index;
use crate::types::StaticIndex;

fn paper_index() -> StaticIndex {
    let constraints = vec![vec![1u32, 2, 1], vec![3, 1, 2], vec![3, 1, 3]];
    build_static_index(&constraints, 4, 3, 2)
}

// ── Trie shape ───────────────────────────────────────────────────────────

#[test]
fn node_count_matches_paper() {
    // Unique nodes: root + {1,3} + {2,1} + {1,1} + {1} + {2,3} = 8
    let idx = paper_index();
    assert_eq!(idx.sparse.num_nodes, 8);
}

#[test]
fn total_edge_count() {
    // root(2) + 1→(1) + 3→(1) + [1,2]→(1) + [3,1]→(2) = 7 edges
    let idx = paper_index();
    assert_eq!(idx.sparse.data.len(), 7);
}

// ── CSR invariants ───────────────────────────────────────────────────────

#[test]
fn csr_invariants_hold() {
    assert!(paper_index().sparse.check_invariants().is_ok());
}

#[test]
fn row_pointer_sentinel() {
    let idx = paper_index();
    let last = *idx.sparse.row_pointers.last().unwrap() as usize;
    assert_eq!(
        last,
        idx.sparse.data.len(),
        "sentinel must equal total edge count"
    );
}

// ── Root children ────────────────────────────────────────────────────────

#[test]
fn root_has_two_sorted_children() {
    let idx = paper_index();
    let ch = idx.sparse.children(0);
    assert_eq!(ch.len(), 2);
    assert_eq!(ch[0][0], 1, "first child token should be 1 (sorted)");
    assert_eq!(ch[1][0], 3, "second child token should be 3 (sorted)");
}

// ── Figure 1d spot-checks ────────────────────────────────────────────────

#[test]
fn node4_children_match_figure1d() {
    let idx = paper_index();
    let start = idx.sparse.row_pointers[4] as usize;
    let end = idx.sparse.row_pointers[5] as usize;
    assert_eq!(end - start, 2);
    assert_eq!(idx.sparse.data[start], [2, 6]);
    assert_eq!(idx.sparse.data[start + 1], [3, 7]);
}

#[test]
fn leaf_nodes_have_no_children() {
    let idx = paper_index();
    for leaf in [5usize, 6, 7] {
        let s = idx.sparse.row_pointers[leaf] as usize;
        let e = idx.sparse.row_pointers[leaf + 1] as usize;
        assert_eq!(e - s, 0, "node {leaf} must be a leaf");
    }
}

// ── next_node traversal ──────────────────────────────────────────────────

#[test]
fn traverse_sid_1_2_1() {
    let idx = paper_index();
    let sp = &idx.sparse;
    let n1 = sp.next_node(0, 1).expect("root → token 1");
    let n2 = sp.next_node(n1, 2).expect("node → token 2");
    let leaf = sp.next_node(n2, 1).expect("node → token 1");
    assert!(sp.is_leaf(leaf), "end of SID [1,2,1] must be a leaf");
}

#[test]
fn traverse_sid_3_1_2_and_3_1_3_share_prefix() {
    let idx = paper_index();
    let sp = &idx.sparse;
    let n1 = sp.next_node(0, 3).unwrap();
    let n2 = sp.next_node(n1, 1).unwrap();
    // Both tokens 2 and 3 must be valid from n2
    let l2 = sp.next_node(n2, 2).expect("→ SID [3,1,2]");
    let l3 = sp.next_node(n2, 3).expect("→ SID [3,1,3]");
    assert!(sp.is_leaf(l2));
    assert!(sp.is_leaf(l3));
    assert_ne!(l2, l3, "distinct SIDs must end at distinct leaves");
}

// ── Dense mask ───────────────────────────────────────────────────────────

#[test]
fn dense_mask_valid_prefixes() {
    let idx = paper_index();
    assert!(idx.dense.get(1, 2), "prefix [1,2] must be valid");
    assert!(idx.dense.get(3, 1), "prefix [3,1] must be valid");
    assert!(!idx.dense.get(0, 0));
    assert!(!idx.dense.get(1, 1));
    assert!(!idx.dense.get(2, 2));
    assert!(!idx.dense.get(3, 3));
}

#[test]
fn dense_mask_state_for_prefix() {
    let idx = paper_index();
    // After [3,1] we must be at node 4 (which has children token 2→6 and 3→7)
    let node = idx.dense.state_for(&[3, 1]).expect("prefix [3,1] exists");
    assert_eq!(
        idx.sparse.degree(node),
        2,
        "node reached by [3,1] must have degree 2"
    );
}

#[test]
fn dense_mask_prefix_1_2_leads_to_degree_1_node() {
    let idx = paper_index();
    let node = idx.dense.state_for(&[1, 2]).expect("prefix [1,2] exists");
    assert_eq!(
        idx.sparse.degree(node),
        1,
        "node reached by [1,2] must have exactly one child (token 1)"
    );
}

// ── Max branches ─────────────────────────────────────────────────────────

#[test]
fn max_branches_level0_is_2() {
    // Root has 2 children (tokens 1 and 3)
    let idx = paper_index();
    assert_eq!(idx.sparse.max_branches[0], 2);
}

#[test]
fn max_branches_level2_is_2() {
    // Node 4 (at level 2) has the highest branching: tokens 2 and 3
    let idx = paper_index();
    assert_eq!(idx.sparse.max_branches[2], 2);
}

// ── Deduplication ────────────────────────────────────────────────────────

#[test]
fn duplicate_constraints_do_not_add_nodes() {
    let constraints = vec![
        vec![1u32, 2, 1],
        vec![1u32, 2, 1], // exact duplicate
    ];
    let idx = build_static_index(&constraints, 4, 3, 2);
    // A non-deduplicating trie would produce more nodes; expect 4 (root + 3)
    assert_eq!(
        idx.sparse.num_nodes, 4,
        "duplicate constraints must not inflate the trie"
    );
}

// ── Edge cases ───────────────────────────────────────────────────────────

#[test]
fn single_constraint_is_linear_chain() {
    let idx = build_static_index(&[vec![0u32, 1, 2]], 4, 3, 2);
    // root + 3 nodes = 4; each internal node has degree 1
    assert_eq!(idx.sparse.num_nodes, 4);
    for node in 0..3 {
        assert_eq!(idx.sparse.degree(node), 1);
    }
    assert!(idx.sparse.is_leaf(3));
}

#[test]
fn all_tokens_same_gives_single_path() {
    let idx = build_static_index(&[vec![2u32, 2, 2]], 4, 3, 2);
    assert_eq!(idx.sparse.num_nodes, 4);
    assert_eq!(idx.sparse.next_node(0, 2), Some(1));
    assert_eq!(idx.sparse.next_node(1, 2), Some(2));
    assert_eq!(idx.sparse.next_node(2, 2), Some(3));
    assert!(idx.sparse.is_leaf(3));
}

#[test]
fn large_fanout_root() {
    // All 4 possible first tokens present → root has degree 4
    let constraints: Vec<Vec<u32>> = (0..4).map(|i| vec![i, 0, 0]).collect();
    let idx = build_static_index(&constraints, 4, 3, 2);
    assert_eq!(idx.sparse.degree(0), 4);
    assert_eq!(idx.sparse.max_branches[0], 4);
}
