// src/transition.rs

use std::collections::HashMap;
use std::collections::VecDeque;

use crate::types::{DenseMask, StaticIndex, TransitionMatrix};

// ──────────────────────────────────────────────────────────────────────────────
// Internal trie node
// ──────────────────────────────────────────────────────────────────────────────

struct TrieNode {
    /// Child nodes keyed by the token that leads to them.
    children: HashMap<u32, Box<TrieNode>>,
    /// BFS-assigned integer ID (set during `enumerate_nodes`).
    node_id: u32,
    /// Depth of this node in the trie (root = 0).
    level: u32,
    /// True iff at least one constraint sequence ends here.
    is_terminal: bool,
}

impl TrieNode {
    fn new(level: u32) -> Self {
        Self {
            children: HashMap::new(),
            node_id: 0,
            level,
            is_terminal: false,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Phase 1 — trie construction
// ──────────────────────────────────────────────────────────────────────────────

/// Inserts all constraint sequences into a fresh trie and returns the root.
fn build_trie(constraints: &[Vec<u32>], vocab_size: u32, _sid_length: u32) -> Box<TrieNode> {
    let mut root = Box::new(TrieNode::new(0));

    for seq in constraints {
        let mut cur: *mut TrieNode = root.as_mut();

        for &token in seq {
            debug_assert!(
                token < vocab_size,
                "token {token} out of vocabulary (|V|={vocab_size})"
            );
            // SAFETY: `cur` always points to a live node owned by `root`.
            let node = unsafe { &mut *cur };
            let level = node.level + 1;
            let child = node
                .children
                .entry(token)
                .or_insert_with(|| Box::new(TrieNode::new(level)));
            cur = child.as_mut();
        }

        // Mark the terminal node.
        unsafe { (*cur).is_terminal = true };
    }

    root
}

// ──────────────────────────────────────────────────────────────────────────────
// Phase 2 — BFS enumeration
// ──────────────────────────────────────────────────────────────────────────────

/// BFS over the trie, assigning monotonically increasing integer IDs in
/// level-order.  Returns:
///
/// - `node_map`    : raw pointer → integer ID (used by CSR builder)
/// - `level_nodes` : `level_nodes[l]` = number of nodes at depth l
///
/// BFS order guarantees that for any node with ID i, all nodes on shallower
/// levels have IDs < i, which matches the paper's Figure 1d layout.
fn enumerate_nodes(root: &TrieNode) -> (HashMap<*const TrieNode, u32>, Vec<u32>) {
    let mut node_map: HashMap<*const TrieNode, u32> = HashMap::new();
    let mut level_counts: Vec<u32> = Vec::new();
    let mut queue: VecDeque<*const TrieNode> = VecDeque::new();
    let mut next_id: u32 = 0;

    queue.push_back(root as *const _);

    while let Some(ptr) = queue.pop_front() {
        // SAFETY: all pointers in the queue are live nodes owned by `root`.
        let node = unsafe { &*ptr };

        // Extend level_counts if this is the first node we see at this depth.
        while level_counts.len() <= node.level as usize {
            level_counts.push(0);
        }
        level_counts[node.level as usize] += 1;

        node_map.insert(ptr, next_id);
        next_id += 1;

        // Enqueue children sorted by token so the ID assignment is deterministic
        // across runs (important for reproducible Parquet snapshots).
        let mut children: Vec<(&u32, &Box<TrieNode>)> = node.children.iter().collect();
        children.sort_by_key(|(tok, _)| *tok);
        for (_, child) in children {
            queue.push_back(child.as_ref() as *const _);
        }
    }

    (node_map, level_counts)
}

// ──────────────────────────────────────────────────────────────────────────────
// Phase 3 — max branch factor per level
// ──────────────────────────────────────────────────────────────────────────────

/// Returns a Vec of length `sid_length` where entry `l` is the maximum number
/// of children any node at depth `l` has.  Used by VNTK to size its output
/// buffer without dynamic allocation.
fn compute_max_branches(root: &TrieNode, sid_length: u32) -> Vec<u32> {
    let mut max_branches = vec![0u32; sid_length as usize];
    let mut stack: Vec<*const TrieNode> = vec![root as *const _];

    while let Some(ptr) = stack.pop() {
        let node = unsafe { &*ptr };
        if (node.level as usize) < max_branches.len() {
            let deg = node.children.len() as u32;
            if deg > max_branches[node.level as usize] {
                max_branches[node.level as usize] = deg;
            }
        }
        for child in node.children.values() {
            stack.push(child.as_ref() as *const _);
        }
    }

    max_branches
}

// ──────────────────────────────────────────────────────────────────────────────
// Phase 4 — CSR construction
// ──────────────────────────────────────────────────────────────────────────────

fn build_csr(
    root: &TrieNode,
    node_map: &HashMap<*const TrieNode, u32>,
    vocab_size: u32,
    sid_length: u32,
    max_branches: &[u32],
) -> TransitionMatrix {
    let num_nodes = node_map.len() as u32;

    // Collect all nodes into a Vec indexed by their BFS ID so we can fill
    // row_pointers in strictly ascending order.
    let mut nodes_by_id: Vec<*const TrieNode> = vec![std::ptr::null(); num_nodes as usize];
    {
        let mut stack: Vec<*const TrieNode> = vec![root as *const _];
        while let Some(ptr) = stack.pop() {
            let id = node_map[&ptr] as usize;
            nodes_by_id[id] = ptr;
            let node = unsafe { &*ptr };
            for child in node.children.values() {
                stack.push(child.as_ref() as *const _);
            }
        }
    }

    let mut row_pointers = Vec::with_capacity(num_nodes as usize + 1);
    let mut data: Vec<[u32; 2]> = Vec::new();

    let mut offset = 0u32;
    for ptr in &nodes_by_id {
        row_pointers.push(offset);

        let node = unsafe { &**ptr };

        // Sort children by token ID for deterministic, cache-friendly access.
        let mut children: Vec<(u32, u32)> = node
            .children
            .iter()
            .map(|(&tok, child)| {
                let next_id = node_map[&(child.as_ref() as *const _)];
                (tok, next_id)
            })
            .collect();
        children.sort_by_key(|&(tok, _)| tok);

        for (tok, next_id) in children {
            data.push([tok, next_id]);
            offset += 1;
        }
    }
    row_pointers.push(offset); // sentinel

    TransitionMatrix {
        row_pointers,
        data,
        max_branches: max_branches.to_vec(),
        num_nodes,
        vocab_size,
        sid_length,
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Phase 5 — dense mask
// ──────────────────────────────────────────────────────────────────────────────

/// Populates the bit-packed dense mask for all prefixes of length `dense_depth`.
///
/// Walks each constraint once, extracting the first `dense_depth` tokens and
/// looking up the destination trie node via `node_map`.
fn build_dense_mask(
    constraints: &[Vec<u32>],
    root: &TrieNode,
    vocab_size: u32,
    dense_depth: u32,
    node_map: &HashMap<*const TrieNode, u32>,
) -> DenseMask {
    let mut mask = DenseMask::new(vocab_size, dense_depth);

    for seq in constraints {
        // Walk the trie along the first `dense_depth` tokens.
        let mut cur: *const TrieNode = root as *const _;

        for (step, &token) in seq.iter().enumerate().take(dense_depth as usize) {
            let node = unsafe { &*cur };
            match node.children.get(&token) {
                Some(child) => cur = child.as_ref() as *const _,
                None => break, // should never happen for valid constraints
            }

            // After consuming exactly `dense_depth` tokens, record the node.
            if step + 1 == dense_depth as usize {
                let node_id = node_map[&cur];
                let prefix = &seq[..dense_depth as usize];
                mask.insert(prefix, node_id);
            }
        }
    }

    mask
}

// ──────────────────────────────────────────────────────────────────────────────
// Public entry point
// ──────────────────────────────────────────────────────────────────────────────

/// Build a STATIC index from a set of Semantic ID sequences.
///
/// # Arguments
/// - `constraints` — `|C|` sequences of token IDs, each of length `sid_length`
/// - `vocab_size`  — vocabulary size |V|; all tokens must be in `[0, vocab_size)`
/// - `sid_length`  — fixed length L of every SID
/// - `dense_depth` — number of layers to cover with the dense mask (typically 2)
///
/// # Panics
/// Panics in debug mode if any token ≥ `vocab_size` or any sequence length ≠
/// `sid_length`.
pub fn build_static_index(
    constraints: &[Vec<u32>],
    vocab_size: u32,
    sid_length: u32,
    dense_depth: u32,
) -> StaticIndex {
    debug_assert!(
        dense_depth <= sid_length,
        "dense_depth ({dense_depth}) must be ≤ sid_length ({sid_length})"
    );
    debug_assert!(
        constraints.iter().all(|s| s.len() == sid_length as usize),
        "every constraint must have exactly sid_length={sid_length} tokens"
    );

    // ── Phase 1 ──────────────────────────────────────────────────────────────
    let trie = build_trie(constraints, vocab_size, sid_length);

    // ── Phase 2 ──────────────────────────────────────────────────────────────
    let (node_map, _level_counts) = enumerate_nodes(&trie);

    // ── Phase 3 ──────────────────────────────────────────────────────────────
    let max_branches = compute_max_branches(&trie, sid_length);

    // ── Phase 4 ──────────────────────────────────────────────────────────────
    let sparse = build_csr(&trie, &node_map, vocab_size, sid_length, &max_branches);

    // ── Phase 5 ──────────────────────────────────────────────────────────────
    let dense = build_dense_mask(constraints, &trie, vocab_size, dense_depth, &node_map);

    #[cfg(debug_assertions)]
    sparse
        .check_invariants()
        .expect("CSR invariants violated after construction");

    StaticIndex {
        dense,
        sparse,
        num_constraints: constraints.len(),
    }
}
