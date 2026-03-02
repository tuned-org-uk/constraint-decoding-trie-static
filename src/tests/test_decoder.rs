use crate::decoder::{
    ConstrainedDecoder, apply_mask, beam_search, constrained_beam_decode, log_softmax_1d,
    log_softmax_3d,
};
use crate::transition::build_static_index;
use crate::types::BeamState;
use std::collections::HashSet;

fn paper_decoder(beam_width: usize) -> ConstrainedDecoder {
    let index = build_static_index(&[vec![1u32, 2, 1], vec![3, 1, 2], vec![3, 1, 3]], 4, 3, 2);
    ConstrainedDecoder::new(index, beam_width, 1)
}

// ── log_softmax ───────────────────────────────────────────────────────────

#[test]
fn log_softmax_sums_to_one_in_prob_space() {
    let lp = log_softmax_1d(&[1.0, 2.0, 3.0, 4.0]);
    let sum: f64 = lp.iter().map(|v| v.exp()).sum();
    assert!((sum - 1.0).abs() < 1e-10, "exp(log_softmax) must sum to 1");
}

#[test]
fn log_softmax_max_entry_is_zero_for_one_hot() {
    // All mass on a single token → that token gets log-prob ≈ 0.
    let mut x = vec![f64::NEG_INFINITY; 4];
    x[2] = 0.0;
    let lp = log_softmax_1d(&x);
    assert!((lp[2] - 0.0).abs() < 1e-10);
}

#[test]
fn log_softmax_numerically_stable_large_inputs() {
    // Without subtracting max, exp(1000) overflows.
    let x: Vec<f64> = (0..4).map(|i| 1000.0 + i as f64).collect();
    let lp = log_softmax_1d(&x);
    assert!(lp.iter().all(|v| v.is_finite()));
}

// ── apply_mask ────────────────────────────────────────────────────────────

#[test]
fn apply_mask_sets_invalid_to_neg_inf() {
    let lp = vec![vec![vec![-1.0f64, -2.0, -3.0, -4.0]]];
    let masks = vec![vec![vec![true, false, true, false]]];
    let out = apply_mask(&lp, &masks);
    assert!(out[0][0][1].is_infinite() && out[0][0][1] < 0.0);
    assert!(out[0][0][3].is_infinite() && out[0][0][3] < 0.0);
    assert_eq!(out[0][0][0], -1.0);
    assert_eq!(out[0][0][2], -3.0);
}

#[test]
fn apply_mask_all_valid_passes_through() {
    let lp = vec![vec![vec![-0.5f64, -1.5]]];
    let masks = vec![vec![vec![true, true]]];
    let out = apply_mask(&lp, &masks);
    assert_eq!(out[0][0], vec![-0.5, -1.5]);
}

// ── beam_search ───────────────────────────────────────────────────────────

#[test]
fn beam_search_selects_top_k_globally() {
    // Single query, two beams, 4 vocab.
    // Beam 0 scores: [-1, -2, -3, -4]  parent=0.0
    // Beam 1 scores: [-0.5, -10, -10, -10] parent=0.0
    // Best globally: beam1+tok0=-0.5, beam0+tok0=-1.0
    let lp = vec![vec![
        vec![-1.0f64, -2.0, -3.0, -4.0],
        vec![-0.5, -10.0, -10.0, -10.0],
    ]];
    let par = vec![vec![0.0f64, 0.0]];
    let (toks, scores, srcs) = beam_search(&lp, &par, 2);
    assert_eq!(toks[0][0], 0u32, "best token is 0 from beam 1");
    assert_eq!(srcs[0][0], 1, "sourced from beam 1");
    assert!((scores[0][0] - (-0.5)).abs() < 1e-10);
    assert_eq!(toks[0][1], 0u32, "second best: token 0 from beam 0");
    assert_eq!(srcs[0][1], 0);
}

#[test]
fn beam_search_skips_neg_inf_tokens() {
    let lp = vec![vec![vec![f64::NEG_INFINITY, -1.0, f64::NEG_INFINITY, -2.0]]];
    let par = vec![vec![0.0f64]];
    let (toks, _, _) = beam_search(&lp, &par, 2);
    // Only tokens 1 and 3 are finite
    let tok_set: HashSet<u32> = toks[0].iter().copied().collect();
    assert!(tok_set.contains(&1));
    assert!(tok_set.contains(&3));
    assert!(!tok_set.contains(&0));
    assert!(!tok_set.contains(&2));
}

// ── end-to-end decoding ───────────────────────────────────────────────────

#[test]
fn decode_recovers_all_constraints_uniform_logits() {
    let constraints = vec![vec![1u32, 2, 1], vec![3, 1, 2], vec![3, 1, 3]];
    let index = build_static_index(&constraints, 4, 3, 2);
    let result = constrained_beam_decode(&index, &[0.0f32; 4], 3, 3);
    let decoded: HashSet<Vec<u32>> = result.into_iter().collect();
    let expected: HashSet<Vec<u32>> = constraints.into_iter().collect();
    assert_eq!(decoded, expected);
}

#[test]
fn decode_single_constraint_always_returned() {
    let constraints = vec![vec![2u32, 0, 3]];
    let index = build_static_index(&constraints, 4, 3, 2);
    let result = constrained_beam_decode(&index, &[0.0f32; 4], 3, 1);
    assert_eq!(result, vec![vec![2u32, 0, 3]]);
}

#[test]
fn decode_respects_hard_constraint_against_biased_logits() {
    // Model strongly prefers token 0 everywhere; only [3,1,2] is valid.
    let constraints = vec![vec![3u32, 1, 2]];
    let index = build_static_index(&constraints, 4, 3, 2);
    let logits = {
        let mut v = vec![0.0f32; 4];
        v[0] = 100.0; // strongly bias toward token 0
        v
    };
    let result = constrained_beam_decode(&index, &logits, 3, 1);
    assert_eq!(
        result,
        vec![vec![3u32, 1, 2]],
        "hard mask must override model bias"
    );
}

#[test]
fn decode_produces_beam_width_results() {
    let dec = paper_decoder(3);
    let logits_3d = vec![vec![vec![0.0f64; 4]; 3]];
    let seqs = dec.decode(|_, _| logits_3d.clone(), 3);
    assert_eq!(seqs[0].len(), 3);
}

// ── dense / sparse path boundaries ───────────────────────────────────────

#[test]
fn dense_first_token_valid_correct() {
    let dec = paper_decoder(2);
    assert!(!dec.index.dense.first_token_valid(0));
    assert!(dec.index.dense.first_token_valid(1));
    assert!(!dec.index.dense.first_token_valid(2));
    assert!(dec.index.dense.first_token_valid(3));
}

#[test]
fn step0_mask_exposes_only_valid_first_tokens() {
    let dec = paper_decoder(2);
    let state = BeamState::new(1, 2);
    let (masks, _) = dec.dense_lookup(&state, 0);
    // Both beams start at root; same mask expected.
    for mi in 0..2 {
        assert!(!masks[0][mi][0], "token 0 invalid at step 0");
        assert!(masks[0][mi][1], "token 1 valid at step 0");
        assert!(!masks[0][mi][2], "token 2 invalid at step 0");
        assert!(masks[0][mi][3], "token 3 valid at step 0");
    }
}

#[test]
fn sparse_lookup_matches_vntk_directly() {
    let index = build_static_index(&[vec![1u32, 2, 1], vec![3, 1, 2], vec![3, 1, 3]], 4, 3, 2);
    let mut state = BeamState::new(1, 1);
    // Manually position beam 0 at node 4 (reached by prefix [3,1]).
    state.nodes[0][0] = 4;

    let dec = ConstrainedDecoder::new(index.clone(), 1, 1);
    let (masks, _) = dec.sparse_lookup(&state, 2);
    assert!(!masks[0][0][0]);
    assert!(!masks[0][0][1]);
    assert!(masks[0][0][2], "token 2 valid at node 4");
    assert!(masks[0][0][3], "token 3 valid at node 4");
}

// ── resolve_next_node ─────────────────────────────────────────────────────

#[test]
fn resolve_next_node_binary_search() {
    let dec = paper_decoder(2);
    // Node 4 has children: token 2 → node 6, token 3 → node 7
    // VNTK slots: [6, 7]
    let slots = vec![6u32, 7];
    assert_eq!(dec.resolve_next_node(4, 2, &slots, 2), 6);
    assert_eq!(dec.resolve_next_node(4, 3, &slots, 2), 7);
}
