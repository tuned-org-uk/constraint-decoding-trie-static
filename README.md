# [Generative Retrieval] Transition Matrix Trie for Constraint Decoding

Implementation of STATIC from this [paper](https://arxiv.org/abs/2602.22647v1): **Vectorizing the Trie: Efficient Constrained Decoding for LLM-based Generative Retrieval on Accelerators**

This is implemented using parallelisation on CPU, developing the GPU shader will be nice but quite time-consuming.

## Motivation

A Rust library for **fast, production-friendly constrained decoding** over a large, fixed candidate set (e.g., Semantic IDs / item IDs for generative retrieval). It flattens a prefix trie into a CSR transition matrix and uses a vectorized kernel (VNTK) plus a bit-packed dense mask for shallow layers to keep constraint enforcement accelerator-friendly.

This design is inspired by STATIC (“Sparse Transition Matrix‑Accelerated Trie Index for Constrained Decoding”), which recasts trie traversal as vectorized sparse-matrix operations and reports very large speedups and low per-step overhead in an industrial deployment.


## Why this matters commercially

Constrained decoding is what turns “LLM can generate an ID” into “LLM can generate only valid, in-policy IDs”:

- **Hard business logic**: enforce inventory/freshness/eligibility policies at generation-time (no post-filtering surprises).
- **Lower latency overhead**: move from pointer-chasing trie walks to contiguous CSR reads and vectorized scattering.
- **Scale**: supports large constraint sets, where binary-search style methods can become an I/O bottleneck; STATIC frames this as `O(1)` I/O w.r.t. constraint-set size for the kernel.
- **Deterministic outputs**: the index is static and reproducible, which helps auditing and rollout safety.

## What’s included

Project layout:

```bash
src
├── trie.rs          # Prefix trie construction
├── transition.rs    # CSR transition matrix builder
├── dense_mask.rs    # Bit-packed dense mask for shallow layers
├── vntk.rs          # Vectorized Node Transition Kernel
├── decoder.rs       # Constrained decoding step (Algorithm 1)
└── types.rs         # TransitionMatrix, DecodingState, etc.
```

## Quick start

Key structures:

- `DenseMask`: bit-packed validity for the first `d` layers (typically `d=2`).
- `TransitionMatrix`: CSR matrix with interleaved `(token_id, next_node_id)` entries.
- `StaticIndex`: `dense + sparse` combined index.
- `ConstrainedDecoder`: constrained beam step (mask → select → gather).

### Build an index

```rust
use constraint_decoding_trie::transition::build_static_index;

let constraints = vec![
  vec![1u32, 2, 1],
  vec![3u32, 1, 2],
  vec![3u32, 1, 3],
];

let index = build_static_index(
  &constraints,
  /*vocab_size=*/ 4,
  /*sid_length=*/ 3,
  /*dense_depth=*/ 2,
);
```

## Benchmarks (Criterion)

Benchmarks live under `benches/` and use Criterion. Run:

```bash
cargo bench
```

## Notes on correctness

- Children in each CSR row are stored sorted by token ID (deterministic index layout).
- Dense masking uses safe bit-range scans; boundary conditions are covered by tests.
- Sparse masking uses VNTK to produce per-beam masks and next-node slots.

## Roadmap ideas

- Parquet persistence for `StaticIndex`
- SIMD/bitset masks for faster CPU-side gating
- GPU/TPU-friendly kernels (layout already designed for coalesced reads)

## References

STATIC paper: “Vectorizing the Trie: Efficient Constrained Decoding for LLM-based Generative Retrieval on Accelerators” (2026), [paper](https://arxiv.org/abs/2602.22647v1).
