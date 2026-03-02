# [Generative Retrieval] Transition Matrix Trie for Constraint Decoding

Implementation of STATIC from this [paper](https://arxiv.org/abs/2602.22647v1): Vectorizing the Trie: Efficient Constrained Decoding for LLM-based Generative Retrieval on Accelerators

```bash
src
├── trie.rs          # Prefix trie construction
├── transition.rs    # CSR transition matrix builder
├── dense_mask.rs    # Bit-packed dense mask for shallow layers
├── vntk.rs          # Vectorized Node Transition Kernel
├── decoder.rs       # Constrained decoding step (Algorithm 1)
└── types.rs         # TransitionMatrix, DecodingState, etc.
```
