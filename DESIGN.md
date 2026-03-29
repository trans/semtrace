# Semantic Tracing: Greedy Residual Decomposition of Embedding Vectors

**Thomas Sawyer**^1 **& Claude Opus 4.6**^2 · *March 2026* · *Speculative / Pre-empirical*
*^1 Independent Researcher · ^2 Anthropic*

---

## Abstract

We propose a method for decomposing an embedding vector into an ordered sequence of tokens whose embeddings sum to approximate it — a process we call *semantic tracing*. The key insight is that greedy residual subtraction in embedding space is both computationally tractable and potentially semantically meaningful. If the decomposition is faithful, it constitutes a partial inversion of the encoder: a way to *read* meaning from a point in embedding space as natural language. We outline the hypothesis, the algorithm, open empirical questions, and implications for interpretability and semantic memory systems.

---

## 1. Motivation

Embedding spaces are typically treated as opaque. Distances and directions carry meaning, but individual coordinates do not, and there is no standard method to recover natural language from an arbitrary point in the space. We ask: given a vector **v** in embedding space, can we find a short sequence of tokens whose embeddings sum to **v** — and does that sequence constitute a meaningful *reading* of **v**?

If yes, embedding space becomes a readable map. Internal model activations become interpretable. Concept arithmetic becomes auditable.

---

## 2. The Forward Direction: Semantic Tracing

Define the *trace* of a token sequence T = (t₁, t₂, ..., tₙ) as:
```
trace(T) = Σ E(tᵢ)  for i = 1..N
```

where E(t) is the static embedding of token t. This is simply vector addition. The trace is a point in embedding space reached by summing the token embeddings in order.

**Observation 1**: In high-dimensional embedding spaces (d ≥ 512), the probability that two distinct *unordered* token multisets share the same trace approaches zero for practical vocabulary sizes, due to the geometry of high-dimensional spaces.

**Observation 2**: Order is not preserved by summation (commutativity). Any permutation of the same token multiset produces the same trace. We conjecture this is a feature, not a bug: *if two orderings arrive at the same point, they encode the same concept.*

This aligns with how positional encoding works in transformers — order is injected separately (via RoPE or sinusoidal encoding), while the embedding itself encodes semantic content. Our method operates purely on the semantic layer.

---

## 3. The Inverse Direction: Greedy Residual Decomposition

Given a target vector **v**, we seek a short token sequence T such that trace(T) ≈ **v**.

**Algorithm**:
```
residual ← v
tokens ← []

while |residual| > ε:
    t* ← argmin_{t ∈ vocab} |E(t) - residual|
    tokens.append(t*)
    residual ← residual - E(t*)

return tokens
```

At each step, find the token whose embedding is nearest to the current residual, subtract it, and repeat. Terminate when the residual norm falls below threshold ε.

**Complexity**: O(N × |vocab|) where N is the number of tokens in the decomposition. For typical vocabulary sizes (~50k) and small N, this is fast. Approximate nearest neighbour structures (HNSW, FAISS) reduce this further to O(N × log|vocab|).

**N is self-revealing**: the algorithm terminates naturally when residual energy is exhausted. No prior knowledge of sequence length is required. This makes the decomposition *variable-length*, analogous to prime factorisation.

---

## 4. Open Empirical Questions

The central empirical question is whether this decomposition is *semantically faithful* — i.e. whether the recovered tokens constitute a meaningful reading of the original embedding.

**Q1: Static vs contextual embeddings**
Transformer models produce contextual embeddings, where each token's representation is shaped by attention over its neighbours. Static token embeddings (the input embedding matrix) live in the same space but may not span it identically. It is an open question whether contextual embeddings are reachable by sums of static embeddings, or whether attention distorts the space in ways that break the correspondence.

**Q2: Approximate sufficiency**
Exact reconstruction may be unnecessary. If *semantic equivalence* is preserved under small residual error, approximate decomposition is sufficient. The threshold ε is a tunable parameter with semantic implications.

**Q3: Decomposition uniqueness**
Are decompositions unique, or does the greedy path matter? A different greedy ordering might yield different tokens with the same semantic content. This is worth measuring empirically.

**Q4: Concept compressibility**
We conjecture that most natural concepts decompose into small N. If meaning is compressible, N will be small and stable across similar concepts. This would make the decomposition practically useful as a semantic address.

---

## 5. Implications

**Interpretability**: Apply to internal activations of a running model. Read what the model is "thinking" at any layer as natural language token decompositions.

**Semantic memory**: Use as the retrieval layer in a memory system. Stored memories are points in embedding space; retrieval is decomposition. The token sequence is a human-readable summary of what was remembered.

**Concept navigation**: Walk through embedding space along a gradient and watch the decomposition change token by token. Meaning becomes navigable.

**Prompt inversion**: Given a desired output embedding, decompose it to find the natural language prompt that produces it. A form of automatic prompt engineering grounded in geometry.

---

## 6. Relation to Existing Work

This is adjacent to but distinct from:

- **Sparse autoencoders** (mechanistic interpretability) — which decompose activations into learned feature directions, not vocabulary tokens
- **Concept arithmetic** (Mikolov et al.) — which operates on embedding differences post-hoc, not decomposition
- **Greedy decoding** — which selects tokens autoregressively conditioned on context, not by residual subtraction

Semantic tracing is unusual in that it requires no trained decomposition model. It uses only the static embedding matrix, which is available in any transformer.

---

## 7. Next Steps

1. Implement greedy residual decomposition against a standard embedding matrix (e.g. GPT-2, BERT)
2. Test on known embeddings: does E("cat") decompose to ["cat"]? Does E("cat") + E("sat") decompose to ["cat", "sat"] in some order?
3. Test on contextual embeddings: take a mean-pooled sentence embedding and attempt decomposition
4. Measure residual norm as a function of N — does it decay meaningfully?
5. Qualitative evaluation: are recovered tokens semantically related to the source?

---

## 8. Conclusion

Semantic tracing proposes that embedding vectors are not opaque coordinates but *readable sums* — decomposable into the token primitives that constitute their meaning. The greedy residual algorithm is simple, fast, and requires no additional training. Whether it is faithful is an empirical question worth answering. If it is, even approximately, the implications for interpretability, memory, and concept navigation are significant.

*We shall see.*
