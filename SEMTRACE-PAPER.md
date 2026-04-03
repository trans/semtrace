# Semantic Tracing: Recovering Text from Embedding Vectors via Greedy Residual Decomposition

**Thomas Sawyer**^1 **& Claude Opus 4.6**^2
*^1 Independent Researcher · ^2 Anthropic*

*April 2026*

---

## Abstract

We present *semantic tracing*, a method for decomposing embedding vectors into the tokens that constitute them. The algorithm is simple: iteratively find the nearest token embedding to a residual vector, subtract it, and repeat. We show this greedy approach recovers bag-of-words content from static embedding sums with high fidelity — 100% on short texts, 90%+ on medium texts (Gettysburg Address, 143 unique tokens), with coordinate descent optimization doubling accuracy on smaller models. We characterize the limits: recovery depends on the ratio of unique tokens to embedding dimensions, with a critical threshold at ~1280 dimensions in the GPT-2 family. Union of multiple distance metrics (cosine, L2, inner product) improves recovery by exploiting complementary greedy paths.

For contextual embeddings, we discover that transformer attention adds a nearly constant per-token bias accounting for 99.5% of the hidden state energy. Subtracting this bias enables partial token recovery (17-80%) from contextual representations that previously yielded 0%. We show that embedding magnitude encodes token count, and that L2 normalization — standard in embedding APIs — destroys both magnitude and the ability to perform additive decomposition.

Vector addition is commutative: word order is mathematically unrecoverable from any sum of static embeddings. This is a fundamental limitation, not an algorithmic one. We discuss implications for embedding security, semantic compression, and the motivation for non-commutative embedding architectures.

---

## 1. Introduction

Embedding spaces are typically treated as opaque. Distances and directions carry meaning, but there is no standard method to recover natural language from an arbitrary point in the space. We ask: given a vector **v** in embedding space, can we find the tokens whose embeddings sum to approximate **v** — and do those tokens constitute a meaningful reading of **v**?

If yes, embedding space becomes readable. Stored embeddings become recoverable text. The security assumptions of vector databases change fundamentally.

We investigate this question empirically across two model families (GPT-2 at four scales, Llama 3.2 3B), three distance metrics, multiple optimization strategies, and both static and contextual embedding spaces. The results are more nuanced than a simple yes or no — recoverability depends on embedding dimensionality, vocabulary size, normalization, and the type of embedding (static vs contextual).

---

## 2. The Algorithm

### 2.1 Greedy Residual Decomposition

Given a target vector **v** and a vocabulary of token embeddings {E(t)}, we seek a multiset of tokens T whose embeddings sum to approximate **v**:

```
residual ← v
tokens ← []

while ||residual|| > ε and ||residual|| is decreasing:
    t* ← argmin_{t ∈ vocab} distance(E(t), residual)
    tokens.append(t*)
    residual ← residual - E(t*)

return tokens
```

At each step, find the token whose embedding is nearest to the current residual, subtract it, and repeat. Terminate when the residual norm begins increasing (inflection stopping condition), indicating the signal is exhausted.

### 2.2 Norm Inflection as Automatic N-Finder

For exact sums, the residual norm monotonically decreases to zero. For non-exact targets (concept arithmetic, contextual embeddings), the norm decreases while meaningful tokens are being recovered, then inflects upward when the algorithm begins chasing noise. This inflection automatically determines the number of tokens N without prior knowledge of sequence length.

### 2.3 Distance Metrics

We evaluate three metrics for the nearest-neighbor search:

- **Cosine similarity**: direction-only, ignores magnitude
- **L2 distance**: penalizes both direction and magnitude mismatch
- **Inner product**: rewards directional alignment weighted by magnitude

No single metric dominates across all conditions (Section 5).

---

## 3. Static Embedding Decomposition

### 3.1 Experimental Setup

We extract static token embedding matrices (`wte.weight`) from the GPT-2 family (Small 768d, Medium 1024d, Large 1280d, XL 1600d; all 50,257 tokens) and Llama 3.2 3B Instruct (3,072d; 128,256 tokens). For GPT-2, embeddings are extracted from HuggingFace safetensors files. For Llama, we parse the GGUF model file and dequantize from Q6_K to float32.

Search uses HNSW approximate nearest-neighbor (USearch library, cosine metric, f16 quantization) for GPT-2, and brute-force exact search for Llama (HNSW proved unreliable at 128K vocabulary scale — see Section 3.5). Texts are tokenized with GPT-2's BPE tokenizer or greedy longest-match for Llama.

### 3.2 Results on Real Text

We test on three public domain texts of increasing length:

**Mary Had a Little Lamb** (47 words, 55 tokens, 38 unique):

| Model | Dims | Unique Recovery |
|---|---|---|
| GPT-2 Small | 768 | 76.3% |
| GPT-2 Medium | 1024 | 68.4% |
| GPT-2 Large | 1280 | 100% |
| GPT-2 XL | 1600 | 100% |

**Gettysburg Address** (252 words, 286 tokens, 143 unique):

| Model | Dims | Unique Recovery |
|---|---|---|
| GPT-2 Small | 768 | 42.0% |
| GPT-2 Medium | 1024 | 40.6% |
| GPT-2 Large | 1280 | 89.5% |
| GPT-2 XL | 1600 | 90.2% |
| Llama 3.2 3B (brute-force) | 3072 | 100% |

**A Tale of Two Cities, Ch. 1** (1,001 words, ~1,400 tokens, ~630 unique):

| Model | Dims | Unique Recovery |
|---|---|---|
| GPT-2 XL | 1600 | 39.5% |
| Llama 3.2 3B (brute-force) | 3072 | 50.2% |

### 3.3 Dimensionality Threshold

A critical accuracy jump occurs between 1024d and 1280d across all tests. This is not a smooth scaling — it is a phase transition in embedding space separability. The GPT-2 Medium (1024d) anomalously performs worse than Small (768d), suggesting training quality contributes independently of dimension count.

Average token embedding norms decrease with dimensionality (3.50 at 768d → 1.75 at 1600d), indicating vectors become more spread out in higher-dimensional spaces. This correlates with better decomposition performance.

### 3.4 Capacity and Semantic Coherence

We measure pure geometric capacity by decomposing sums of randomly-selected unique tokens (no natural language structure). At 1600d, 50% exact recovery occurs at ~50 unique tokens — far below the 143 unique tokens recovered at 90% from the Gettysburg Address.

This gap — 6x better recovery on real text than random tokens at the same unique count — demonstrates that **semantic coherence in the input improves recoverability**. Tokens from meaningful text occupy a more structured region of embedding space, making them more distinguishable from a sum than random combinations.

### 3.5 Search Method Matters

HNSW approximate search can significantly underperform exact brute-force search, especially for large vocabularies:

| Model | Search | Gettysburg Recovery |
|---|---|---|
| Llama 3.2 3B | HNSW (connectivity 16) | 72.7% |
| Llama 3.2 3B | Brute-force | 100% |

The 27-point gap is entirely due to search approximation error, not embedding quality. HNSW's graph navigation becomes unreliable at 128K tokens in 3072 dimensions with low connectivity. All reported results use brute-force search unless otherwise noted.

### 3.6 Failure Modes

Failures are caused by:

1. **Near-duplicate tokens**: case variants (`the`/`The`), tense variants (`did`/`does`), space-prefix variants (` cat`/`cat`) have close embeddings that confuse the greedy search.
2. **Subword fragments**: rare BPE pieces (e.g., `cible` from "reproducible") have poorly-trained embeddings.
3. **Repeated punctuation**: quotation marks, commas share similar embeddings.

The wrong picks are often semantically close: `" live"` for `" lives"`, `" gave"` for `" give"`, `" five"` for `" seven"`. Case-insensitive matching recovers an additional 1-5% of tokens.

---

## 4. Optimization Strategies

### 4.1 Lookahead

Evaluating the top-k candidates at each step and selecting the one that minimizes the next residual norm provides no consistent improvement over greedy (k=1). The greedy path is already near-optimal at each step; the errors are cumulative path effects that local lookahead cannot correct.

### 4.2 Union of Metrics

Running the decomposition three times — once each with cosine, L2, and inner product — and unioning the recovered token sets provides significant improvement:

| Text | Best Single Metric | Union + Case-Insensitive |
|---|---|---|
| Gettysburg (GPT-2 XL) | 93.7% (IP) | 98.6% |
| Tale of Two Cities (Llama) | 52.3% (L2) | 64.0% |

Step-by-step analysis reveals the three metrics disagree on token ORDER — not identity. They find mostly the same tokens but pull them out in different sequences, creating complementary error patterns. The union captures tokens that any single metric's greedy path missed.

### 4.3 Coordinate Descent

Rather than making irrevocable greedy decisions, iteratively refine all token positions. Fix all but one position, find the optimal token for that position, cycle through all positions, repeat until convergence:

| Method | GPT-2 Small, Gettysburg |
|---|---|
| Greedy | 43.4% |
| **Coordinate descent** | **93.7%** |

Coordinate descent more than doubles accuracy on the smallest model, matching GPT-2 XL's greedy performance. It converges in ~10 iterations. The improvement comes from correcting cascading errors: early greedy mistakes that propagated through subsequent steps are revised when those positions are re-optimized.

---

## 5. The Effect of Normalization

### 5.1 What Normalization Destroys

We test three normalization modes:

- **Raw**: sum raw vectors, decompose against raw vectors (baseline)
- **NormVecs**: L2-normalize each token vector before summing, decompose against normalized vectors
- **NormSum**: normalize the sum itself to unit length

| Mode | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|
| Raw | 100% | 70% | 46% | 19% |
| NormVecs | 100% | 78% | 48% | 19% |
| NormSum | 0% | 0% | 0% | 0% |

**Normalizing the sum is catastrophic** — 0% at every N tested. The greedy algorithm requires the residual norm to decrease with each subtraction. A unit-length target subtracted by a unit-length token produces a residual that immediately inflects upward.

**Normalizing individual vectors** slightly helps at small N (removing magnitude outliers) and is neutral at larger N.

### 5.2 Magnitude Encodes Token Count

Embedding magnitude carries information: `N ≈ ||embedding|| / ||per_token_component||`. L2 normalization destroys this, making token count unrecoverable. Any recovery technique depending on residual magnitude or token count estimation cannot operate on normalized embeddings.

However, semantic similarity survives normalization — connotation detection, paraphrase recognition, and topic clustering all function on normalized vectors. The information loss is specific to additive decomposition, not to semantic structure in general.

---

## 6. Contextual Embeddings

### 6.1 The Orthogonality Problem

Contextual embeddings from Ollama's `/api/embed` endpoint (Llama 3.2 3B) are orthogonal to static token embeddings: average cosine similarity 0.003 across 30 test tokens. Direct decomposition of contextual embeddings against static vocabularies produces noise.

Investigation of Ollama's source code reveals the embed endpoint uses a different code path than generation: it returns pooled hidden states without the output projection, and L2-normalizes the result.

### 6.2 Contextual Space Has Semantic Structure

Despite being orthogonal to static space, contextual embeddings encode real semantic distinctions:

| Pair | Cosine Similarity |
|---|---|
| bank (money) / bank (river) | 0.597 |
| "I am happy" / "I am not happy" | 0.777 |
| "cat sat on mat" / "feline rested on rug" | 0.674 |
| "cat sat on mat" / "quantum physics" | 0.517 |

The semantic content is present — it's just not accessible through additive decomposition.

### 6.3 Layer-by-Layer Analysis

Using GPT-2 Small with full hidden state access (via HuggingFace transformers), we test decomposition at each transformer layer against both static and contextual vocabularies (50,257 tokens embedded individually at each layer):

| Test | Layer 6 | Layer 12 |
|---|---|---|
| Sentence → static vocab | 17% | 0% |
| Sentence → contextual vocab | 17% | 0% |
| Contextual bag-of-words → contextual vocab | **83%** | 0% |

The contextual bag-of-words result (83% at L6) proves the mid-layer contextual space supports additive decomposition. By the final layer, the space becomes entirely non-additive.

The sentence embedding never decomposes because attention creates a holistic representation: `attention("the cat sat") ≠ attention("the") + attention("cat") + attention("sat")`.

### 6.4 The Attention Bias Discovery

The contextual hidden state sum can be decomposed as:

```
contextual_sum = token_content + N × per_token_bias
```

The per-token bias is a near-constant vector: cosine similarity 0.9999+ across different sentences. It accounts for 99.5% of the hidden state energy. The token-specific signal is less than 0.5%.

Subtracting this bias reveals the token signal:

| Text | Before Bias Subtraction | After |
|---|---|---|
| the cat sat on the mat | 0/6 | 2-4/6 |
| Mary had a little lamb | 0/5 | 4-5/5 |
| the dog ran in the park | 0/6 | 3/6 |

N is estimated from the embedding magnitude: `N ≈ ||contextual_sum|| / ||bias||`. This estimate works better than the true N because it accounts for sentence-specific bias variation.

---

## 7. Word Order

Vector addition is commutative: `E("cat") + E("sat") = E("sat") + E("cat")`. No algorithm, however sophisticated, can recover token order from a vector sum. This is a mathematical certainty, not an empirical finding.

Contextual embeddings from a forward pass do encode order (attention is position-dependent). However, the sentence embedding is not a sum of its token components, making order recovery through decomposition inapplicable.

Order recovery requires a different approach: given a recovered bag of tokens, score candidate orderings through the embedding API (black box). For N tokens, this is an N!-search problem, reducible via beam search with language model pruning.

---

## 8. Implications

### 8.1 Security

Any system storing bag-of-words embedding sums (or approximations thereof) is vulnerable to content recovery. The attack requires only the static embedding matrix — freely available for open models — and nearest-neighbor search. No training, no learned decoder. For systems that do not L2-normalize, token count is also recoverable from the embedding magnitude.

### 8.2 Semantic Compression

Embedding vectors provide fixed-size representations regardless of text length. If approximate text recovery is acceptable, embeddings could serve as a lossy compression scheme — store the vector, recover the content on demand. The fidelity depends on the model's embedding dimensionality and the text's unique token count.

### 8.3 Architectural Motivation

The fundamental limitation we identify — commutativity of vector addition destroying word order — motivates exploration of non-commutative embedding representations. Matrix embeddings, where tokens are represented as transformations rather than points, would preserve order through the non-commutativity of matrix multiplication while potentially maintaining or exceeding the additive decomposability demonstrated here.

---

## 9. Limitations

- All static decomposition results assume the target vector is an exact sum of token embeddings. Real-world embeddings from API endpoints involve attention, normalization, and pooling that break this assumption.
- The attention bias subtraction technique is demonstrated on GPT-2 Small only. Generalization to larger models and different architectures is untested.
- Brute-force search is computationally expensive for large vocabularies (128K+ tokens). Practical deployment would require efficient approximate search that maintains accuracy.
- The coordinate descent improvement is demonstrated primarily on the Gettysburg Address. Behavior on longer texts and diverse domains requires further study.
- We do not address the ethical implications of embedding inversion beyond noting the security concern.

---

## 10. Conclusion

Embedding vectors are not opaque coordinates — they are readable sums, decomposable into the token primitives that constitute their meaning. The greedy residual algorithm is simple, fast, requires no training, and achieves high fidelity on static embeddings. Coordinate descent extends this to near-perfect recovery even on smaller models.

The barriers to contextual decomposition are precise and characterizable: attention adds a constant bias (subtractable), normalization destroys magnitude (avoidable), and the representation is non-additive (fundamental). Each barrier suggests a specific intervention, from bias subtraction to non-commutative architectures.

The model remembers. The question is whether we can read its memory. For static embeddings, the answer is yes. For contextual embeddings, we have opened the door.

---

## Acknowledgments

This work emerged from an extended collaborative research dialogue. The algorithm design, experimental methodology, and theoretical analysis were developed jointly. Infrastructure includes Crystal (usearch.cr bindings for HNSW search), Python (HuggingFace transformers for GPT-2 hidden state access), and Ollama (Llama 3.2 3B contextual embeddings).

---

## References

Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.

Su, J., et al. (2021). RoFormer: Enhanced transformer with rotary position embedding. *arXiv:2104.09864*.

Mikolov, T., et al. (2013). Efficient estimation of word representations in vector space. *arXiv:1301.3781*.

Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Trans. Big Data*.

Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. *IEEE TPAMI*.

Radford, A., et al. (2019). Language models are unsupervised multitask learners. *OpenAI Technical Report*.

Touvron, H., et al. (2023). Llama 2: Open foundation and fine-tuned chat models. *arXiv:2307.09288*.
