# Semantic Tracing: Recovering Token Content from Embedding Vectors via Greedy Residual Decomposition

**Thomas Sawyer**^1 **& Claude Opus 4.6**^2
*^1 Independent Researcher · ^2 Anthropic*

*April 2026*

---

## Abstract

We present *semantic tracing*, a training-free method for recovering token content from embedding vectors via greedy residual decomposition. The algorithm iteratively finds the nearest token embedding to a residual vector, subtracts it, and repeats. On static token-sum embeddings, this approach recovers bag-of-words content with high fidelity — 100% of unique tokens on short texts and 90%+ on medium texts (Gettysburg Address, 143 unique tokens) using GPT-2 XL (1600d). Coordinate descent optimization improves recovery from 43% to 97% on smaller models. We observe a sharp accuracy threshold between 1024 and 1280 dimensions in the GPT-2 family. Union of multiple distance metrics (cosine, L2, inner product) further improves recovery by exploiting complementary greedy paths.

For contextual embeddings, we observe a dominant near-constant bias component in transformer hidden states, accounting for approximately 99.5% of hidden state energy (GPT-2 Small, layer 6). Subtracting this component enables partial token recovery (17-80%) from representations that previously yielded 0%. Embedding magnitude encodes token count, and L2 normalization — standard in embedding APIs — destroys both magnitude and the ability to perform additive decomposition.

Vector addition is commutative: word order is mathematically unrecoverable from any sum of static embeddings. We discuss implications for embedding security, semantic compression, and the motivation for non-commutative embedding architectures.

---

## 1. Introduction

Embedding spaces are typically treated as opaque. Distances and directions carry meaning, but there is no standard method to recover natural language from an arbitrary point in the space. We ask: given a vector **v** that is a sum of static token embeddings, can we recover the constituent tokens — and how does this capability extend (or fail to extend) to contextual embeddings produced by transformer forward passes?

We investigate this question empirically across two model families (GPT-2 at four scales, Llama 3.2 3B), three distance metrics, multiple optimization strategies, and both static and contextual embedding spaces. The results are nuanced: static token-sum embeddings are often directly decomposable into their constituent tokens; contextual embeddings are not, though partial recovery is possible after specific interventions. Recoverability depends on embedding dimensionality, vocabulary size, normalization, and the type of embedding.

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

At each step, find the token whose embedding is nearest to the current residual, subtract it, and repeat. Terminate when the residual norm begins increasing (inflection stopping condition), indicating the signal is exhausted. Repeated tokens are allowed in the recovered multiset.

### 2.2 Norm Inflection as Automatic N-Finder

For exact sums, the residual norm monotonically decreases to zero. For non-exact targets (concept arithmetic, contextual embeddings), the norm decreases while meaningful tokens are being recovered, then inflects upward when the algorithm begins chasing noise. This inflection automatically determines the number of tokens N without prior knowledge of sequence length.

### 2.3 Distance Metrics

We evaluate three metrics for the nearest-neighbor search:

- **Cosine similarity**: direction-only, ignores magnitude
- **L2 distance**: penalizes both direction and magnitude mismatch
- **Inner product**: rewards directional alignment weighted by magnitude

No single metric dominates across all conditions (Section 4.2).

### 2.4 Greedy Decomposition as Geometric Probe

Greedy residual decomposition has no hidden parameters. Unlike a trained decoder (e.g., Vec2Text), which can learn to work around geometric obstacles without revealing them, greedy decomposition fails transparently. When it cannot recover a token, that token is not separable from the residual at that step — providing an interpretable probe of geometric separability in the embedding space.

Coordinate descent (Section 4.3) serves as a complementary probe. It finds the same tokens greedy finds but corrects path-dependent errors, confirming the tokens were geometrically present but reached by a suboptimal sequence of choices. Together, the two methods triangulate the geometry: greedy reveals what is accessible step-by-step; coordinate descent reveals what is accessible globally. Their failures are as informative as their successes.

---

## 3. Static Embedding Decomposition

### 3.1 Experimental Setup

We extract static token embedding matrices (`wte.weight`) from the GPT-2 family (Small 768d, Medium 1024d, Large 1280d, XL 1600d; all 50,257 tokens) and Llama 3.2 3B Instruct (3,072d; 128,256 tokens). For GPT-2, embeddings are extracted from HuggingFace safetensors files. For Llama, we parse the GGUF model file and dequantize from Q6_K to float32.

Search uses HNSW approximate nearest-neighbor (USearch library, cosine metric, f16 quantization) for GPT-2, and brute-force exact search for Llama (HNSW proved unreliable at 128K vocabulary scale — see Section 3.5). Texts are tokenized with GPT-2's BPE tokenizer or greedy longest-match for Llama.

**Recovery metric**: unique token recovery — the fraction of distinct token IDs in the original that appear in the recovered set. We also report total recovery (counting duplicate occurrences) and case-insensitive matching where noted. Punctuation and space-prefixed tokens are treated as distinct IDs unless case-insensitive matching is specified.

**Reproducibility note**: HNSW index construction involves random graph initialization, introducing ~3-5% variance between index builds. All HNSW-based results are representative of typical runs. Brute-force results are deterministic and exactly reproducible. Where precision matters, we report brute-force numbers as authoritative.

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

### 3.3 A Sharp Dimensionality Threshold

We observe an abrupt accuracy jump between 1024d and 1280d. On the Gettysburg Address: 40.6% → 89.5%. On Mary Had a Little Lamb: 68.4% → 100%. This is not smooth scaling; in our experiments, recovery exhibits a threshold-like transition.

One interpretation: below the threshold, near-duplicate tokens (case variants, tense variants, space-prefix variants) cannot be distinguished by the greedy search. Above it, the additional dimensions provide enough room for these close tokens to separate. The sharpness is consistent with separation being a threshold phenomenon in high-dimensional geometry.

| Dims | Avg Token Norm | Gettysburg Unique Recovery |
|---|---|---|
| 768 | 3.50 | 42.0% |
| 1024 | 3.22 | 40.6% |
| 1280 | 1.92 | 89.5% |
| 1600 | 1.75 | 90.2% |

GPT-2 Medium (1024d) anomalously performs worse than Small (768d) across all tests, suggesting training dynamics contribute independently of dimension count. Average token norms decrease with dimensionality (3.50 → 1.75), consistent with tokens spreading further apart in higher dimensions.

### 3.4 Semantic Coherence Amplifies Recovery

A control experiment measuring pure geometric capacity — decomposing sums of randomly-selected unique tokens with no linguistic structure — reveals a substantial gap. At 1600d, 50% exact recovery occurs at ~50 random unique tokens. The Gettysburg Address, with 143 unique tokens, achieves 90%.

In our experiments, naturally occurring text is substantially more recoverable than random token sets of comparable size — approximately 6x at 143 unique tokens. This suggests that training created embeddings where semantically coherent token combinations are more separable than random ones. The "capacity" of an embedding space for additive decomposition appears to depend not only on dimensionality but on the structure of the input.

### 3.5 Search Method Matters

HNSW approximate search can significantly underperform exact brute-force search, especially for large vocabularies:

| Model | Search | Gettysburg Recovery |
|---|---|---|
| Llama 3.2 3B | HNSW (connectivity 16) | 72.7% |
| Llama 3.2 3B | Brute-force | 100% |

The 27-point gap is entirely due to search approximation error, not embedding quality. HNSW's graph navigation becomes unreliable at 128K tokens in 3072 dimensions with the tested connectivity. All reported results use brute-force search unless otherwise noted.

### 3.6 Failure Modes

Failures are consistently caused by:

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
| Gettysburg (GPT-2 XL, brute-force) | 93.7% (IP) | 98.6% |
| Tale of Two Cities (Llama, brute-force) | 52.3% (L2) | 64.0% |

Step-by-step analysis reveals the three metrics disagree on token ORDER — not identity. They find mostly the same tokens but pull them out in different sequences, creating complementary error patterns. The union captures tokens that any single metric's greedy path missed.

### 4.3 Coordinate Descent

Rather than making irrevocable greedy decisions, we iteratively refine all token positions. For each position: fix all other positions, compute the ideal vector for that position (`target - sum_of_others`), find the nearest token to that ideal, and swap if it improves total distance. Cycle through all positions and repeat until no position improves.

Initialization: the greedy decomposition result. Convergence criterion: no swap improves the objective in a full cycle.

| Method | GPT-2 Small, Gettysburg |
|---|---|
| Greedy | 43.4% |
| **Coordinate descent** | **97.2%** |

Coordinate descent more than doubles accuracy on the smallest model, exceeding GPT-2 XL's greedy performance (90.2%). It converges in ~12 iterations on the Gettysburg Address (286 tokens). The improvement comes from correcting cascading errors: early greedy mistakes that propagated through subsequent steps are revised when those positions are re-optimized. The result is sensitive to the greedy initialization; the 97.2% figure is from a single run and may vary by ~3% across different HNSW index builds.

---

## 5. The Effect of Normalization

### 5.1 What Normalization Destroys

We test three normalization modes on GPT-2 XL with randomly-selected unique tokens (5 trials, brute-force search):

- **Raw**: sum raw vectors, decompose against raw vectors (baseline)
- **NormVecs**: L2-normalize each token vector before summing, decompose against normalized vectors
- **NormSum**: normalize the sum itself to unit length

| Mode | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|
| Raw | 100% | 70% | 46% | 19% |
| NormVecs | 100% | 78% | 48% | 19% |
| NormSum | 0% | 0% | 0% | 0% |

**Normalizing the sum eliminates recovery** — 0% at every N tested. The greedy algorithm requires the residual norm to decrease with each subtraction. A unit-length target subtracted by a unit-length token produces a residual that immediately inflects upward.

**Normalizing individual vectors** slightly helps at small N (removing magnitude outliers) and is neutral at larger N.

### 5.2 Magnitude Encodes Token Count

We observe that embedding magnitude carries information: `N ≈ ||embedding|| / ||per_token_component||`. L2 normalization destroys this relationship, making token count unrecoverable by magnitude-dependent methods.

Semantic similarity survives normalization — connotation detection, paraphrase recognition, and topic clustering all function on normalized vectors. The information loss is specific to additive decomposition and magnitude-dependent techniques, not to semantic structure in general.

---

## 6. Interpretive Framework: Norm-Stratified Representations

The findings on normalization (Section 5), attention bias (Section 7.4), and magnitude-as-N (Section 5.2) can be organized under a common geometric interpretation. We present this as an interpretive framework and hypothesis, not as a fully established mechanism.

### 6.1 Observed Layered Structure

In our experiments with GPT-2 Small at layer 6, a contextual embedding contains (at minimum):

1. **Vocabulary mean** (~3000 norm): the shared direction of all token embeddings. Centering removes this, revealing semantic structure underneath.
2. **Attention bias** (~2500 norm per token): a near-constant component across sentences (cosine similarity 0.9999+). Accounts for approximately 99.5% of the centered hidden state energy.
3. **Token content** (~16 norm per token): the residual semantic signal. Less than 0.5% of total energy.

We hypothesize that each transformer layer contributes its own bias stratum — one per attention + normalization operation. We compute the aggregate in this work; decomposing per-layer biases is an open direction.

### 6.2 Magnitude as Compositional Depth

The magnitude of a contextual embedding appears to encode how much was composed into it. A single token has magnitude ~`||bias||`. A 6-token sentence has magnitude ~`6 × ||bias||`. This relationship enables N estimation and suggests that L2 normalization discards not just scale but compositional depth.

### 6.3 Cross-Space Mapping

A linear projection W (768×768) maps contextual L6 embeddings to static embeddings. W is learned via least-squares regression on all 50,257 paired token embeddings: `W = argmin ||ctx @ W - static||`. Evaluated by nearest-neighbor identity: for each token, we check whether the nearest static embedding to the mapped contextual embedding is the correct token. The resulting projection preserves token identity with 89.9% accuracy (8,986/10,000 tokens tested).

The mapping succeeds on individual tokens but fails on sentence embeddings — consistent with attention's contribution being non-linear for composed representations even when approximately linear for individual tokens.

### 6.4 Quantization Sensitivity: A Testable Prediction

The framework predicts that quantization should selectively destroy the contextual token signal while leaving static decomposition unaffected. Rationale: the token signal lives in the low-order bits (0.5-1.1% of total energy), exactly the precision that quantization removes. Static decomposition uses the same quantized values for both target and vocabulary, so quantization errors cancel.

We test this by simulating f32, f16, int8, and int4 precision on the same GPT-2 Small hidden states (23-token input, "Mary had a little lamb..."):

**Static decomposition:**

| Precision | Recovery |
|---|---|
| f32 | 18/21 |
| f16 | 18/21 |
| int8 | 18/21 |
| int4 | 19/21 |

Unaffected — quantization errors cancel between target and vocabulary.

**Contextual decomposition after bias subtraction:**

| Precision | Recovery | Signal % |
|---|---|---|
| f32 | 6/21 | 0.85% |
| f16 | 4/21 | 0.85% |
| int8 | 0/21 | 0.85% |
| int4 | 0/21 | 0.90% |

The signal norm is preserved (~555 at all precisions) but the signal **direction** is corrupted by quantization noise at int8 and below. f16 loses 2 tokens compared to f32.

The signal-to-bias ratio decreases with token count (1.1% at 6 tokens, 0.85% at 23 tokens), consistent with more tokens compounding the bias and pushing the token signal deeper into the fractional precision. This suggests that **quantized models may be more resistant to contextual embedding inversion** — an unintentional defense with a geometric explanation.

### 6.5 The Norm-Stratified Residual Hypothesis

We propose that transformer hidden states are **norm-stratified compositions**: each attention + normalization operation contributes a bias stratum occupying a distinct magnitude range. Under this hypothesis:

- The outermost (highest-magnitude) strata encode model-wide computation artifacts
- Middle strata encode sentence-level context
- The innermost (lowest-magnitude) strata encode token-specific semantics

Quantization selectively destroys the deepest strata first — predicting that token-specific signals are disproportionately vulnerable to precision reduction. The quantization experiment (Section 6.4) is consistent with this prediction. The signal-to-bias ratio decreasing with token count (Section 6.2) follows directly: more tokens compound the outer strata, pushing the semantic signal deeper.

A full treatment — per-layer stratum decomposition, empirical norm distributions across model families, and the relationship between layer count and effective precision range — is deferred to subsequent work.

---

## 7. Contextual Embeddings

### 7.1 The Orthogonality Problem

Contextual embeddings from Ollama's `/api/embed` endpoint (Llama 3.2 3B) are orthogonal to static token embeddings: average cosine similarity 0.003 across 30 test tokens. Direct decomposition of contextual embeddings against static vocabularies produces noise.

Investigation of Ollama's source code reveals the embed endpoint uses a different code path than generation: it returns pooled hidden states without the output projection, and L2-normalizes the result.

### 7.2 Contextual Space Has Semantic Structure

Despite being orthogonal to static space, contextual embeddings encode real semantic distinctions:

| Pair | Cosine Similarity |
|---|---|
| bank (money) / bank (river) | 0.597 |
| "I am happy" / "I am not happy" | 0.777 |
| "cat sat on mat" / "feline rested on rug" | 0.674 |
| "cat sat on mat" / "quantum physics" | 0.517 |

The semantic content is present — it is not accessible through additive decomposition.

### 7.3 Layer-by-Layer Analysis

Using GPT-2 Small with full hidden state access (via HuggingFace transformers), we test decomposition at each transformer layer against both static and contextual vocabularies (50,257 tokens embedded individually at each layer):

| Test | Layer 6 | Layer 12 |
|---|---|---|
| Sentence → static vocab | 17% | 0% |
| Sentence → contextual vocab | 17% | 0% |
| Contextual bag-of-words → contextual vocab | **83%** | 0% |

The contextual bag-of-words result (83% at L6) shows that the mid-layer contextual space still supports additive decomposition. By the final layer, even additive decomposition fails.

The sentence embedding does not decompose at any layer because attention creates a holistic representation: `attention("the cat sat") ≠ attention("the") + attention("cat") + attention("sat")`.

### 7.4 The Attention Bias Discovery

We observe that the contextual hidden state sum can be decomposed as:

```
contextual_sum ≈ token_content + N × per_token_bias
```

The per-token bias is near-constant across sentences: cosine similarity 0.9999+ across 8 tested reference sentences of varying content and length. It accounts for approximately 99.5% of the hidden state energy (GPT-2 Small, layer 6). The token-specific signal is less than 0.5%.

Subtracting this bias reveals the token signal:

| Text | Before Bias Subtraction | After |
|---|---|---|
| the cat sat on the mat | 0/6 | 2-4/6 |
| Mary had a little lamb | 0/5 | 4-5/5 |
| the dog ran in the park | 0/6 | 3/6 |

N is estimated from the embedding magnitude: `N ≈ ||contextual_sum|| / ||bias||`. This estimate works better than the true N because it accounts for sentence-specific bias variation. Bias is computed as the average per-token residual from 4-8 reference sentences.

---

## 8. Word Order and the Full Recovery Pipeline

### 8.1 The Commutativity Barrier

Vector addition is commutative: `E("cat") + E("sat") = E("sat") + E("cat")`. No algorithm can recover token order from a vector sum. This is a mathematical certainty. We verified empirically: all permutations of token-position assignments produce identical sum vectors (differences of ~10⁻⁷, attributable to floating-point ordering).

### 8.2 A Three-Stage Pipeline (Proposed)

The findings from static and contextual experiments suggest a pipeline for full text recovery from contextual embeddings. Stages 1-2 are demonstrated; Stage 3 is proposed but untested.

**Stage 1: Bias subtraction → candidate tokens.** Subtract the precomputed attention bias (Section 6), estimate N from magnitude (Section 5.2) or binary search (Section 8.4), greedy decompose. Produces a partial bag of words (17-80%) plus semantic neighbors of missing tokens.

**Stage 2: Coordinate descent → refined bag.** Initialize from Stage 1, iteratively optimize each position (Section 4.3). On static embeddings this achieves 97%; on contextual, improvement is limited by landscape noise (Section 8.3).

**Stage 3: Order recovery via API (untested).** Given the refined bag, generate candidate orderings and score each through the embedding API: `score = similarity(embed(candidate), target)`. The N! search space could be constrained by language model perplexity pruning and coordinate descent over token positions. For N=10 with beam width 5 and 10 swap iterations: ~500 API calls.

### 8.3 The Residual Landscape

The optimization landscape differs between static and contextual spaces.

**Static space appears well-conditioned for greedy and coordinate-wise search.** For exact additive targets, the objective `||target - Σ E(tokens)||` has a single global minimum at zero. The residual norm monotonically decreases with each correct token subtraction. This is consistent with coordinate descent converging reliably (Section 4.3).

**Contextual space appears noisier and more path-dependent.** After bias subtraction, the target is approximately but not exactly a sum of centered contextual embeddings. The bias imprecision (~0.005% variation across sentences) creates residual noise of ~460 norm against a token signal of ~97 norm. This appears to produce local minima that can trap both greedy search and coordinate descent.

### 8.4 Bitonic Property and Binary Search for N

Sweeping the bias subtraction parameter N reveals a characteristic V-shaped (bitonic) residual curve: the residual decreases monotonically as N approaches the optimal value from below, reaches a minimum, then increases monotonically as N overshoots. For "the cat sat on the mat" (true N=6):

| N | Residual | Tokens Found |
|---|---|---|
| 4.0 | 2741 | 0/6 |
| 5.0 | 487 | 0/6 |
| 5.5 | **151** | **3/6** |
| 6.0 | 245 | 0/6 |
| 7.0 | 2405 | 0/6 |

The minimum coincides with maximum token recovery. This bitonicity enables binary search for optimal N in ~8 evaluations (log₂ of the search range). The property holds across all sentences tested, though the optimal N is consistently fractional (~0.5 below the true integer count), reflecting systematic bias underestimation.

For normalized embeddings (where magnitude is lost), the search range is bounded by plausible sentence lengths (3-100 tokens). Binary search over this range requires ~8 evaluations, each consisting of a bias subtraction and partial decomposition.

### 8.5 Future Directions for Contextual Recovery

Progress on contextual decomposition likely requires smoothing the landscape. Three paths we identify:

1. **Per-layer bias subtraction.** We computed the aggregate attention bias across all 12 transformer layers. Each layer likely contributes its own bias stratum. Subtracting these individually, from the outermost layer inward, could progressively reveal the token signal. This is untested but directly motivated by the aggregate bias result.

2. **Larger models with stronger token signals.** The token-to-bias energy ratio (0.5% at GPT-2 Small L6) may improve with model scale.

3. **Black-box API scoring** (Section 8.2, Stage 3), which bypasses the landscape entirely by using the model's own forward pass as the objective function.

---

## 9. Related Work

**Embedding inversion.** Morris et al. (2023) demonstrate that text embeddings can be inverted with high fidelity using Vec2Text, a trained corrector model that iteratively refines text hypotheses to match a target embedding. They achieve 92% recovery on 32-token inputs with a BLEU score of 97.3. Subsequent work extends this to zero-shot settings (Zero2Text) and cross-lingual contexts. These methods train neural models on (text, embedding) pairs. Our approach is fundamentally different: we use no training, only the static embedding matrix and nearest-neighbor search. Where Vec2Text learns to invert embeddings, we characterize the geometry that makes inversion possible — the dimensionality threshold, the role of normalization, the attention bias structure, the quantization sensitivity. Trained decoders exploit this geometry implicitly; our approach makes it explicit and measurable.

**Representation anisotropy.** Ethayarajh (2019) shows that contextual word representations are anisotropic — they occupy a narrow cone in embedding space. Our observation that contextual hidden states are dominated by a near-constant bias direction is consistent with this finding and provides a quantitative characterization: the dominant direction accounts for ~99.5% of energy (GPT-2 Small, layer 6).

**Embedding privacy.** The security implications of embedding inversion are increasingly recognized. Li et al. (2023) show generative models can recover full sentences from sentence embeddings. IronCore Labs (2024) catalog practical attack vectors against vector databases. Defenses include Gaussian noise injection and differential privacy, both of which degrade retrieval quality. Our finding that L2 normalization eliminates additive decomposition suggests normalization serves as a partial (if unintentional) defense against this specific attack vector.

**Concept arithmetic.** Mikolov et al. (2013) establish that embedding spaces support vector arithmetic (king - man + woman ≈ queen). Our work extends this in the opposite direction: rather than composing meaning from arithmetic, we decompose meaning via subtraction. The greedy residual algorithm is the operational inverse of embedding addition.

**Hidden-state probing.** Conneau et al. (2018) and subsequent work use linear probes to extract linguistic properties from hidden states. Our approach is complementary: rather than training probes to predict properties, we directly decompose the vector into its token constituents, requiring no training data or supervision.

---

## 10. Implications

### 10.1 Security

Systems storing bag-of-words embedding sums (or approximations thereof) may be vulnerable to content recovery using only the static embedding matrix — freely available for open models — and nearest-neighbor search. No training or learned decoder is required. For systems that do not L2-normalize, token count may also be recoverable from the embedding magnitude.

### 10.2 Semantic Compression

Embedding vectors provide fixed-size representations regardless of text length. If approximate text recovery is acceptable, embeddings could serve as a lossy compression scheme — store the vector, recover the content on demand. The fidelity depends on the model's embedding dimensionality and the text's unique token count.

### 10.3 Architectural Motivation

The fundamental limitation we identify — commutativity of vector addition destroying word order — motivates exploration of non-commutative embedding representations. Matrix embeddings, where tokens are represented as transformations rather than points, would preserve order through the non-commutativity of matrix multiplication while potentially maintaining or exceeding the additive decomposability demonstrated here.

---

## 11. Limitations

- All static decomposition results assume the target vector is an exact sum of token embeddings. Real-world embeddings from API endpoints involve attention, normalization, and pooling. We address these barriers (Sections 5-7) but full contextual recovery remains partial.
- The attention bias observation is demonstrated on GPT-2 Small only. The bias is near-constant (cosine similarity 0.9999+ across 8 tested sentences), but generalization to larger models and different architectures requires verification.
- The layered bias structure (Section 6.5) is a hypothesis motivated by our observations, not a demonstrated mechanism. Per-layer decomposition is untested.
- Brute-force search is computationally expensive for large vocabularies (128K+ tokens). HNSW approximate search introduces significant accuracy loss at this scale. Practical deployment requires either smaller vocabularies or improved approximate search.
- The coordinate descent result (97.2%) is primarily demonstrated on one text (Gettysburg Address) with GPT-2 Small. The technique is general but scaling behavior requires further study. The result depends on the greedy initialization and may vary by ~3% across runs.
- The quantization experiment simulates precision reduction on a single model and text. The prediction that quantization degrades contextual recovery more than static recovery is supported by our tests but has not been verified across model families.
- This work raises security concerns about embedding storage systems. We note the concern but do not provide a full threat model or mitigation framework.

---

## 12. Conclusion

For static token-sum embeddings, greedy residual decomposition often recovers constituent tokens directly, requiring no training and no learned decoder. Coordinate descent extends this to near-complete recovery even on smaller models. Together, these methods serve as interpretable probes: their transparent failure modes characterize the geometric separability of the embedding space.

The barriers to contextual decomposition are precise and characterizable: we observe a dominant bias-like component (subtractable), normalization destroys magnitude information (avoidable in principle), and attention creates non-additive representations (fundamental). Each barrier suggests a specific intervention.

These observations suggest a deeper geometric structure to transformer computation — one where magnitude encodes not just token count but compositional depth across layers. The norm-stratified residual hypothesis (Section 6.5) offers a framework that organizes our empirical findings and generates testable predictions, pointing toward a geometric theory of transformer representations.

The model remembers. The question is whether we can read its memory. For static embeddings, the answer is yes. For contextual embeddings, we have opened the door.

---

## Acknowledgments

This work emerged from an extended collaborative research dialogue. The algorithm design, experimental methodology, and theoretical analysis were developed jointly. Infrastructure includes Crystal (usearch.cr bindings for HNSW search), Python (HuggingFace transformers for GPT-2 hidden state access), and Ollama (Llama 3.2 3B contextual embeddings). All code and experiment outputs are available at https://github.com/trans/semtrace.

---

## References

Conneau, A., Kruszewski, G., Lample, G., Barrault, L., & Baroni, M. (2018). What you can cram into a single $&!#* vector: Probing sentence embeddings for linguistic properties. *ACL*.

Ethayarajh, K. (2019). How contextual are contextualized word representations? Comparing the geometry of BERT, ELMo, and GPT-2 representations. *EMNLP*.

Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Trans. Big Data*.

Li, H., et al. (2023). Sentence embedding leaks more information than you expect: Generative embedding inversion attack to recover the whole sentence. *arXiv*.

Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. *IEEE TPAMI*.

Mikolov, T., et al. (2013). Efficient estimation of word representations in vector space. *arXiv:1301.3781*.

Morris, J. X., Kuleshov, V., Shmatikov, V., & Rush, A. M. (2023). Text embeddings reveal (almost) as much as text. *EMNLP 2023*. arXiv:2310.06816.

Radford, A., et al. (2019). Language models are unsupervised multitask learners. *OpenAI Technical Report*.

Su, J., et al. (2021). RoFormer: Enhanced transformer with rotary position embedding. *arXiv:2104.09864*.

Touvron, H., et al. (2023). Llama 2: Open foundation and fine-tuned chat models. *arXiv:2307.09288*.

Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
