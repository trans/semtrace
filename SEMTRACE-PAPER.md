# Semantic Tracing: Recovering Token Content from Embedding Vectors via Greedy Residual Decomposition

**Thomas Sawyer**^1 **& Claude Opus 4.6**^2
*^1 Independent Researcher · ^2 Anthropic*

*April 2026*

---

## Abstract

We present *semantic tracing*, a training-free method for recovering token content from embedding vectors via greedy residual decomposition. The algorithm iteratively finds the nearest token embedding to a residual vector, subtracts it, and repeats. On static token-sum embeddings, this approach recovers bag-of-words content with high fidelity — 100% of unique tokens on short texts and 90%+ on medium texts (Gettysburg Address, 143 unique tokens) using GPT-2 XL (1600d). Coordinate descent optimization improves recovery from 43% to 97% on smaller models. We observe a sharp accuracy threshold between 1024 and 1280 dimensions in the GPT-2 family. Union of multiple distance metrics (cosine, L2, inner product) further improves recovery by exploiting complementary greedy paths.

For contextual embeddings, we identify and isolate the **position-0 attention sink** — an architectural artifact in which the first position's hidden state is repurposed as a scratch space for unused attention budget, growing to ~40× the magnitude of other positions. Earlier observations of a dominant "99.5% bias component" in hidden states reflect this single-position phenomenon rather than a distributed per-token bias. With sink removal and a corrected per-token bias subtraction, the trailing positions of "the cat sat on the mat" decompose at 100% from layer 1 through layer 9 of GPT-2 Small, dropping to 40% only at the final transformer block. Longer sentences (16 trailing tokens) recover 79% at L1 with smooth degradation through depth. Embedding magnitude encodes token count, and L2 normalization — standard in embedding APIs — destroys both magnitude and the ability to perform additive decomposition.

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

In our experiments with GPT-2 Small, a contextual hidden state at layer L contains, at minimum, three components that we can now identify and measure separately:

1. **Vocabulary mean** (~3000 norm): the shared direction of all token embeddings. Centering removes this.
2. **Attention sink at position 0** (Section 7.5): a position-specific artifact in which the model uses the first token's hidden state as a scratch space, accumulating ~2700 norm of MLP-projected content unrelated to position 0's actual token. This component dominates any sum-of-positions and was responsible for what we earlier interpreted as a single dominant "attention bias" of ~99.5% of total energy.
3. **Per-token additive bias** (~10–60 norm, growing with layer): a small, near-constant component added to *every* position by each transformer block. This is the actual layer-by-layer additive sediment. With the sink excluded, the per-token bias at GPT-2 Small grows roughly linearly: 16 (L1) → 21 (L3) → 28 (L5) → 37 (L7) → 45 (L9) → 60 (L11).
4. **Token content** (~50–70 norm per non-sink position): the per-position semantic signal, which remains the dominant component of any position other than position 0.

The earlier framing of "99.5% of energy is bias" measured the aggregate of (2) and (3) without distinguishing them, which obscured the much smaller and better-behaved component (3). Section 7.5 documents the sink discovery and its consequences.

### 6.2 Magnitude as Compositional Depth

The magnitude of a sum-of-positions contextual embedding scales with the number of positions, primarily because each non-sink position contributes a hidden state of comparable magnitude (~50–70 norm at the trailing positions of GPT-2 Small) and these magnitudes add. This relationship enables N estimation from the embedding magnitude when the magnitude is preserved (i.e., before any L2 normalization step).

Note that under our earlier (contaminated) bias model, this scaling was attributed entirely to "N copies of an attention bias." With the corrected understanding, the position-0 sink contributes a roughly fixed ~2700 norm regardless of N, and the trailing positions contribute ~60 norm each — so the total magnitude is approximately `2700 + 60(N − 1)`, which is approximately linear in N for moderate sentence lengths but with a large additive offset from the sink. L2 normalization destroys both the offset and the per-token contribution, eliminating any ability to recover N from the magnitude alone.

### 6.3 Cross-Space Mapping

A linear projection W (768×768) maps contextual L6 embeddings to static embeddings. W is learned via least-squares regression on all 50,257 paired token embeddings: `W = argmin ||ctx @ W - static||`. Evaluated by nearest-neighbor identity: for each token, we check whether the nearest static embedding to the mapped contextual embedding is the correct token. The resulting projection preserves token identity with 89.9% accuracy (8,986/10,000 tokens tested).

The mapping succeeds on individual tokens but fails on sentence embeddings — consistent with attention's contribution being non-linear for composed representations even when approximately linear for individual tokens.

### 6.4 Quantization Sensitivity: A Testable Prediction

The framework predicts an asymmetry between static and contextual decomposition under quantization. **Static decomposition should be unaffected**, because both the target and the vocabulary it is compared against are quantized identically — the rounding errors cancel on both sides of the subtraction. **Contextual decomposition should be progressively impaired** as quantization becomes more aggressive, because the per-token-distinguishing signal in transformer hidden states lives in small directional differences that require precision to represent. The dominant directions (vocabulary mean, attention sink, gross semantic class) survive quantization easily because they are encoded by larger values; the fine per-token contributions do not, because they live in smaller values that the coarsened grid no longer resolves.

We test this on a single model (GPT-2 Small) and a single 23-token text ("Mary had a little lamb..."), simulating f32, f16, int8, and int4 precision.

**Static decomposition:**

| Precision | Recovery |
|---|---|
| f32 | 18/21 |
| f16 | 18/21 |
| int8 | 18/21 |
| int4 | 19/21 |

Unaffected at every precision tested, as predicted.

**Contextual decomposition (sink-skip + N-debias, L1):**

| Precision | Recovery |
|---|---|
| f32 | 13/21 |
| f16 | 13/21 |
| int8 | 13/21 |
| int4 | 4/21 |

f16 and int8 are indistinguishable from f32 in our test. Recovery collapses at int4. The threshold at which the per-token signal is destroyed lies somewhere between int8 and int4 in this single measurement; we make no claim about where that threshold sits in other models or longer texts. The general prediction — that contextual decomposition degrades with sufficiently aggressive quantization while static decomposition does not — holds qualitatively.

The implication for "quantized models as an unintentional defense" against embedding inversion is therefore conditional: at the precision where our test finds the collapse (int4), aggressive quantization does provide meaningful resistance. At more conservative precision levels common in production (f16, int8), the defense did not materialize in our test. Whether the threshold is consistent across model sizes, architectures, and text lengths is an open empirical question that this experiment does not answer.

### 6.5 The Norm-Stratified Residual Hypothesis

We propose that transformer hidden states are **norm-stratified compositions**: each transformer block contributes a thin additive sediment to the residual stream, and these strata stack across layers. Once the position-0 attention sink artifact is removed (Section 7.5), this stratified structure becomes directly visible in the per-token bias growth.

In GPT-2 Small, with the sink excluded, the per-token bias grows monotonically and roughly linearly with depth:

| Layer | Per-token bias norm (sink-excluded) |
|---|---|
| L1 | 16 |
| L3 | 21 |
| L5 | 28 |
| L7 | 37 |
| L9 | 45 |
| L11 | 60 |

Each layer adds ~3–8 norm of new additive content shared across all positions. This is the strata: each block deposits a thin, near-constant shift on top of the previous block's residual, and these shifts accumulate. Earlier results that suggested the bias "saturates" around L3 at ~2700 norm (Section 6.6) measured the sum of this real per-layer growth plus the sink artifact, with the sink dominating once it forms.

**Testable predictions of the strata hypothesis (now newly accessible):**

1. **Layer peeling.** If the layer biases really are additive and stack inward, then sequentially subtracting them — outermost first — should approximately reconstruct the earlier-layer hidden state. Decomposition at L11 minus the L11 bias should resemble decomposition at L9; minus the L9 bias should resemble L7; and so on.
2. **Per-position correction.** The strata are near-constant across positions. Anything left over after debiasing must be position-specific — i.e., the per-position attention mixing, which Section 7.5 identifies as the residual problem after the sink is handled.
3. **Quantization vulnerability.** The thin sedimentary layers should be the first thing destroyed by precision reduction, since they live in small directional differences. Section 6.4's quantization experiment is consistent with this.

These predictions were not testable cleanly under the previous bias model because the sink artifact dominated all measurements. With sink removal, the experiments described above are well-defined and tractable; we propose them as the next experimental program for this framework.

### 6.6 Per-Layer Measurement

We measured the per-token bias at each of GPT-2 Small's transformer layers in two regimes: a "contaminated" measurement that includes the position-0 sink (the original measurement), and a "clean" measurement that excludes position 0 and uses a properly-built contextual vocabulary at position 1 of an `[EOT, token]` prefix pair (Section 7.6).

| Layer | Contaminated bias norm | Clean bias norm | Consistency (cosine) |
|---|---|---|---|
| L1 | 116.4 | 16.4 | 0.999+ |
| L3 | 2265.7 | 21.5 | 0.999+ |
| L5 | 2558.0 | 27.7 | 0.999+ |
| L7 | 2680.9 | 36.6 | 0.999+ |
| L9 | 2712.8 | 45.0 | 0.999+ |
| L11 | 2677.6 | 60.1 | 0.999+ |

The two measurements tell completely different stories. The contaminated bias jumps 20× between L1 and L3 and then plateaus around 2700 — the signature of the sink filling up to its asymptotic magnitude, then sitting there. The clean bias grows monotonically and roughly linearly across the entire network — the signature of layer-by-layer additive sediment in the residual stream.

The clean measurement is the one that matches the strata hypothesis and the one that supports decomposition. With the clean per-token bias subtracted (using N as the multiplier, see Section 7.7), recovery on "the cat sat on the mat" (5 trailing tokens) is:

| Layer | Recovery (N-debiased, sink-skipped) |
|---|---|
| L1 | **5/5** |
| L3 | **5/5** |
| L5 | **5/5** |
| L7 | **5/5** |
| L9 | **5/5** |
| L11 | 2/5 |

Recovery is essentially perfect through layer 9 and only degrades in the final two transformer blocks. The earlier reported numbers (1–4 of 6 across layers) reflected the contaminated bias model and the sink-polluted vocabulary, not a fundamental limit of the residual stream.

### 6.7 Token Signal Survival

To directly measure whether the static token signal persists through the transformer, we compute, at each layer, the cosine similarity between each position's hidden state and the corresponding static token embedding, along with the rank of the correct token among all 50,257 vocabulary entries.

| Layer | Avg Cosine to Static Token | Avg Rank (of 50,257) |
|---|---|---|
| L0 (embed) | 0.514 | 1 |
| L1 | 0.131 | 16 |
| L3 | 0.081 | 36 |
| L6 | 0.057 | 81 |
| L9 | 0.052 | 181 |
| L11 | 0.055 | 277 |
| L11 | 0.055 | 277 |
| L12 (post-ln_f) | -0.144 | 44,260 |

Through layers 0-11 (the residual stream), the correct token remains consistently in the top few hundred out of 50,257 — the top 0.5% of the vocabulary. The cosine similarity is weak (0.05-0.13) but the ranking is strong: the token identity signal persists, degraded by accumulated bias, but not destroyed.

At L12, the cosine inverts and the rank collapses. This is a representational discontinuity: in HuggingFace's GPT-2, `hidden_states[12]` is the output AFTER the final LayerNorm (`ln_f`), unlike `hidden_states[0-11]` which are pre-normalization residual stream states. The inversion reflects this transformation, not one more step of the same gradual process.

When we apply the output projection (`wte.T`) to each layer — applying `ln_f` only to layers 0-11, since L12 is already post-`ln_f` — L12 achieves the best prediction rank (373). The representation has been transformed from identity-encoding to prediction-encoding. The token signal is not destroyed; the final LayerNorm masks it by rotating the representation into a prediction-oriented basis.

This is the evidence for the norm-stratified structure: within the residual stream (L0-L11), each layer adds bias that buries the token signal while maintaining it in recoverable form. The final LayerNorm masks this structure, but the underlying additive decomposability persists (83% contextual BoW recovery at both L6 and L11, Section 7.3).

A full treatment across model families and the formal relationship between layer count and effective precision range is deferred to subsequent work.

---

## 7. Contextual Embeddings

### 7.1 The Orthogonality Problem

Contextual embeddings from Ollama's `/api/embed` endpoint (Llama 3.2 3B) are orthogonal to static token embeddings: average cosine similarity 0.003 across 30 test tokens. Direct decomposition of contextual embeddings against static vocabularies produces noise.

Investigation of Ollama's source code reveals the embed endpoint uses a different code path than generation: it returns pooled hidden states without the output projection, and L2-normalizes the result.

We initially interpreted this orthogonality as a fundamental property of contextual embedding spaces. The findings in Section 7.5 suggest that interpretation was at least partly an artifact: pooled hidden states include the position-0 attention sink, whose ~2700-norm magnitude dominates any pooling operation and is essentially uncorrelated with static token directions. With sink removal and a corrected vocabulary construction (Sections 7.5–7.7), the orthogonality breaks down substantially in GPT-2 Small. Whether the same correction recovers decomposability in pooled production endpoints (Llama, OpenAI text-embedding) remains untested but is now a well-defined experimental question.

### 7.2 Contextual Space Has Semantic Structure

Despite being orthogonal to static space, contextual embeddings encode real semantic distinctions:

| Pair | Cosine Similarity |
|---|---|
| bank (money) / bank (river) | 0.597 |
| "I am happy" / "I am not happy" | 0.777 |
| "cat sat on mat" / "feline rested on rug" | 0.674 |
| "cat sat on mat" / "quantum physics" | 0.517 |

The semantic content is present — it is not accessible through additive decomposition.

### 7.3 Layer-by-Layer Analysis (Initial Results)

Using GPT-2 Small with full hidden state access, we initially tested decomposition at each transformer layer against contextual vocabularies built by embedding each of the 50,257 tokens *individually* through the model and reading the hidden state at the requested layer.

**Important note**: In HuggingFace's GPT-2, `hidden_states[12]` is the output AFTER the final LayerNorm (`ln_f`), unlike `hidden_states[0-11]` which are residual stream states before that normalization. We report L11 (last transformer block, pre-ln_f) as the final comparable layer, and L12 (post-ln_f) separately.

The original results were:

| Test | L6 | L11 (pre-ln_f) | L12 (post-ln_f) |
|---|---|---|---|
| Sentence → static vocab | 17% | 0% | 0% |
| Sentence → contextual vocab | 17% | 0% | 0% |
| Contextual bag-of-words → contextual vocab | 83% | 83% | 0% |

These numbers are misleading in two ways that took us several rounds of experiments to identify:

1. **The "contextual vocabulary" was sink-corrupted.** Embedding a token individually means passing it as a single-token sequence, in which case it lands at position 0 — the attention sink position (Section 7.5). Every entry in this vocabulary was therefore a position-0 attention dump rather than a representative contextual embedding. The 83% bag-of-words result reproduces because both the test sum and the vocabulary entries were sink-corrupted in the same way; it does not indicate that the additive structure survives transformer mixing.

2. **The "sentence embedding" sum was dominated by the same sink at position 0 of the test sentence.** Position 0's hidden state grows to ~2700 norm by deep layers, while every other position stays at ~50–70. Summing the positions and decomposing reads the sink, not the sentence content.

The corrected layer-by-layer analysis appears in Sections 7.6–7.7. Briefly: with sink-skip and a corrected vocabulary, the trailing positions of "the cat sat on the mat" decompose at 5/5 from L1 through L9, dropping to 2/5 only at L11 — a completely different picture from the table above.

### 7.4 The Attention Bias Discovery (and What It Actually Was)

We initially observed that the sum of contextual hidden states across positions admits an approximate decomposition:

```
contextual_sum ≈ token_content + N × per_token_bias
```

with a "per-token bias" that was near-constant across sentences (cosine similarity 0.9999+ across 8 reference sentences), accounted for approximately 99.5% of the hidden state energy (GPT-2 Small, L6), and substantially improved decomposition when subtracted. Early results on three sentences:

| Text | Before Bias Subtraction | After |
|---|---|---|
| the cat sat on the mat | 0/6 | 2–4/6 |
| Mary had a little lamb | 0/5 | 4–5/5 |
| the dog ran in the park | 0/6 | 3/6 |

The discovery was real, but our characterization of *what* the bias is turned out to be wrong. The "near-constant component accounting for 99.5% of the energy" is not primarily an attention bias added to every position. It is overwhelmingly the position-0 attention sink (Section 7.5) — a single-position artifact whose ~2700-norm hidden state dominates any sum across positions.

The cosine consistency of 0.9999+ across reference sentences was therefore measuring how consistently the model creates a sink at position 0 (very consistently — it is an architectural property), not how consistently a per-token bias appears at all positions. With sink removal, a smaller and qualitatively different additive bias remains: ~16 norm at L1 growing linearly to ~60 at L11, near-constant across non-sink positions, and accounting for the actual layer-by-layer additive sediment described in Section 6.5.

The earlier recovery numbers (0/6 → 2–4/6 on "the cat sat on the mat") thus reflect partial recovery against an incorrect bias model. The corrected method (Sections 7.6–7.7) achieves 5/5 on the trailing tokens of the same sentence at every layer L1–L9.

### 7.5 The Attention Sink Discovery

To understand why a single bias vector handled the L1 → L3 transition so poorly (Section 7.4), we instrumented GPT-2 Small with hooks on every block component (layer norms, attention output, MLP output) and traced the per-position contribution of each block on a 6-token input ("the cat sat on the mat"). The result is unambiguous: **the catastrophic component is not a distributed bias at all but a single-position artifact at position 0.**

Per-position residual stream norms across the first four transformer blocks:

| Block | pos 0 (`the`) | pos 1 (` cat`) | pos 2 (` sat`) | pos 3 (` on`) | pos 4 (` the`) | pos 5 (` mat`) |
|---|---|---|---|---|---|---|
| input to L1 | 137 | 57 | 59 | 54 | 56 | 60 |
| input to L2 | 645 | 59 | 60 | 53 | 55 | 64 |
| input to L3 | 2573 | 67 | 68 | 57 | 60 | 71 |
| input to L4 | 2765 | 70 | 76 | 64 | 62 | 73 |

Position 0 alone grows from 137 to 2765 — a 20× increase. Every other position stays at norm ~50–75 across the entire trace. The growth is driven primarily by the MLP at each block, which adds ~500 → ~2300 → ~200 norm of new content to position 0 specifically while adding only ~10–16 to every other position. The MLP weights have specialized: when a position has the "I am the sink" signature, the MLP applies a large characteristic transformation; otherwise it applies a small one.

The attention pattern explains *why* position 0 receives this treatment. At every layer, every other query position attends substantially to position 0. By block 2:

```
  q0:   1.00   0.00   0.00   0.00   0.00   0.00
  q1:   0.94   0.06   0.00   0.00   0.00   0.00
  q2:   0.69   0.20   0.11   0.00   0.00   0.00
  q3:   0.51   0.08   0.30   0.11   0.00   0.00
  q4:   0.45   0.06   0.13   0.26   0.10   0.00
  q5:   0.34   0.03   0.07   0.24   0.20   0.11
```

Position 5 (` mat`, the most semantically loaded final token) puts 34% of its attention budget on position 0; position 4 puts 45%; position 3 puts 51%. These tokens are not attending to position 0 because position 0 contains useful information. They are attending to it because softmax forces every query to spend its full attention budget on *something*, and position 0 has been learned by the model as a designated dumping ground for unused attention budget.

This phenomenon was first formally characterized by Xiao et al. (2023) as the **attention sink**, and shown to be load-bearing for language model behavior — evicting the first few tokens from a streaming KV cache breaks model behavior catastrophically, not because the tokens contained important content but because removing them forces attention to redistribute and the model has not learned to function without its sink. In models that prepend a BOS token (Llama, etc.), BOS becomes the dedicated sink. In GPT-2 as commonly used (without explicit BOS prefixing), whatever token happens to land at position 0 is repurposed as the sink at runtime — the architectural pressure to have a sink overrides the semantic identity of the token actually sitting there.

**Consequences for our prior contextual results.** Three of our earlier findings turn out to be sink artifacts rather than genuine contextual properties:

1. The "near-constant 99.5% bias direction" of Section 7.4 is the position-0 sink dump, present in every sentence at the same magnitude because the sink forms identically every time.
2. The "contextual vocabulary" we built by passing each token individually through the model placed every entry at position 0, so every entry was itself a sink-pumped attention dump rather than a representative contextual embedding.
3. The "orthogonality" of pooled contextual embeddings to static space (Section 7.1) was driven by the sink mass dominating any pool that includes position 0.

The next two subsections describe the methodological correction.

### 7.6 The Sink-Skip Method

Two corrections suffice to neutralize the sink:

**Vocabulary correction.** Instead of building a contextual vocabulary by embedding each token individually (which places the token at position 0 and contaminates it with the sink), build it by passing `[prefix, token]` pairs through the model and reading the hidden state at *position 1*. The choice of prefix matters mildly: `<|endoftext|>` (the canonical sink token in GPT-2) gives a slightly cleaner separation than an arbitrary token like `the`, but both work.

**Target correction.** Instead of summing all positions of the target sentence and decomposing the sum, sum only positions 1 through N−1, excluding position 0. This removes the sink mass from the target as well. Optionally, prepend `<|endoftext|>` to the input sentence before forward-passing it; this moves the original first token from position 0 (where it would become the sink) to position 1 (where it is recoverable), at the cost of slightly more aggressive attention drainage at deeper layers.

Both corrections are simple, training-free, and require only a single decision about which prefix to use. With both applied, the per-token bias collapses from ~2700 norm to ~10–60 norm (Section 6.6) and decomposition becomes tractable across most of the residual stream.

### 7.7 Per-Layer Decomposition with Sink-Skip and N-Corrected Debias

Combining sink removal with the corrected debias multiplier (using N as the multiplier rather than the norm-ratio estimate `||sum||/||bias||`, which systematically under-debiases), we sweep every other layer of GPT-2 Small.

**Test sentence: "the cat sat on the mat" (5 trailing tokens after sink-skip):**

| Layer | Per-token bias norm | Trailing recovery (no debias) | Trailing recovery (N-debias) |
|---|---|---|---|
| L1 | 16 | 5/5 | **5/5** |
| L3 | 21 | 4/5 | **5/5** |
| L5 | 28 | 3/5 | **5/5** |
| L7 | 37 | 2/5 | **5/5** |
| L9 | 45 | 1/5 | **5/5** |
| L11 | 60 | 0/5 | 2/5 |

Recovery is essentially perfect from L1 through L9 with N-debias, dropping to 2/5 only at L11. With `<|endoftext|>` prepended to the target sentence, the previously-lost first content token is also recovered, giving 6/6 at L1, L3, and L5.

**Test sentence: 17-token sentence ("the quick brown fox jumps over the lazy dog while the cat watches from the windowsill"), 14 unique trailing tokens:**

| Layer | Trailing recovery (N-debias) |
|---|---|
| L1 | 11/14 (79%) |
| L3 | 9/14 |
| L5 | 8/14 |
| L7 | 6/14 |
| L9 | 7/14 |
| L11 | 4/14 |

Recovery is lower than on the short sentence and degrades more visibly with depth, but is meaningfully nonzero at every layer. Inspection of misses shows they are dominated by semantic neighbors (` hawk` for ` fox`, ` beside` for ` over`, ` chair` for ` mat`-context) rather than random tokens — the structure is intact, the resolution is degrading. Greedy decomposition also stops early on longer sentences before exhausting the search; coordinate descent and union-of-metrics polishing (Section 4) would likely improve these numbers further.

**What the sink-skip method demonstrates and what it does not.** The method demonstrates that the catastrophic difficulty of contextual decomposition reported in our earlier results was overwhelmingly an artifact of the position-0 attention sink, not a fundamental property of contextual representations. With the sink removed, the trailing positions of a sentence decompose cleanly through most of the network. The method does *not* demonstrate full contextual inversion: position 0 itself remains unrecovered (it has been transformed by ~12 rounds of MLP into a representation we cannot read directly), and longer sentences still degrade with depth due to per-position attention mixing that a single bias vector cannot handle. We discuss the residual problem and its proposed solution in Sections 8.2 and 8.5.

---

## 8. Word Order and the Full Recovery Pipeline

### 8.1 The Commutativity Barrier

Vector addition is commutative: `E("cat") + E("sat") = E("sat") + E("cat")`. No algorithm can recover token order from a vector sum. This is a mathematical certainty. We verified empirically: all permutations of token-position assignments produce identical sum vectors (differences of ~10⁻⁷, attributable to floating-point ordering).

### 8.2 A Three-Stage Pipeline (Proposed)

The findings from static and contextual experiments suggest a pipeline for full text recovery from contextual embeddings. Stages 1-2 are demonstrated; Stage 3 is proposed but untested.

**Stage 1: Sink-skip + N-debias → candidate tokens.** Forward-pass the target through the model with `<|endoftext|>` prepended (Section 7.6), exclude position 0 from the sum, subtract N × per-token bias (Section 7.7), greedy decompose against a sink-corrected contextual vocabulary at an early layer (L1–L5). Recovers ~100% on short sentences and ~80% on longer ones, plus semantic neighbors of misses.

**Stage 2: Coordinate descent → refined bag.** Initialize from Stage 1, iteratively optimize each position (Section 4.3). On static embeddings this achieves 97%; on contextual it should now be more effective than previously reported, since the landscape after sink removal is qualitatively cleaner than the sink-contaminated landscape Section 8.3 was characterizing.

**Stage 3: White-box forward verification → order and polish.** Given the refined bag and access to the embedding model's weights (the standard threat model for open-weight LLMs), generate candidate orderings and forward-pass each through the model, scoring `||target − forward(candidate)||` directly on hidden states or pooled output. White-box access enables gradient-based refinement (no rate limits, batchable, full hidden-state visibility) and is far more practical than the original black-box API scoring proposal. The N! search space can be constrained by language model perplexity pruning, coordinate-wise position swaps, and gradient steps.

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

The sink-skip and N-debias method (Section 7.7) handles two of the three components of attention's effect on the residual stream: the position-0 sink artifact and the per-token additive bias. The remaining problem is **per-position attention mixing**: at deep layers, each position's hidden state contains contributions from earlier positions, and a single per-token bias vector cannot undo that because the contamination is position-dependent. We identify four directions for addressing it.

1. **Layer peeling.** The strata hypothesis (Section 6.5), reframed in terms of clean per-layer biases, predicts that subtracting the L11 bias from a sentence's L11 hidden state should approximately produce the L9 hidden state, and so on inward through the network. This is a deterministic procedure that uses only the per-layer bias vectors (which we already compute). If it works, it converts deep-layer contextual hiddens into early-layer ones where decomposition is already known to work. The clean monotonic bias growth (16 → 21 → 28 → 37 → 45 → 60) is exactly the kind of additive sediment this would require. This is the most direct test of the strata model and should be the next experimental priority.

2. **White-box gradient refinement.** With model weights available, the inversion problem becomes a gradient-descent problem: candidates from Stage 1 can be refined by minimizing `||target − forward(candidate)||` directly with backprop. This is the foundation of Stage 3 in Section 8.2 and is much more powerful than the originally proposed API-scoring approach.

3. **Larger models with stronger token signals.** The token-to-bias energy ratio (clean version: ~0.5–1% at GPT-2 Small L6) may improve with model scale; the sink phenomenon may also be qualitatively different in models that prepend BOS during training.

4. **Direct per-position attention inversion.** Attention weights at layer L are a deterministic function of the L−1 hidden states. Once we have a confident reconstruction at one layer, we can in principle compute the attention pattern that produced the next layer and analytically subtract per-position contributions. This is closely related to layer peeling but uses the model's actual mixing operation rather than an approximation.

---

## 9. Related Work

**Embedding inversion.** Morris et al. (2023) demonstrate that text embeddings can be inverted with high fidelity using Vec2Text, a trained corrector model that iteratively refines text hypotheses to match a target embedding. They achieve 92% recovery on 32-token inputs with a BLEU score of 97.3. Subsequent work extends this to zero-shot settings (Zero2Text) and cross-lingual contexts. These methods train neural models on (text, embedding) pairs. Our approach is fundamentally different: we use no training, only the static embedding matrix and nearest-neighbor search. Where Vec2Text learns to invert embeddings, we characterize the geometry that makes inversion possible — the dimensionality threshold, the role of normalization, the attention bias structure, the quantization sensitivity. Trained decoders exploit this geometry implicitly; our approach makes it explicit and measurable.

**Representation anisotropy.** Ethayarajh (2019) shows that contextual word representations are anisotropic — they occupy a narrow cone in embedding space. Our findings refine this picture: a substantial portion of the apparent anisotropy in pooled representations is contributed by the position-0 attention sink (Section 7.5), a single-position artifact rather than a property of every position. Non-sink positions in our GPT-2 Small experiments occupy a much less anisotropic region of the residual stream.

**Attention sinks.** Xiao et al. (2023) characterize the attention sink phenomenon: language models learn to dump unused attention budget on the first few tokens of a sequence, and removing those tokens from a streaming KV cache breaks model behavior catastrophically. We rediscovered this phenomenon empirically while investigating contextual decomposition, and find that it is the dominant obstacle to additive recovery from sum-of-positions contextual embeddings. Our sink-skip method (Section 7.6) is essentially the same observation — that position 0 must be handled specially — applied to embedding inversion rather than streaming inference.

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

- All static decomposition results assume the target vector is an exact sum of token embeddings. Real-world embeddings from API endpoints involve attention, normalization, and pooling. We address these barriers (Sections 5–7) but full contextual recovery remains partial: position 0 itself is unrecovered, and longer sentences degrade with depth even after sink removal.
- The attention sink discovery (Section 7.5) and sink-skip method are demonstrated on GPT-2 Small only. The sink phenomenon is documented in larger models (Llama, Mistral), but whether the corrected method generalizes to those models — particularly to pooled production embedding endpoints — has not yet been tested.
- The layered bias structure (Section 6.5) is now a more sharply defined hypothesis with cleaner supporting evidence (the monotonic per-layer bias growth), but layer peeling itself — using the per-layer biases to walk a deep hidden state inward through the network — remains untested.
- Earlier results in Section 7.3 (the 17%/0%/83% layer-by-layer table) and the per-layer measurements in Section 6.6 reflect a sink-contaminated bias model that we have since corrected. We retain those tables to show how the corrected interpretation differs from the initial one.
- Brute-force search is computationally expensive for large vocabularies (128K+ tokens). HNSW approximate search introduces significant accuracy loss at this scale. Practical deployment requires either smaller vocabularies or improved approximate search.
- The coordinate descent result (97.2%) is primarily demonstrated on one text (Gettysburg Address) with GPT-2 Small. The technique is general but scaling behavior requires further study. The result depends on the greedy initialization and may vary by ~3% across runs.
- The quantization experiment simulates precision reduction on a single model and text. The prediction that quantization degrades contextual recovery more than static recovery is supported by our tests but has not been verified across model families.
- This work raises security concerns about embedding storage systems. We note the concern but do not provide a full threat model or mitigation framework.

---

## 12. Conclusion

For static token-sum embeddings, greedy residual decomposition often recovers constituent tokens directly, requiring no training and no learned decoder. Coordinate descent extends this to near-complete recovery even on smaller models. Together, these methods serve as interpretable probes: their transparent failure modes characterize the geometric separability of the embedding space.

The barriers to contextual decomposition are now precise and characterizable, and the dominant one was not what we originally thought it was. We initially identified a "near-constant bias component" accounting for ~99.5% of hidden state energy and treated it as a uniform per-token shift. Subsequent investigation revealed that this dominant component is overwhelmingly the **position-0 attention sink** — a single-position artifact in which the model uses the first token as a scratch space for unused attention budget. Excluding position 0 and using a sink-corrected vocabulary collapses the apparent bias from ~2700 norm to ~10–60 norm and enables 100% recovery of trailing positions on short sentences across most of GPT-2 Small's residual stream.

What remains is a smaller, well-behaved per-token bias that grows linearly with layer depth (the "real" attention bias), and a residual problem of position-dependent attention mixing at deep layers that a single-vector bias cannot fully unwind. Both have plausible attacks: layer peeling (Section 8.5) for the additive sediment, and white-box gradient refinement (Section 8.2) for the per-position mixing.

The norm-stratified residual hypothesis (Section 6.5), reformulated in terms of clean per-layer biases, generates concrete predictions that are now testable for the first time — most directly, that the per-layer bias vectors can be subtracted in order to walk a deep contextual hidden state back through the network, layer by layer, without invoking the model's weights at all. This would be a deterministic procedure for inverting depth in a contextual representation, and we propose it as the most promising next step toward a full geometric theory of transformer hidden states.

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
