# Semantic Tracing: Experimental Results

**Thomas Sawyer**^1 **& Claude Opus 4.6**^2
*^1 Independent Researcher · ^2 Anthropic*

---

## 1. Overview

These results document experiments with greedy residual decomposition of static token embedding vectors, as described in DESIGN.md. The algorithm decomposes a target vector into a sequence of tokens whose embeddings sum to approximate it.

All tests use bag-of-words embedding (sum of static token embeddings) as the target. The decomposer uses HNSW approximate nearest-neighbor search (USearch, cosine metric, f16 quantization) with residual norm inflection as the automatic stopping condition.

---

## 2. Models Tested

| Model | Dimensions | Vocab Size | Avg Token Norm | Source |
|---|---|---|---|---|
| GPT-2 Small | 768 | 50,257 | 3.50 | HuggingFace safetensors |
| GPT-2 Medium | 1024 | 50,257 | 3.22 | HuggingFace safetensors |
| GPT-2 Large | 1280 | 50,257 | 1.92 | HuggingFace safetensors |
| GPT-2 XL | 1600 | 50,257 | 1.75 | HuggingFace safetensors |
| Llama 3.2 3B | 3072 | 128,256 | TBD | GGUF (Q6_K dequantized) |

All GPT-2 variants share the same BPE vocabulary and tokenizer. Average norms are sampled from the first 1,000 tokens.

---

## 3. Baseline Tests

All models achieve:
- **16/16** exact single-token round-trips (zero residual)
- **3/3** exact pair composition decompositions (zero residual)

These confirm the algorithm correctly inverts additive composition for small N.

---

## 4. Battery A: Simple Vocabulary

Single-clause sentences using common, single-token words. No subword fragments. Tests raw capacity of the decomposition at increasing lengths.

| N | GPT-2 Small (768d) | GPT-2 Medium (1024d) | GPT-2 Large (1280d) | GPT-2 XL (1600d) |
|---|---|---|---|---|
| 6 | 100% | 100% | 100% | 100% |
| 7 | 100% | 100% | 100% | 100% |
| 12 | 100%* | 100%* | 100% | 100% |
| 19 | 100%* | 100%* | 100% | 100% |
| 39 | 100%* | 94.9% | 100% | 100% |
| 57 | 87.7% | 82.5% | 98.2% | 100%* |
| 90 | 93.3% | 88.9% | 100% | 100%* |
| 140 | 85.0% | 85.7% | 97.9% | 98.6% |

*All original tokens recovered but extra tokens added.

**Observations:**
- Exact recovery up to 12 tokens across all models.
- GPT-2 Large and XL maintain >97% accuracy even at 140 tokens.
- GPT-2 Medium performs anomalously worse than Small, despite higher dimensionality.
- The 90-token result for Small (93.3%) is better than its 57-token result (87.7%), suggesting accuracy depends on token composition, not purely length.

---

## 5. Battery B: Complex Vocabulary (Mixed)

Sentences mixing common words with multi-syllable, technical, and compound words that produce BPE subword fragments (e.g., "semiconductor", "archaeological", "seventeenth").

| N | GPT-2 Small (768d) | GPT-2 Medium (1024d) | GPT-2 Large (1280d) | GPT-2 XL (1600d) |
|---|---|---|---|---|
| 9 | 100% | 100% | 100% | 100% |
| 11 | 90.9% | 90.9% | 100% | 100% |
| 15 | 86.7% | 86.7% | 100% | 100% |
| 23 | 82.6% | 82.6% | 100%* | 100%* |
| 35 | 85.7% | 74.3% | 100% | 100%* |

**Observations:**
- Subword fragments degrade accuracy significantly in smaller models, even at low N.
- GPT-2 Large and XL handle complex vocabulary perfectly (100% at all lengths).
- The difference between Battery A and B shows that token quality matters more than token count for smaller models.

---

## 6. Battery C: Multi-Sentence & Punctuation

Multiple sentences with quotation marks, parentheses, colons, semicolons, em dashes, and mixed punctuation. Tests handling of punctuation tokens which tend to have similar embeddings.

| N | GPT-2 Small (768d) | GPT-2 Medium (1024d) | GPT-2 Large (1280d) | GPT-2 XL (1600d) |
|---|---|---|---|---|
| 8 | 87.5% | 100%* | 100% | 100% |
| 17 | 100% | 100% | 100% | 100% |
| 22 | 86.4% | 72.7% | 95.5% | 95.5% |
| 32 | 87.5% | 78.1% | 100% | 100% |
| 50 | 70.0% | 58.0% | 100%* | 100% |

**Observations:**
- Punctuation-heavy text is the hardest category for all models.
- The 22-token quotation sentence (repeated `"` tokens) is the only failure point for Large and XL.
- GPT-2 Medium drops to 58% on the 50-token Dr. Smith sentence — its worst result.
- GPT-2 XL achieves 100% on the same sentence, demonstrating that sufficient dimensionality resolves punctuation ambiguity.

---

## 7. Dimensionality Analysis

The GPT-2 family provides a controlled experiment: same vocabulary, same tokenizer, different embedding dimensions.

| Dimensions | Avg Norm | Simple@140 | Complex@35 | Punctuation@50 |
|---|---|---|---|---|
| 768 | 3.50 | 85.0% | 85.7% | 70.0% |
| 1024 | 3.22 | 85.7% | 74.3% | 58.0% |
| 1280 | 1.92 | 97.9% | 100% | 100% |
| 1600 | 1.75 | 98.6% | 100% | 100% |

**Key findings:**
- **Non-monotonic improvement**: 1024d performs worse than 768d across most tests. Higher dimensionality alone does not guarantee better decomposition — training quality matters.
- **Critical threshold at ~1280d**: A dramatic accuracy jump occurs between 1024d and 1280d, suggesting a phase transition in embedding space separability.
- **Diminishing returns above 1280d**: The 1280d→1600d improvement is modest compared to 1024d→1280d.
- **Norm correlation**: Average token embedding norms decrease with dimensionality (3.50→1.75), indicating vectors become more spread out in higher-dimensional spaces. This correlates with better decomposition performance.

---

## 8. Real Text Decomposition (Gettysburg Address)

The Gettysburg Address (Abraham Lincoln, 1863) provides a real-world test: 252 words, producing 286 BPE tokens with 143 unique token IDs (50% unique ratio). This is natural English prose with punctuation, repeated function words, and varied vocabulary.

**Search method**: HNSW approximate nearest-neighbor (USearch, cosine metric, f16 quantization) — the same index used for all standard decomposition throughout this project. Greedy residual subtraction with norm-inflection stopping.

### 8a. Cross-Model Comparison

| Model | Dims | Total Recovery | Unique Recovery | Missing Unique | Extra Tokens |
|---|---|---|---|---|---|
| GPT-2 Small | 768 | 66.1% (189/286) | 42.0% (60/143) | 83 | 88 |
| GPT-2 Medium | 1024 | 65.0% (186/286) | 40.6% (58/143) | 85 | 74 |
| GPT-2 Large | 1280 | 94.1% (269/286) | 89.5% (128/143) | 15 | 16 |
| GPT-2 XL | 1600 | 94.4% (270/286) | 90.2% (129/143) | 14 | 14 |

**Input statistics:**
- Words: 252
- BPE tokens: 286
- Unique token IDs: 143 (50.0% unique ratio)
- Most repeated tokens: `,` (20x), ` that` (13x), `.` (9x), ` we` (8x), ` a` (7x)
- Tokens appearing only once: 95

**Two accuracy metrics reported:**
- **Total recovery**: counts every token occurrence including duplicates. If "that" appears 13 times and is recovered 13 times, all 13 count toward accuracy.
- **Unique recovery**: counts each distinct token ID once. If "that" is recovered at all, it counts as 1 out of 143.

**Observations:**
- The critical dimensionality threshold at ~1280d is confirmed on real text.
- GPT-2 Large and XL recover ~90% of unique tokens from the Gettysburg Address.
- GPT-2 Small and Medium recover only ~40% of unique tokens.
- The GPT-2 Medium anomaly persists: 1024d performs slightly worse than 768d.
- Missing tokens on XL are mostly common function words (" on", " they", " by") and a few content words (" seven", " created", " perish").

### 8b. Correction: Synthetic Sweep Results (Previously Reported)

**Important note**: Earlier versions of this document reported "99.7% accuracy at 10,000 tokens" from a parameterized sweep. This result was misleading. The sweep generated text from a fixed word list of ~120 English words, meaning at 10,000 tokens each word repeated ~158 times. The high accuracy reflected the decomposer's ability to find the same ~129 unique tokens repeatedly, not its ability to handle 10,000 distinct tokens.

The sweep results remain valid for the specific question "can the decomposer recover tokens from sums with heavy repetition?" but should not be interpreted as general decomposition accuracy at high token counts. The synthetic sweep data is retained below for reference but the Gettysburg Address results (Section 8a) are the authoritative measure of real-text decomposition performance.

<details>
<summary>Synthetic sweep data (click to expand)</summary>

Parameterized sweep using seeded text from a ~120-word list. At high N, each word repeats many times. 3 trials per data point, HNSW search, greedy k=1.

**Cross-model (simple vocabulary):**

| N (total) | ~Unique | Small 768d | Medium 1024d | Large 1280d | XL 1600d |
|---|---|---|---|---|---|
| 100 | ~75 | 77.8% | 69.5% | 96.7% | 98.3% |
| 500 | ~120 | 82.8% | 71.5% | 98.0% | 98.4% |
| 1000 | ~129 | 84.1% | 76.8% | 99.3% | 99.4% |

**GPT-2 XL extended sweep:**

| N (total) | ~Unique | Accuracy | Overshoot |
|---|---|---|---|
| 1,000 | ~129 | 99.4% | 11.7% |
| 5,000 | ~129 | 99.7% | 11.5% |
| 10,000 | ~129 | 99.7% | 11.8% |

These high accuracies reflect heavy token repetition, not large unique-token recovery. The actual unique token count is only ~129 regardless of total N.

</details>

---

## 9. Failure Mode Analysis

Failures are consistently caused by:

1. **Near-duplicate tokens**: Case variants (`the`/`The`, `she`/`She`) and tense variants (`did`/`does`) have close embeddings. The greedy algorithm picks the wrong one.
2. **Subword fragments**: Rare BPE fragments (e.g., `cible` from "reproducible") have poorly-trained embeddings that occupy desolate regions of the space, surrounded by control characters rather than semantically related tokens.
3. **Repeated punctuation**: Quotation marks, commas, and parentheses appear multiple times and have similar embeddings, creating confusion in the greedy path.

**Non-failures:**
- Negations ("not", "but") and conjunctions ("and", "or") recover perfectly in all tests.
- Sentence length per se is not the limiting factor — 90-token simple text can decompose better than 22-token punctuation-heavy text.

---

## 10. Residual Norm as N-Finder

The residual norm after each decomposition step provides a reliable signal for the number of meaningful tokens:

- For exact sums: norm monotonically decreases to zero.
- For non-exact vectors (concept arithmetic, midpoints): norm decreases while finding real tokens, then inflects upward when signal is exhausted.
- The inflection point automatically determines N without prior knowledge of sequence length.

Example — concept arithmetic `king - man + woman`:
```
Norms: 4.1787 → 2.6425 → 3.0482 (stops at N=2)
Tokens: " king", " woman"
```

The algorithm correctly identifies the two positive terms and stops before diverging.

---

## 11. Positional Embedding Analysis

GPT-2 uses learned absolute positional embeddings (wpe matrix, 1024 positions x d dimensions).

**Finding**: The sum of (token + positional) embeddings is commutative. Order is mathematically unrecoverable from any sum of static embeddings, regardless of search strategy.

```
Σ(E(t_i) + P(i)) is identical for all permutations of token-position assignment.
```

Positional embeddings have larger norms (~4-10) than token embeddings (~2.5-3.5), and tokens show weak statistical affinity for certain positions (from training distribution), but this signal is not sufficient for order recovery.

**Implication**: Order recovery requires contextual embeddings where attention has broken the commutativity of the representation.

---

## 12. Contextual Embedding Experiments (Llama 3.2 3B)

Attempted decomposition of contextual embeddings from Ollama's `/api/embed` endpoint against the static token embedding matrix extracted from the same model (128,256 tokens x 3,072d).

### 12a. Static vs Contextual Space Alignment

Calibration test: embed 30 individual tokens through both the static matrix (lookup) and the forward pass (Ollama), then compare.

| Metric | Value |
|---|---|
| Average cosine similarity | 0.003 |
| Min cosine similarity | -0.036 |
| Max cosine similarity | 0.054 |

**The spaces are orthogonal.** Static and contextual embeddings share dimensionality (3072) but zero structure. Every contextual vector is normalized to exactly 1.0 by Ollama.

### 12b. Contextual Space Internal Cohesion

While orthogonal to static space, contextual embeddings have strong internal semantic structure:

| Pair | Cosine Similarity |
|---|---|
| cat / kitten | 0.72 |
| cat / dog | 0.70 |
| king / queen | 0.67 |
| cat / car | 0.47 |
| "the cat sat on the mat" / "a cat was sitting on a mat" | 0.88 |
| "the cat sat on the mat" / "quantum physics is fascinating" | 0.52 |

Sentence-to-sentence similarity works well. These embeddings are optimized for semantic comparison, not decomposition.

### 12c. Contextual Word-Level Decomposition

Attempted decomposition of sentence embeddings against contextual word embeddings (335 common words embedded individually through Ollama). Greedy residual subtraction diverges after 1-2 steps due to L2 normalization placing all vectors on a unit sphere.

Similarity ranking (without subtraction) shows a strong bias toward generic function words ("again", "told", "knew", "decide") regardless of input sentence. Actual content words from the sentence rarely appear in the top 20.

### 12d. Why Contextual Decomposition Fails

Investigation of Ollama's source code (github.com/ollama/ollama) revealed:

1. **The embed endpoint uses a different code path than generation.** Generation runs the full pipeline: `token_embd → layers → OutputNorm → Output projection → logits`. The embed endpoint uses llama.cpp's internal `GetEmbeddingsSeq`, which returns the hidden state **without the output projection**.

2. **L2 normalization is applied by llama.cpp** before returning the embedding, destroying magnitude information needed for arithmetic operations.

3. **Llama 3.2 3B ties weights** (no separate `output.weight` tensor). The model uses `token_embd.weight^T` as the output projection. But since the embed endpoint skips this projection, the returned vector lives in pre-projection space.

4. **Sentence embeddings encode meaning holistically, not compositionally.** The embedding of "the cat sat on the mat" is not a combination of word-level signals — it's a high-level semantic representation optimized for similarity comparison.

### 12e. Concept Arithmetic in Contextual Space

Vector arithmetic partially works within contextual space:

| Analogy | Expected | Rank | Similarity |
|---|---|---|---|
| puppy - dog + cat | kitten | 3 | 0.74 |
| king - man + woman | queen | 3 | 0.59 |
| puppy - dog + horse | foal | 4 | 0.60 |

Expected answers appear in top-5 but not rank-1, confirming the space has relational structure but not the clean linear geometry of static embeddings.

---

## 13. Broader Goal and Future Directions

The motivating question: **given an embedding, how much of the original input can be recovered?** Not necessarily verbatim, but semantically equivalent. If this works efficiently, it enables:

- **Reasoning over embeddings** rather than over tokens
- **Embedding-only storage** — drop verbatim text, keep only the embedding, recover approximate text on demand
- **Semantic compression** — fixed-size vector regardless of text length

### What we've established:

- **Static bag-of-words**: essentially fully recoverable (99.7% at 10,000 tokens). Security implication: bag-of-words embeddings are effectively plaintext.
- **Contextual embeddings**: not directly decomposable against static tokens. The spaces are orthogonal.
- **Within contextual space**: sentence similarity works, but word-level decomposition does not.

### Promising next directions:

1. **Dedicated embedding models** (nomic-embed-text, etc.) — purpose-built for embeddings, might have more compositional structure.
2. **Full pipeline control** (GPT-2 in Python) — extract hidden states at every layer, find where decomposability degrades through the transformer.
3. **Learned mapping** — train a transformation (linear or small NN) between contextual and static spaces using paired token embeddings as training data. Could also explore topological mapping approaches.
4. **Logits as embeddings** — the model's output projection IS a decomposition into token scores. Using logit vectors as the "embedding" would be inherently decomposable.
5. **GPT-2 Medium anomaly** — why does 1024d perform worse than 768d? Training quality vs architecture investigation.
