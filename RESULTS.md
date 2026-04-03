# Semantic Tracing: Experimental Results

**Thomas Sawyer**^1 **& Claude Opus 4.6**^2
*^1 Independent Researcher · ^2 Anthropic*

---

## Experiment Index

Each experiment is self-contained in `experiments/NNN-name/` with:
- `run.cr` — code to reproduce the experiment
- `results.md` — detailed analysis and findings
- `output.txt` — raw output from the run (when captured)

Shared corpus texts are in `experiments/texts/`.

| # | Experiment | Key Finding |
|---|---|---|
| 001 | [Gettysburg Address](experiments/001-gettysburg-address/results.md) | GPT-2 XL: 90.2% unique recovery on 286 tokens. Critical threshold at 1280d. |
| 002 | [Mary Had a Little Lamb](experiments/002-mary-had-a-little-lamb/results.md) | GPT-2 Large/XL: 100% recovery on 55 tokens. |
| 003 | [Tale of Two Cities](experiments/003-tale-of-two-cities/results.md) | GPT-2 XL: 37.9% at 638 unique tokens. Capacity limit demonstrated. |
| 004 | [Optimization & Semantic Accuracy](experiments/004-optimization-and-semantic/results.md) | No optimization beats plain greedy. Semantic matching (<0.7) recovers 100% on Gettysburg. |
| 005 | [Capacity Test](experiments/005-capacity-test/results.md) | Random token baseline: ~50 unique tokens at 50% exact. Semantic coherence provides 6x improvement over random. |
| 006 | [Llama 3.2 HNSW](experiments/006-llama-gettysburg/results.md) | Llama HNSW: 72.7% — later shown to be a search accuracy issue, not embedding quality. |
| 007 | [Metric Comparison](experiments/007-metric-comparison/results.md) | Inner product is best metric (93.7% brute-force) but HNSW degrades it. |
| 008 | [Llama Brute-Force](experiments/008-llama-brute-force/results.md) | Llama brute-force: 100% on Gettysburg, 50.2% on Tale. HNSW was the bottleneck, not the embeddings. |
| 009 | [Union of Metrics](experiments/009-union-metrics/results.md) | Union of cosine+L2+IP: 98.6% GPT-2 XL, 64% Llama on Tale. Different metrics find same tokens in different order. |
| 010 | [Contextual Embeddings](experiments/010-contextual-embeddings/results.md) | Contextual BoW decomposes at 83% (L6 cosine). Sentence embeddings never decompose — attention is non-additive. GPT-2 hidden states lack semantic discrimination. |
| 011 | [Attention Bias Subtraction](experiments/011-attention-bias/results.md) | Attention adds a constant per-token bias (99.5% of energy). Subtracting it enables 17-80% contextual decomposition. Embedding norm encodes token count N. |

---

## Summary of Best Results

### Gettysburg Address (252 words, ~143 unique tokens)

| Model | Method | Exact | Case-Insensitive |
|---|---|---|---|
| GPT-2 XL (1600d) | Single metric (IP, brute-force) | 93.7% | 95.1% |
| GPT-2 XL (1600d) | Union of 3 metrics + case-insensitive | — | 98.6% |
| Llama 3.2 3B (3072d) | Any metric (brute-force) | 100% | 100% |

### A Tale of Two Cities, Ch.1 (1001 words, ~630 unique tokens)

| Model | Method | Exact | Case-Insensitive |
|---|---|---|---|
| GPT-2 XL (1600d) | Single metric (cosine, brute-force) | 39.5% | 43.9% |
| GPT-2 XL (1600d) | Union of 3 metrics + case-insensitive | — | 54.9% |
| Llama 3.2 3B (3072d) | Single metric (L2, brute-force) | 52.3% | 53.6% |
| Llama 3.2 3B (3072d) | Union of 3 metrics + case-insensitive | — | 64.0% |

---

## Key Findings

1. **The algorithm works.** Greedy residual decomposition recovers bag-of-words content from embedding vector sums with no training, no learned decoder — just the static embedding matrix and nearest-neighbor search.

2. **Dimensionality matters, with a threshold.** GPT-2 family shows a critical jump at 1280d. Llama's 3072d achieves 100% on medium texts with exact search.

3. **Search accuracy is critical.** HNSW approximate search can lose 27+ percentage points vs brute-force (Llama: 72.7% HNSW vs 100% brute-force on Gettysburg).

4. **No single metric dominates.** IP is best at medium scale, cosine at large scale, L2 best for Llama. Union of all three outperforms any single metric.

5. **L2-normalizing the sum destroys magnitude-dependent signal.** Normalization preserves semantic similarity (connotation, paraphrase detection) but discards magnitude information needed for additive decomposition and token count estimation. Recovery techniques that depend on residual magnitude or token count cannot operate on normalized embeddings.

6. **Semantic coherence improves recovery 6x** over random tokens at the same unique count.

7. **Order is unrecoverable from static embeddings.** Vector addition is commutative — positional information requires contextual embeddings.

8. **Contextual embeddings (Ollama) are orthogonal to static space.** Direct decomposition produces noise. The embed endpoint skips the output projection and L2-normalizes.

9. **Attention adds a constant per-token bias** that accounts for 99.5% of contextual embedding energy. Subtracting it reveals the token signal, enabling 17-80% contextual decomposition from previously 0%.

10. **Embedding magnitude encodes token count.** N can be estimated from `||embedding|| / ||bias||`. L2 normalization destroys this, preventing any magnitude-dependent recovery technique from estimating N or performing residual subtraction.

11. **The model preserves token content internally.** The barrier to additive decomposition is not the model but the serving layer — normalization discards magnitude and pooling collapses per-position representations. However, semantic content survives normalization (similarity and connotation detection still work), so recovery techniques that don't depend on magnitude may still be viable on normalized embeddings.
