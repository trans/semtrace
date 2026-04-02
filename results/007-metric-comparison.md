# Experiment 007: Distance Metric Comparison

**Date**: 2026-04-02
**Benchmark**: Manual (Crystal eval, brute-force and HNSW)

---

## Objective

Compare three distance metrics for greedy residual decomposition: cosine similarity, L2 (Euclidean) distance, and inner product (dot product). Tests both exact brute-force search and HNSW approximate search to separate the metric's quality from the search algorithm's accuracy.

---

## Metrics Explained

- **Cosine similarity**: `cos_sim(a, b) = (a · b) / (||a|| × ||b||)`. Measures angular alignment. Ignores magnitude entirely.
- **L2 distance**: `||a - b||²`. Measures Euclidean distance. Penalizes both directional and magnitude mismatch.
- **Inner product**: `a · b`. Measures both directional alignment AND magnitude alignment. Prefers vectors that point in the same direction AND have appropriate magnitude.

For greedy residual subtraction, the ideal metric should find the token whose subtraction best reduces the residual. Inner product naturally captures this because subtracting a vector with high inner product against the residual removes the most energy.

---

## Method

All tests use GPT-2 XL (1600d, 50,257 tokens). Standard greedy decomposition with norm-inflection stopping.

- **Brute-force**: compute the metric between the residual and every token embedding (50,257 comparisons per step). Exact — guaranteed to find the true nearest neighbor for the given metric.
- **HNSW**: approximate nearest-neighbor search via USearch library. f16 quantization, connectivity 16.

---

## Results: Gettysburg Address (286 tokens, 143 unique)

### Brute-Force (Exact Search)

| Metric | Unique Recovery | Steps |
|---|---|---|
| Cosine | 88.1% (126/143) | 284 |
| L2 | 91.6% (131/143) | 288 |
| **Inner Product** | **93.7%** (134/143) | 286 |

### HNSW (Approximate Search)

| Metric | Unique Recovery | Steps |
|---|---|---|
| **Cosine** | **90.2%** (129/143) | 288 |
| L2 | 58.0% (83/143) | — |
| Inner Product | 81.8% (117/143) | 269 |

### Cross-Reference

| Metric | Brute-Force | HNSW | HNSW vs Brute |
|---|---|---|---|
| Cosine | 88.1% | 90.2% | +2.1% (lucky approximation) |
| L2 | 91.6% | 58.0% | -33.6% (severe degradation) |
| Inner Product | 93.7% | 81.8% | -11.9% (moderate degradation) |

---

## Analysis

1. **Inner product is the best metric for this task.** With exact search, it recovers 93.7% — 5.6 percentage points more than cosine (88.1%) and 2.1 more than L2 (91.6%). This makes theoretical sense: inner product selects the token whose subtraction removes the most energy from the residual, which is exactly what greedy decomposition needs.

2. **HNSW degrades non-cosine metrics significantly.** HNSW's graph navigation uses distance for neighbor pruning, which implicitly assumes the triangle inequality holds. Cosine (angular distance) satisfies this approximately. L2 satisfies it exactly but performs poorly in HNSW for high-dimensional data. Inner product violates it entirely, causing HNSW to miss the true nearest neighbors.

3. **Cosine HNSW slightly outperforms cosine brute-force** (90.2% vs 88.1%). This is counterintuitive — approximate search should be equal or worse than exact. The explanation: HNSW's approximation error occasionally leads to different greedy paths that happen to cascade better. This is a coincidence of this specific input, not a general property.

4. **L2 HNSW is catastrophically bad** (58.0%). L2 in high dimensions suffers from the "curse of dimensionality" — distances concentrate, making HNSW's pruning unreliable at 1600d.

---

## Implications

The optimal decomposition pipeline would use:
- **Inner product metric** for search (best accuracy)
- **Exact search** (brute-force or IP-optimized index like FAISS IVF)

For GPT-2's 50,257 vocabulary at 1600d, brute-force inner product costs ~80M multiply-adds per step. At 286 steps for the Gettysburg Address, that's ~23 billion operations — feasible on modern hardware but slower than HNSW.

For practical use, **cosine HNSW remains the best tradeoff** — 90.2% accuracy with fast O(log N) search per step.

---

## Reproduction

```bash
# Build inner product index
crystal build benchmarks/build_index.cr -o bin/build_index
bin/build_index --data data/gpt2-xl --ip

# Brute-force comparison requires crystal eval (see experiment source)
```
