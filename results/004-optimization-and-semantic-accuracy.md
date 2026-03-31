# Experiment 004: Optimization Strategies and Semantic Accuracy

**Date**: 2026-03-31
**Benchmarks**: `benchmarks/optimization_test.cr`, manual analysis

---

## Objective

Two questions:
1. Do lookahead search (k=5) or L2-normalized token vectors improve decomposition accuracy?
2. When the decomposer picks the "wrong" token, how semantically close is it to the correct one?

---

## Part A: Optimization Strategies

Tested four configurations on GPT-2 XL (1600d) using HNSW search for all modes:

- **Baseline**: greedy k=1, raw embeddings
- **Lookahead**: greedy k=5, raw embeddings (pick candidate that minimizes next residual)
- **Normalized**: greedy k=1, L2-normalized token embeddings, unnormalized sum
- **Both**: greedy k=5, normalized embeddings

### Gettysburg Address (286 tokens, 143 unique)

| Config | Total Recovery | Unique Recovery |
|---|---|---|
| Baseline (k=1, raw) | **88.8%** (254/286) | **90.2%** (129/143) |
| Lookahead (k=5, raw) | 87.8% (251/286) | 88.8% (127/143) |
| Normalized (k=1, norm) | 88.1% (252/286) | 86.7% (124/143) |
| Both (k=5, norm) | 88.1% (252/286) | 88.1% (126/143) |

### Tale of Two Cities Ch.1 (1401 tokens, 638 unique)

| Config | Total Recovery | Unique Recovery |
|---|---|---|
| Baseline (k=1, raw) | **63.0%** (882/1401) | **37.9%** (242/638) |
| Lookahead (k=5, raw) | 62.7% (879/1401) | 37.5% (239/638) |
| Normalized (k=1, norm) | 64.1% (898/1401) | 39.2% (250/638) |
| Both (k=5, norm) | 63.9% (895/1401) | 37.8% (241/638) |

### Finding

**No optimization strategy meaningfully outperforms plain greedy search.** Lookahead and normalization both produce marginal changes (+/- 1-2%) that are within noise. At the Gettysburg scale, baseline is actually the best. At the larger scale, normalization provides a tiny improvement.

The accuracy ceiling is set by the geometric capacity of the space, not by the search strategy. Improving the search cannot recover information that is not distinguishable in the sum.

---

## Part B: Semantic Accuracy

When the decomposer picks the "wrong" token, how wrong is it? We measure cosine distance between each missing token and its nearest token in the recovered set.

### Reference: Cosine Distances Between Related Tokens (GPT-2 XL)

To calibrate thresholds, we measured distances between known related pairs:

| Pair | Cosine Distance |
|---|---|
| ` add` / ` adds` | 0.41 |
| `man` / `men` | 0.42 |
| ` big` / ` large` | 0.49 |
| `the` / `The` | 0.51 |
| ` life` / ` lives` | 0.53 |
| ` happy` / ` glad` | 0.53 |
| ` take` / ` took` | 0.55 |
| ` dog` / ` cat` | 0.62 |
| ` created` / ` conceived` | 0.69 |

Even obvious near-synonyms are at cosine distance 0.4-0.6 in GPT-2 XL's static embedding space. A threshold of 0.6 captures morphological variants and close synonyms. 0.7 captures broader semantic relatedness.

### Gettysburg Address: Missing Tokens and Their Nearest Recovered Match

14 unique tokens not recovered (out of 143):

| Missing | Nearest Recovered | Cos Dist | Relation |
|---|---|---|---|
| ` seven` | ` five` | 0.27 | Same category (numbers) |
| ` add` | ` adds` | 0.41 | Morphological variant |
| ` on` | ` in` | 0.41 | Close preposition |
| ` perish` | ` perished` | 0.45 | Morphological variant |
| ` But` | ` It` | 0.47 | Function word |
| ` lives` | ` life` | 0.53 | Morphological variant |
| ` they` | ` their` | 0.53 | Same referent |
| ` take` | ` took` | 0.55 | Tense variant |
| ` larger` | ` greater` | 0.51 | Synonym |
| ` increased` | ` greater` | 0.59 | Related concept |
| ` by` | ` in` | 0.59 | Preposition |
| ` might` | ` can` | 0.61 | Modal verb |
| ` which` | ` that` | 0.65 | Relative pronoun |
| ` created` | ` conceived` | 0.69 | Thematically relevant |

### Accuracy at Multiple Cosine Thresholds

| Threshold | Gettysburg (143 unique) | Tale of Two Cities (638 unique) |
|---|---|---|
| Exact | 90.2% | 37.9% |
| < 0.3 | 90.9% | 39.8% |
| < 0.4 | 90.9% | 42.5% |
| < 0.5 | 93.7% | 49.8% |
| < 0.6 | 97.9% | 57.4% |
| < 0.7 | 100.0% | 72.6% |

### Finding

**Semantic recovery is significantly higher than exact-match recovery.** At a 0.6 threshold (which captures morphological variants and close synonyms):
- Gettysburg: 97.9% (up from 90.2% exact)
- Tale of Two Cities: 57.4% (up from 37.9% exact)

At 0.7 threshold, the Gettysburg Address achieves **100% semantic recovery** — every token is either exactly recovered or has a semantically related substitute in the recovered set.

---

## Method

All tests use GPT-2 XL (1600d, 50,257 tokens). HNSW search (cosine, f16) for all configurations including the normalized variants (separate HNSW index built with pre-normalized vectors).

Semantic accuracy: for each unique token in the original that is NOT in the recovered set, compute cosine distance to every token in the recovered set and take the minimum. If this minimum is below the threshold, count it as a semantic match.

---

## Reproduction

```bash
crystal build benchmarks/optimization_test.cr -o bin/optimization_test

# Optimization comparison
bin/optimization_test --data data/gpt2-xl --file /path/to/text
```
