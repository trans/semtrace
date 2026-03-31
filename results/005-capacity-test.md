# Experiment 005: Embedding Space Capacity

**Date**: 2026-03-31
**Benchmark**: `benchmarks/capacity_test.cr`

---

## Objective

Measure the raw geometric capacity of bag-of-words additive encoding: how many unique tokens can be summed into a single vector and recovered by greedy decomposition? This serves as a **control experiment** — using random tokens with no semantic coherence, it isolates the geometric limits of the space from the effects of natural language structure.

Comparing these results to real-text experiments (001–003) reveals whether semantic coherence in the input improves recoverability.

---

## Method

1. Select a pool of well-trained tokens from the vocabulary. Filter to the middle 80% of the norm distribution (excluding outlier tokens with unusually large or small norms). Skip the first 256 tokens (control characters, byte tokens). This yields ~40,000 tokens from GPT-2 XL's 50,257 vocabulary.
2. For each trial: randomly sample N unique token IDs from this pool (no repeats), sum their embeddings, run greedy decomposition, measure recovery.
3. Report three metrics:
   - **Exact**: fraction of original token IDs found in the recovered set
   - **Semantic <0.5**: exact matches + missing tokens whose nearest recovered token is within cosine distance 0.5
   - **Semantic <0.7**: exact matches + missing tokens whose nearest recovered token is within cosine distance 0.7
4. 5 trials per N value, averaged.

**Search**: HNSW (USearch, cosine, f16). Greedy k=1, norm-inflection stopping.

**Key difference from real-text experiments**: No natural language structure. No repetition. No semantic coherence. Purely random token combinations. This is the hardest possible input for the decomposer.

---

## Results: GPT-2 XL (1600d)

Token pool: 40,000 tokens (middle 80% of norms, avg norm: 1.917)

| N Unique | Exact | Semantic <0.5 | Semantic <0.7 | Recovered |
|---|---|---|---|---|
| 10 | 100.0% | 100.0% | 100.0% | 14 |
| 25 | 70.4% | 72.0% | 80.8% | 25 |
| 50 | 46.0% | 48.4% | 59.6% | 44 |
| 75 | 31.7% | 34.1% | 50.4% | 65 |
| 100 | 19.0% | 21.2% | 45.0% | 87 |
| 150 | 9.5% | 12.5% | 37.3% | 128 |
| 200 | 6.6% | 8.9% | 33.1% | 170 |
| 250 | 4.6% | 7.3% | 34.2% | 212 |
| 300 | 3.1% | 5.6% | 34.4% | 253 |
| 400 | 3.1% | 5.8% | 34.1% | 337 |
| 500 | 2.4% | 5.2% | 34.0% | 420 |

### Capacity Thresholds

| Metric | 50% Capacity | Asymptote |
|---|---|---|
| Exact match | ~50 tokens | ~3% at 300+ |
| Semantic <0.5 | ~50 tokens | ~5% at 300+ |
| Semantic <0.7 | ~75 tokens | ~34% at 200+ |

---

## Comparison: Random Tokens vs Real Text

The critical finding emerges when comparing these results to real-text experiments:

| Input | Unique Tokens | Exact Recovery | Semantic <0.7 |
|---|---|---|---|
| Random tokens (this experiment) | 50 | 46.0% | 59.6% |
| Random tokens | 100 | 19.0% | 45.0% |
| Random tokens | 143 | ~14%* | ~40%* |
| **Gettysburg Address** (Exp 001) | **143** | **90.2%** | **100%** |
| Random tokens | 500 | 2.4% | 34.0% |
| Random tokens | 638 | ~2%* | ~34%* |
| **Tale of Two Cities** (Exp 003) | **638** | **37.9%** | **72.6%** |

*Interpolated from surrounding data points.

**At 143 unique tokens**: random achieves ~14% exact, real text achieves 90%. A 6x improvement.
**At 638 unique tokens**: random achieves ~2% exact, real text achieves 38%. A 19x improvement.

### Interpretation

**Semantic coherence dramatically improves recoverability.** When the input tokens form meaningful text, the decomposer recovers far more than when tokens are random. This demonstrates that:

1. **The embedding space has structure that reflects semantic relationships.** Tokens from coherent text occupy a more structured region of the space than random combinations, making them more distinguishable from a sum.

2. **Order matters at the input.** Even though the sum is commutative (order-destroying), the fact that the input tokens were drawn from a coherent sequence — rather than randomly — affects how separable they are in the sum. The training process created embeddings where semantically related tokens cooperate rather than interfere.

3. **Real-world performance exceeds theoretical capacity.** The random-token capacity (~50 exact at 1600d) significantly underestimates what the decomposer achieves on real text (~143 exact). Any capacity analysis must account for the non-random structure of natural language.

---

## Reproduction

```bash
crystal build benchmarks/capacity_test.cr -o bin/capacity_test

bin/capacity_test --data data/gpt2-xl --trials 5 --max 500
```
