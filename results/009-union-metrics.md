# Experiment 009: Union of Multiple Distance Metrics

**Date**: 2026-04-02
**Benchmark**: `benchmarks/union_test.cr` (compiled with `--release`)

---

## Objective

Test whether running greedy decomposition three times with different distance metrics (cosine, L2, inner product) and unioning the recovered token sets improves accuracy beyond any single metric. Also test case-insensitive matching to capture case-variant tokens (e.g., `" here"` / `" Here"`).

---

## Motivation

Experiment 007 showed that different metrics produce different results (88.1% cosine vs 93.7% IP on Gettysburg). Step-by-step trace analysis revealed the metrics **disagree on ordering, not identity** — they find mostly the same tokens but pull them out in different sequences. This means their error patterns are complementary: tokens one metric misses due to greedy path divergence might be found by another metric on a different path.

---

## Method

1. Run greedy residual decomposition three times independently, each with a different metric:
   - **Cosine similarity**: direction-only, ignores magnitude
   - **L2 distance**: penalizes both direction and magnitude mismatch
   - **Inner product**: rewards both directional alignment and magnitude
2. Union all three recovered token ID sets.
3. Match against original tokens using:
   - **Exact match**: token ID equality
   - **Case-insensitive**: lowercase-stripped string equality (catches `here`/`Here`, `can`/`Can`)

All searches use **brute-force exact search** (linear scan). Compiled with `--release`.

---

## Results

### Gettysburg Address

| Method | GPT-2 XL (1600d, 143 unique) | Llama 3.2 3B (3072d, 143 unique) |
|---|---|---|
| Cosine (exact) | 88.1% | 100.0% |
| L2 (exact) | 91.6% | 100.0% |
| IP (exact) | 93.7% | 100.0% |
| **Union (exact)** | **97.2%** | **100.0%** |
| **Union (case-insensitive)** | **98.6%** | **100.0%** |
| Missing after union+CI | 2 | 0 |
| Extraneous unique IDs | 24 | 0 |

GPT-2 XL misses only 2 tokens after union + case-insensitive: `" lives"` and `" which"`. Llama achieves perfect recovery on all three metrics independently.

### A Tale of Two Cities, Chapter 1

| Method | GPT-2 XL (1600d, 638 unique) | Llama 3.2 3B (3072d, 625 unique) |
|---|---|---|
| Cosine (exact) | 39.5% | 50.2% |
| L2 (exact) | 37.6% | 52.3% |
| IP (exact) | 37.8% | 48.2% |
| **Union (exact)** | **49.1%** | **62.1%** |
| **Union (case-insensitive)** | **54.9%** | **64.0%** |
| Missing after union+CI | 288 | 225 |
| Extraneous unique IDs | 536 | 660 |

---

## Union Improvement

| Text | Best Single Metric | Union + Case-Insensitive | Improvement |
|---|---|---|---|
| Gettysburg (GPT-2 XL) | 93.7% | 98.6% | +4.9 pts |
| Gettysburg (Llama) | 100% | 100% | — |
| Tale (GPT-2 XL) | 39.5% | 54.9% | +15.4 pts |
| Tale (Llama) | 52.3% | 64.0% | +11.7 pts |

The union approach provides the largest improvement on the hardest task (Tale of Two Cities), where greedy path divergence between metrics is greatest.

---

## Analysis of Missing Tokens (Llama, Tale of Two Cities)

225 unique tokens remain unrecovered after union + case-insensitive matching:

- **~87 subword fragments** (≤4 chars): `wis`, `uli`, `des`, `ual`, `ter`, `ored` — BPE pieces from words like "wisdom", "incredulity", "despair". These are meaningless in isolation and occupy poorly-trained regions of the embedding space.
- **~135 full words**: `worst`, `epoch`, `spring`, `insisted`, `Westminster`, `appearance`, `revelations` — real content words the decomposer genuinely missed.

The subword fragments are inherently hard to recover — they have no independent semantic meaning and their embeddings may be poorly separated. The full-word misses represent the true capacity limit of the embedding space at this unique token count (625).

---

## Why Union Works

Step-by-step trace analysis (Gettysburg, GPT-2 XL) revealed:

1. **Metrics disagree 84% of the time** on which token to pick next.
2. **Disagreements are about ordering, not identity.** All three metrics find valid tokens from the original text — they just prefer different high-frequency function words at each step.
3. **Cosine is conservative**: picks directionally-aligned tokens (`,`, `" to"`, `" in"`) first, finds content words late.
4. **IP is aggressive**: picks magnitude-appropriate tokens, finds content words earlier but occasionally makes wrong picks.
5. **Their error patterns are complementary**: tokens missed on one greedy path are often found on another.

The union captures the strengths of each metric's path without inheriting any single path's cascading errors.

---

## Extraneous Tokens

The union produces more unique token IDs than the original (e.g., 1048 vs 625 for Llama Tale). Many "extraneous" tokens are:
- Case variants of correct tokens (matched by case-insensitive)
- Morphological variants (`" life"` for `" lives"`, `" gave"` for `" give"`)
- Near-synonyms that a semantic threshold would match

The extraneous count is the cost of running three independent decompositions. For bag-of-words recovery, this overshoot is acceptable — the goal is to find as many original tokens as possible, and false positives can be filtered.

---

## Reproduction

```bash
crystal build benchmarks/union_test.cr -o bin/union_test --release

# GPT-2 XL
bin/union_test --data data/gpt2-xl --file benchmarks/texts/tale-ch1.txt
bin/union_test --data data/gpt2-xl --file /tmp/gettysburg.txt

# Llama 3.2 3B (requires pre-tokenized IDs)
bin/union_test --data data/llama-3-2-3b-instruct --ids data/llama-3-2-3b-instruct/gettysburg_ids.json
bin/union_test --data data/llama-3-2-3b-instruct --ids data/llama-3-2-3b-instruct/tale_ch1_ids.json
```
