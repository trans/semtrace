# Experiment 002: Mary Had a Little Lamb Decomposition

**Date**: 2026-03-30
**Benchmark**: `benchmarks/real_text.cr --builtin maryhadalittlelamb`

---

## Objective

Measure greedy residual decomposition accuracy on a short, well-known English text across four GPT-2 model sizes. Complements Experiment 001 (Gettysburg Address, 286 tokens) with a smaller input (55 tokens).

---

## Corpus

Mary Had a Little Lamb (Sarah Josepha Hale, 1830). Text used:

> Mary had a little lamb, its fleece was white as snow, and everywhere that Mary went, the lamb was sure to go. It followed her to school one day, which was against the rules. It made the children laugh and play to see a lamb at school.

---

## Corpus Statistics

| Metric | Value |
|---|---|
| Words | 47 |
| Unique words (case-insensitive, stripped punctuation) | 34 |
| BPE tokens (GPT-2 tokenizer) | 55 |
| Unique token IDs | 38 |
| Unique token ratio | 69.1% |
| Tokens appearing once | 28 |

**Most repeated tokens:**

| Token | Count |
|---|---|
| `,` | 4 |
| ` lamb` | 3 |
| ` was` | 3 |
| ` the` | 3 |
| ` to` | 3 |

---

## Method

Identical to Experiment 001:

1. Tokenize corpus with GPT-2 BPE tokenizer.
2. Sum static token embeddings (`wte.weight`) to produce target vector.
3. Greedy residual decomposition with HNSW nearest-neighbor search (cosine, f16).
4. Norm-inflection stopping condition.
5. Pure greedy, k=1, no lookahead.

**Search**: HNSW (USearch, cosine, f16, connectivity 32, expansion_add 128, expansion_search 256).

---

## Embedding Sources

Same as Experiment 001:

| Model | HuggingFace Repo | Dimensions |
|---|---|---|
| GPT-2 Small | `openai-community/gpt2` | 768 |
| GPT-2 Medium | `openai-community/gpt2-medium` | 1024 |
| GPT-2 Large | `openai-community/gpt2-large` | 1280 |
| GPT-2 XL | `openai-community/gpt2-xl` | 1600 |

---

## Results

### Accuracy

| Model | Dims | Total Recovery | Unique Recovery | Missing Unique | Extra Tokens |
|---|---|---|---|---|---|
| GPT-2 Small | 768 | 78.2% (43/55) | 76.3% (29/38) | 9 | 13 |
| GPT-2 Medium | 1024 | 72.7% (40/55) | 68.4% (26/38) | 12 | 15 |
| GPT-2 Large | 1280 | 100.0% (55/55) | 100.0% (38/38) | 0 | 2 |
| GPT-2 XL | 1600 | 100.0% (55/55) | 100.0% (38/38) | 0 | 2 |

### Missing Tokens (GPT-2 Small)

9 unique tokens not recovered:

| Token | Occurrences in Original |
|---|---|
| ` the` | 3 |
| ` It` | 2 |
| ` its` | 1 |
| ` flee` | 1 |
| ` go` | 1 |
| ` followed` | 1 |
| ` made` | 1 |
| ` play` | 1 |
| ` at` | 1 |

### Missing Tokens (GPT-2 Medium)

12 unique tokens not recovered:

| Token | Occurrences in Original |
|---|---|
| ` the` | 3 |
| ` It` | 2 |
| ` little` | 1 |
| ` its` | 1 |
| ` snow` | 1 |
| ` everywhere` | 1 |
| ` Mary` | 1 |
| ` one` | 1 |
| ` against` | 1 |
| ` made` | 1 |
| ` play` | 1 |
| ` see` | 1 |

---

## Observations

1. **GPT-2 Large and XL achieve 100% recovery** on both total and unique metrics. Every one of the 38 unique tokens and all 55 token occurrences are recovered perfectly.

2. **The 1280d threshold is confirmed**: Small (768d) and Medium (1024d) recover 76% and 68% unique respectively. Large (1280d) jumps to 100%.

3. **GPT-2 Medium anomaly persists**: 1024d (68.4% unique) performs worse than 768d (76.3% unique).

4. **Common function words are the failure mode**: ` the` and ` It` are missed by both Small and Medium. These tokens have near-duplicates in the vocabulary (e.g., `the`/`The`/` the`/` The`).

5. **Higher unique ratio (69.1%) vs Gettysburg (50.0%)**: This text has less repetition, making each token more critical. Despite this, Large and XL still achieve perfect recovery.

---

## Reproduction

```bash
crystal build benchmarks/real_text.cr -o bin/real_text

bin/real_text --builtin maryhadalittlelamb                        # GPT-2 Small
bin/real_text --data data/gpt2-medium --builtin maryhadalittlelamb
bin/real_text --data data/gpt2-large --builtin maryhadalittlelamb
bin/real_text --data data/gpt2-xl --builtin maryhadalittlelamb
```
