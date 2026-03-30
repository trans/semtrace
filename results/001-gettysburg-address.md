# Experiment 001: Gettysburg Address Decomposition

**Date**: 2026-03-30
**Benchmark**: `benchmarks/real_text.cr --builtin gettysburg`

---

## Objective

Measure greedy residual decomposition accuracy on real English prose across four GPT-2 model sizes (768d, 1024d, 1280d, 1600d). All models share the same 50,257-token BPE vocabulary and tokenizer.

---

## Corpus

The Gettysburg Address (Abraham Lincoln, November 19, 1863). Full text:

> Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this. But, in a larger sense, we can not dedicate — we can not consecrate — we can not hallow — this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here. It is rather for us to be here dedicated to the great task remaining before us — that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion — that we here highly resolve that these dead shall not have died in vain — that this nation, under God, shall have a new birth of freedom — and that government of the people, by the people, for the people, shall not perish from the earth.

---

## Corpus Statistics

| Metric | Value |
|---|---|
| Words | 252 |
| Unique words (case-insensitive, stripped punctuation) | 133 |
| BPE tokens (GPT-2 tokenizer) | 286 |
| Unique token IDs | 143 |
| Unique token ratio | 50.0% |
| Tokens appearing once | 95 |

**Most repeated tokens:**

| Token | Count |
|---|---|
| `,` | 20 |
| ` that` | 13 |
| `.` | 9 |
| ` we` | 8 |
| ` a` | 7 |

---

## Method

1. Tokenize the corpus using GPT-2's BPE tokenizer (shared across all four models).
2. Look up the static token embedding for each token ID from the model's `wte` (word token embedding) matrix. These are the raw learned embeddings from each model's training — the first layer of the transformer.
3. Sum all 286 token embeddings to produce the target vector: `V = Σ E(t_i)` for all tokens in the corpus.
4. Apply greedy residual decomposition:
   - Find the token whose embedding is nearest to the current residual (by cosine similarity).
   - Subtract that token's embedding from the residual.
   - Repeat until the residual norm increases (inflection stopping condition).
5. Compare the recovered token multiset to the original.

**Search method**: HNSW approximate nearest-neighbor index (USearch library). Cosine distance metric. f16 quantization for the index. Connectivity 32, expansion_add 128, expansion_search 256.

**Stopping condition**: Halt when the residual L2 norm increases compared to the previous step (norm inflection).

**No lookahead or optimization**: Pure greedy, k=1 at each step.

---

## Embedding Sources

All embeddings are static token embeddings (`wte.weight`) extracted from HuggingFace safetensors files:

| Model | HuggingFace Repo | Dimensions | File Size |
|---|---|---|---|
| GPT-2 Small | `openai-community/gpt2` | 768 | 147 MB |
| GPT-2 Medium | `openai-community/gpt2-medium` | 1024 | 196 MB |
| GPT-2 Large | `openai-community/gpt2-large` | 1280 | 245 MB |
| GPT-2 XL | `openai-community/gpt2-xl` | 1600 | 307 MB |

---

## Results

### Accuracy

| Model | Dims | Total Recovery | Unique Recovery | Missing Unique | Extra Tokens |
|---|---|---|---|---|---|
| GPT-2 Small | 768 | 66.1% (189/286) | 42.0% (60/143) | 83 | 88 |
| GPT-2 Medium | 1024 | 65.0% (186/286) | 40.6% (58/143) | 85 | 74 |
| GPT-2 Large | 1280 | 94.1% (269/286) | 89.5% (128/143) | 15 | 16 |
| GPT-2 XL | 1600 | 94.4% (270/286) | 90.2% (129/143) | 14 | 14 |

**Metric definitions:**
- **Total recovery**: Each token occurrence counts independently. If ` that` appears 13 times in the original and is recovered 13 times, all 13 count.
- **Unique recovery**: Each distinct token ID counts once. If ` that` is recovered at all, it counts as 1 out of 143 unique IDs.
- **Missing unique**: Number of distinct token IDs present in the original but absent from the recovered set.
- **Extra tokens**: Number of tokens in the recovered set that do not appear in the original (overshoot before the inflection stop triggers).

### Missing Tokens (GPT-2 XL)

14 unique tokens not recovered:

| Token | Occurrences in Original |
|---|---|
| ` on` | 2 |
| ` they` | 2 |
| ` seven` | 1 |
| ` created` | 1 |
| ` lives` | 1 |
| ` might` | 1 |
| ` But` | 1 |
| ` larger` | 1 |
| ` add` | 1 |
| ` take` | 1 |
| ` increased` | 1 |
| ` which` | 1 |
| ` by` | 1 |
| ` perish` | 1 |

Missing tokens are a mix of common function words (` on`, ` they`, ` by`, ` which`) and less-common content words (` perish`, ` increased`, ` larger`). This is consistent with the known failure mode: near-duplicate token embeddings cause the greedy search to select a similar but incorrect token.

### Embedding Space Statistics

| Model | Dims | Min Norm | Max Norm | Avg Norm | Median Norm |
|---|---|---|---|---|---|
| GPT-2 Small | 768 | 2.45 | 5.34 | 3.50 | 3.51 |
| GPT-2 Medium | 1024 | 2.01 | 4.65 | 3.22 | 3.28 |
| GPT-2 Large | 1280 | 0.88 | 4.87 | 1.92 | 1.90 |
| GPT-2 XL | 1600 | 0.82 | 3.71 | 1.75 | 1.79 |

Norm statistics from the first 1,000 tokens of each model.

---

## Observations

1. **Critical dimensionality threshold at ~1280d**: Accuracy jumps from ~40% unique recovery (768d, 1024d) to ~90% (1280d, 1600d). This is the most prominent finding.

2. **GPT-2 Medium anomaly**: 1024d performs slightly worse than 768d (40.6% vs 42.0% unique), despite higher dimensionality. This anomaly has been consistent across all experiments.

3. **Diminishing returns above 1280d**: The improvement from 1280d to 1600d is small (89.5% → 90.2% unique), suggesting the embedding space is near-saturated for this vocabulary size and task.

4. **Average embedding norm decreases with dimensionality**: 3.50 → 3.22 → 1.92 → 1.75. Lower norms in higher dimensions indicate vectors are more spread out, reducing near-duplicate confusion.

5. **Overshoot is minimal for larger models**: GPT-2 XL produces only 14 extra tokens (4.9% overshoot), compared to 88 for Small (30.8%).

---

## Reproduction

```bash
# Prepare model data (one-time, downloads from HuggingFace)
bin/semtrace prepare gpt2
bin/semtrace prepare gpt2-medium
bin/semtrace prepare gpt2-large
bin/semtrace prepare gpt2-xl

# Build the benchmark
crystal build benchmarks/real_text.cr -o bin/real_text

# Run
bin/real_text --builtin gettysburg                    # GPT-2 Small (default)
bin/real_text --data data/gpt2-medium --builtin gettysburg
bin/real_text --data data/gpt2-large --builtin gettysburg
bin/real_text --data data/gpt2-xl --builtin gettysburg
```
