# Experiment 003: A Tale of Two Cities, Chapter 1

**Date**: 2026-03-31
**Benchmark**: `benchmarks/real_text.cr --file <path>`

---

## Objective

Measure greedy residual decomposition accuracy on a longer prose passage (~1,000 words) to test how performance degrades with increasing unique token count. Complements Experiment 001 (Gettysburg Address, 143 unique tokens) and Experiment 002 (Mary Had a Little Lamb, 38 unique tokens).

---

## Corpus

Chapter 1 ("The Period") of *A Tale of Two Cities* by Charles Dickens (1859). Retrieved from Project Gutenberg (https://www.gutenberg.org/files/98/98-0.txt).

The text begins "It was the best of times, it was the worst of times..." and ends "...along the roads that lay before them."

Full text stored at: `benchmarks/texts/tale-ch1.txt` (not included inline due to length — 1,001 words).

---

## Corpus Statistics

| Metric | Value |
|---|---|
| Words | 1,001 |
| BPE tokens (GPT-2 tokenizer) | 1,401 |
| Unique token IDs | 638 |
| Unique token ratio | 45.5% |
| Tokens appearing once | 494 |

This corpus has 4.5x more unique tokens than the Gettysburg Address (638 vs 143), making it a significantly harder decomposition target.

---

## Method

Identical to Experiments 001 and 002:

1. Tokenize corpus with GPT-2 BPE tokenizer.
2. Sum static token embeddings (`wte.weight`) to produce target vector.
3. Greedy residual decomposition with HNSW nearest-neighbor search (cosine, f16).
4. Norm-inflection stopping condition.
5. Pure greedy, k=1, no lookahead.

**Search**: HNSW (USearch, cosine, f16, connectivity 32, expansion_add 128, expansion_search 256).

---

## Embedding Sources

Same as Experiments 001 and 002:

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
| GPT-2 Small | 768 | 43.4% (608/1401) | 13.2% (84/638) | 554 | 534 |
| GPT-2 Medium | 1024 | 46.1% (646/1401) | 13.9% (89/638) | 549 | 497 |
| GPT-2 Large | 1280 | 66.7% (935/1401) | 33.9% (216/638) | 422 | 390 |
| GPT-2 XL | 1600 | 69.2% (970/1401) | 37.9% (242/638) | 396 | 323 |

---

## Cross-Experiment Comparison

Unique token recovery across all three experiments:

| Corpus | Words | Tokens | Unique | Small 768d | Medium 1024d | Large 1280d | XL 1600d |
|---|---|---|---|---|---|---|---|
| Mary Had a Little Lamb | 47 | 55 | 38 | 76.3% | 68.4% | 100% | 100% |
| Gettysburg Address | 252 | 286 | 143 | 42.0% | 40.6% | 89.5% | 90.2% |
| Tale of Two Cities Ch.1 | 1,001 | 1,401 | 638 | 13.2% | 13.9% | 33.9% | 37.9% |

The degradation with increasing unique token count is steep. XL drops from 100% at 38 unique tokens to 38% at 638. This suggests a fundamental capacity limit in bag-of-words additive encoding, not merely an algorithmic limitation.

---

## Observations

1. **Capacity limit**: At 638 unique tokens, even the largest model (1600d) recovers only 38%. The embedding space cannot faithfully represent this many distinct tokens as a single additive sum.

2. **Dimensionality still matters**: The 1280d threshold holds — Large (34%) and XL (38%) significantly outperform Small (13%) and Medium (14%). But even the larger models are well below practical utility at this unique token count.

3. **GPT-2 Medium anomaly weakens**: At high unique counts, Medium (14%) slightly outperforms Small (13%), unlike the lower-count experiments where Medium was consistently worse. The anomaly may be specific to smaller decomposition tasks.

4. **Rich literary vocabulary is harder**: Dickens uses many rare and multi-syllable words ("incredulity", "superlative", "revelations") that produce BPE subword fragments, which are inherently harder to recover.

5. **Overshoot is significant**: All models produce hundreds of extra tokens before the inflection stop triggers, indicating the greedy search is struggling to find meaningful signal in the residual.

---

## Theoretical Note

The capacity limit has a mathematical basis. An unordered bag of N items drawn from a vocabulary of V has far less information content than an ordered sequence. For a simple analogy: 1600 ordered binary digits encode 2^1600 patterns, but 1600 unordered binary digits encode only 1601 patterns (the count of 1s). While embedding arithmetic operates in continuous space (not binary), the principle holds: **additive composition (which is commutative) fundamentally limits the number of distinguishable token combinations that can be represented in a given dimensionality.**

---

## Reproduction

```bash
# Download the text
curl -s "https://www.gutenberg.org/files/98/98-0.txt" | \
  sed -n '/^It was the best of times/,/^CHAPTER II/p' | head -n -3 > tale-ch1.txt

crystal build benchmarks/real_text.cr -o bin/real_text

bin/real_text --file tale-ch1.txt                        # GPT-2 Small
bin/real_text --data data/gpt2-medium --file tale-ch1.txt
bin/real_text --data data/gpt2-large --file tale-ch1.txt
bin/real_text --data data/gpt2-xl --file tale-ch1.txt
```
