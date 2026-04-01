# Experiment 006: Llama 3.2 3B vs GPT-2 XL — Gettysburg Address

**Date**: 2026-04-01
**Benchmark**: Manual (Crystal eval with pre-tokenized IDs)

---

## Objective

Compare decomposition performance between Llama 3.2 3B (3072d, 128k vocab) and GPT-2 XL (1600d, 50k vocab) on the same text. Tests whether more dimensions automatically improves recovery, or whether the token-to-dimension ratio matters more.

---

## Corpus

Gettysburg Address — same text as Experiment 001.

| Metric | GPT-2 (all sizes) | Llama 3.2 3B |
|---|---|---|
| Total tokens | 286 | 286 |
| Unique tokens | 143 | 143 |

Both tokenizers produce the same token count on this text (coincidence — different tokenization, same count).

---

## Models

| Model | Dimensions | Vocab Size | Tokens/Dimension | Source |
|---|---|---|---|---|
| GPT-2 XL | 1600 | 50,257 | 31.4 | HuggingFace safetensors |
| Llama 3.2 3B Instruct | 3072 | 128,256 | 41.7 | GGUF (Q6_K dequantized) |

---

## Method

1. Tokenize corpus (GPT-2: BPE tokenizer; Llama: greedy longest-match against vocab).
2. Sum static token embeddings to produce target vector.
3. Greedy residual decomposition with HNSW nearest-neighbor search.
4. Norm-inflection stopping condition. Pure greedy, k=1.

**GPT-2 XL search**: HNSW (cosine, f16, connectivity 32, expansion_add 128, expansion_search 256).
**Llama search**: HNSW (cosine, f16, connectivity 16, expansion_add 128, expansion_search 64). Pre-built index loaded from disk (770 MB).

---

## Results

### Accuracy Comparison

| Model | Dims | Vocab | Unique Recovery | Total Recovery |
|---|---|---|---|---|
| GPT-2 XL | 1600 | 50k | **90.2%** (129/143) | **94.4%** (270/286) |
| Llama 3.2 3B | 3072 | 128k | 72.7% (104/143) | 85.3% (244/286) |

### Semantic Accuracy

| Threshold | GPT-2 XL | Llama 3.2 3B |
|---|---|---|
| Exact | 90.2% | 72.7% |
| < 0.5 | 93.7% | 72.7% |
| < 0.6 | 97.9% | 72.7% |
| < 0.7 | 100.0% | 72.7% |

GPT-2 XL's wrong picks are near-synonyms (cosine distance 0.3-0.7). Llama's wrong picks are noise (cosine distance 0.85-0.96). No semantic recovery beyond exact match for Llama.

### Llama Missing Tokens (sample)

| Missing | Nearest Recovered | Cos Dist | Quality |
|---|---|---|---|
| ` lives` | ` live` | 0.91 | Noise |
| `-field` | ` field` | 0.86 | Noise |
| ` dedicate` | ` Dedicated` | 0.83 | Noise |
| ` battle` | `Smarty` | 0.94 | Garbage |
| ` Liberty` | byte garbage | 0.94 | Garbage |
| ` government` | ` far` | 0.95 | Garbage |

Even "lives"/"live" (distance 0.91) is effectively random in Llama's space. Compare to GPT-2 XL where "lives"/"life" was 0.53 — still distinguishable.

---

## Observations

1. **More dimensions does not mean better decomposition.** Llama has 2x the dimensions of GPT-2 XL but performs significantly worse (72.7% vs 90.2%). The critical factor is the **token-to-dimension ratio**: 41.7 tokens/dim for Llama vs 31.4 for GPT-2 XL. The larger vocabulary dilutes the space.

2. **Embedding space geometry differs fundamentally.** GPT-2 XL's wrong picks are semantically close (near-synonyms, morphological variants). Llama's wrong picks are noise. This suggests Llama's embedding space is organized differently — possibly less linearly structured, or the Q6_K dequantization introduced distortion.

3. **Instruct tuning may reduce decomposability.** Llama 3.2 3B Instruct is fine-tuned for instruction following, which may reshape the embedding geometry away from the additive structure that bag-of-words decomposition depends on. A base model might perform differently.

4. **Cosine distances between related tokens are much higher in Llama.** Even obvious pairs like "lives"/"live" are at 0.91 (essentially orthogonal). This is not specific to our decomposition — it reflects the fundamental structure of Llama's static embedding space.

---

## Reproduction

```bash
# Build Llama index (one-time, ~15 min)
crystal build benchmarks/build_index.cr -o bin/build_index
bin/build_index --data data/llama-3-2-3b-instruct

# Tokenize (requires Python, greedy longest-match)
# See results/006 source for tokenization script

# Run decomposition
crystal eval '...'  # See experiment source
```
