# Experiment 008: Llama 3.2 3B — Brute-Force vs HNSW

**Date**: 2026-04-02
**Benchmark**: `benchmarks/metric_test_llama.cr` (compiled with `--release`)

---

## Objective

Re-evaluate Llama 3.2 3B decomposition using exact brute-force search instead of the HNSW approximate index. Experiment 006 showed Llama (3072d, 128k vocab) performing worse than GPT-2 XL (1600d, 50k vocab) at 72.7% vs 90.2% on the Gettysburg Address. This experiment tests whether the poor result was caused by the embedding space or the approximate search.

---

## Background

In Experiment 006, we built an HNSW index for Llama with connectivity 16 (reduced from the usual 32 to save memory for the 128k-token vocabulary). HNSW approximate search found 72.7% of unique tokens on the Gettysburg Address, leading us to conclude that Llama's embeddings were poorly structured for decomposition.

This experiment repeats the test with brute-force search: computing the exact cosine similarity (or inner product) between the residual and every token in the 128k vocabulary at each step. This is slow (~150s for Gettysburg, ~670s for Tale of Two Cities) but eliminates any approximation error.

---

## Method

1. Load Llama 3.2 3B static token embeddings (128,256 tokens × 3,072 dimensions, dequantized from Q6_K).
2. Tokenize text using greedy longest-match against the Llama vocabulary.
3. Sum token embeddings to produce target vector.
4. Greedy residual decomposition with **brute-force exact search** (linear scan of all 128,256 tokens per step).
5. Norm-inflection stopping condition.
6. Test both cosine similarity and inner product metrics.

**No HNSW index is used.** Each step computes the metric against every token embedding.

Compiled with `crystal build --release` for LLVM optimization of the inner loop.

---

## Corpus

Same texts as Experiments 001 and 003:

| Text | Words | Tokens (Llama) | Unique Tokens |
|---|---|---|---|
| Gettysburg Address | 252 | 286 | 143 |
| A Tale of Two Cities, Ch.1 | 1,001 | 1,355 | 625 |

Note: Llama tokenization produces slightly different token counts than GPT-2 (286 vs 286 for Gettysburg by coincidence; 1,355 vs 1,401 for Tale of Two Cities).

---

## Results

### Gettysburg Address (286 tokens, 143 unique)

| Search | Metric | Exact | Semantic <0.7 | Steps | Time |
|---|---|---|---|---|---|
| HNSW (Exp 006) | Cosine | 72.7% (104/143) | 72.7% | 253 | ~2s |
| **Brute-force** | **Cosine** | **100.0%** (143/143) | **100.0%** | **286** | **151s** |
| **Brute-force** | **IP** | **100.0%** (143/143) | **100.0%** | **286** | **82s** |

### A Tale of Two Cities, Ch.1 (1,355 tokens, 625 unique)

| Search | Metric | Exact | Semantic <0.7 | Steps | Time |
|---|---|---|---|---|---|
| HNSW (Exp 003*) | Cosine | 37.9% (242/638)† | 72.6% | 1,338 | ~8s |
| **Brute-force** | **Cosine** | **50.2%** (314/625) | **50.2%** | **1,278** | **673s** |
| **Brute-force** | **IP** | **48.2%** (301/625) | **48.2%** | **1,231** | **356s** |

*Exp 003 used GPT-2 XL, not Llama. †GPT-2 tokenizer produces 638 unique vs Llama's 625.

---

## Cross-Model Comparison (Brute-Force, Cosine)

| Model | Dims | Vocab | Gettysburg (143 unique) | Tale of Two Cities (~630 unique) |
|---|---|---|---|---|
| GPT-2 XL | 1,600 | 50,257 | 88.1% | 39.5% |
| **Llama 3.2 3B** | **3,072** | **128,256** | **100.0%** | **50.2%** |

---

## Key Findings

### 1. Llama's embeddings are excellent — HNSW was the bottleneck

The 72.7% result from Experiment 006 was entirely caused by HNSW approximation error, not by the embedding space. With exact search, Llama achieves **100% perfect recovery** on the Gettysburg Address — every one of 143 unique tokens recovered in exactly 286 steps with zero overshoot.

The HNSW index we built (connectivity 16) was too sparse for 128k tokens at 3072 dimensions. The graph didn't have enough edges for reliable navigation in this high-dimensional space.

### 2. More dimensions genuinely helps

With exact search eliminating the approximation confound:
- Llama (3072d): 100% on Gettysburg, 50.2% on Tale of Two Cities
- GPT-2 XL (1600d): 88.1% on Gettysburg, 39.5% on Tale of Two Cities

Higher dimensionality provides better token separation, confirming the theoretical expectation. The larger vocabulary (128k vs 50k) doesn't negate the dimensional advantage.

### 3. Semantic matching adds nothing on Llama

For the Tale of Two Cities, Llama's exact, semantic <0.5, and semantic <0.7 scores are all identical (50.2%). When Llama misses a token, the nearest recovered token is far away (cosine distance >0.7). Unlike GPT-2 XL where wrong picks are often near-synonyms (cosine distance 0.3-0.6), Llama's wrong picks are effectively random in cosine space.

This may be an artifact of Llama's much larger vocabulary — with 128k tokens, near-synonyms might occupy different regions than in GPT-2's denser 50k space.

### 4. Inner product doesn't help Llama

Unlike GPT-2 XL where IP outperformed cosine (93.7% vs 88.1%), on Llama IP is slightly worse (48.2% vs 50.2% on Tale of Two Cities). On Gettysburg both achieve 100%. The embedding magnitude patterns differ between models.

### 5. The accuracy vs speed tradeoff is stark

| Method | Gettysburg Accuracy | Time per text |
|---|---|---|
| HNSW (connectivity 16) | 72.7% | ~2s |
| Brute-force cosine | 100.0% | 151s |
| Brute-force IP | 100.0% | 82s |

A 75x slowdown buys 27 percentage points on Gettysburg. For research this is fine; for production, a denser HNSW graph (connectivity 32 or 64) or an alternative like FAISS IVF could close the gap.

---

## Implications

The earlier conclusion that "Llama's embedding space is poorly structured for decomposition" (Experiment 006) was **wrong**. The embedding space is well-structured — better than GPT-2 XL's. The failure was in the approximate search algorithm, not the embeddings.

This means:
- **Dimensionality analysis** from the GPT-2 family is confirmed: more dimensions → better decomposition.
- **Approximate search quality is a critical variable** that must be controlled in all experiments.
- Future work should either use exact search or carefully validate HNSW accuracy against brute-force baselines.

---

## Reproduction

```bash
# Tokenize for Llama (one-time, requires Python)
python3 -c "
import json
vocab = json.load(open('data/llama-3-2-3b-instruct/vocab.json'))
token_to_id = {v: int(k) for k, v in vocab.items()}
tokens_by_len = sorted(token_to_id.keys(), key=len, reverse=True)
def tokenize(text):
    ids, pos = [], 0
    while pos < len(text):
        for tok in tokens_by_len:
            if text[pos:pos+len(tok)] == tok:
                ids.append(token_to_id[tok]); pos += len(tok); break
        else: pos += 1
    return ids
# Gettysburg
ids = tokenize(open('/tmp/gettysburg.txt').read().strip())
json.dump(ids, open('data/llama-3-2-3b-instruct/gettysburg_ids.json','w'))
# Tale of Two Cities
ids = tokenize(open('benchmarks/texts/tale-ch1.txt').read().strip())
json.dump(ids, open('data/llama-3-2-3b-instruct/tale_ch1_ids.json','w'))
"

# Compile with optimizations
crystal build benchmarks/metric_test_llama.cr -o bin/metric_test_llama --release

# Run
bin/metric_test_llama --data data/llama-3-2-3b-instruct \
  --ids data/llama-3-2-3b-instruct/gettysburg_ids.json

bin/metric_test_llama --data data/llama-3-2-3b-instruct \
  --ids data/llama-3-2-3b-instruct/tale_ch1_ids.json
```
