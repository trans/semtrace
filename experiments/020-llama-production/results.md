# Experiment 020: Llama 3.2 3B Production Embedding Inversion

**Date**: 2026-04-10 – 2026-04-11
**Code**: `experiments/contextual/llama_probe.py`, `experiments/contextual/llama_centered_pipeline.py`, `experiments/contextual/llama_hillclimb.py`, `experiments/contextual/llama_generate_and_verify.py`, `experiments/contextual/llama_centered_pipeline.py`

---

## Objective

Test whether the embedding inversion techniques developed on GPT-2 Small generalize to a real production embedding endpoint: Llama 3.2 3B Instruct via Ollama's `/api/embed`. This is the fully black-box scenario — no model weights, no hidden states, only the embed and generate APIs.

---

## Llama Embedding Properties (vs GPT-2)

| Property | GPT-2 Small (pooled L12) | Llama 3.2 3B (Ollama embed) |
|---|---|---|
| Dimensionality | 768 | 3072 |
| Normalization | NOT L2-normalized | L2-normalized (norm = 1.0) |
| Cosine discrimination | Saturated at 0.99+ (centroid-dominated) | **Discriminative** (0.48–0.80 range) |
| Semantic clustering | Good (after centering) | Good (directly) |
| Additivity (sentence ≈ sum of words) | Partial at L1 (cos 0.89), fails at L12 | Fails (cos 0.53, barely above random) |
| Leave-one-out per-word signal | Weak (saturated cosine) | **Strong** (10× dynamic range, delta 0.05–0.51) |
| Greedy decomposition | 89% at L1, 31% at L12 | 1/5 overlap (fails — not a sum) |

**Key finding: cosine IS the useful metric on Llama** (no centering needed for sentence-level comparison). But for word-level probing, centering still helps (improves individual-word identification from 0/5 to 3/5 correct in top-15).

---

## Math of Mean-Pool + L2-Normalize

```
output = normalize(mean(h_1, h_2, ..., h_N))
       = normalize((1/N) × Σ h_i)
       = (Σ h_i) / ||Σ h_i||
```

**The N cancels.** Mean-pool + normalize = normalize(sum). The output is purely the DIRECTION of the sum of hidden states. One scalar is lost (the magnitude ||Σ h_i||, which correlates with N). The direction (3071 degrees of freedom) is fully preserved.

**N estimation from the embedding**: tested via repeated tokens, centroid distance, RoPE dimension analysis, and embedding value statistics. **None work reliably.** Content variation drowns out the length signal. Centroid distance gives a rough "short vs long" estimate (3 bins) but can't distinguish N=5 from N=7. Testing showed knowing N exactly helps only ~5-10% — the bottleneck is content words, not length.

---

## Sentence-Level Verification

The embedding uniquely identifies sentences by cosine:

| Comparison to "the cat sat on the mat" | Cosine |
|---|---|
| Exact match | **1.0000** |
| Tense change ("sits" vs "sat") | 0.977 |
| Article change ("a" vs "the") | 0.945 |
| Verb swap ("lay" vs "sat") | 0.913 |
| Noun swap ("dog" vs "cat") | 0.912 |
| Reversed meaning ("mat sat on cat") | 0.662 |
| Unrelated ("quantum physics") | 0.517 |

**Word order IS distinguishable** (reversed = 0.66 vs original = 1.00). The "commutativity barrier" from static-sum decomposition does not apply to production embeddings.

---

## Content Word Extraction: Hierarchical N-gram Probing

The main methodological contribution. Individual word cosine to target is noisy (0/5 correct in top-15 raw, 3/5 with centering). But probing with n-grams — embedding word COMBINATIONS and comparing to target — amplifies the signal at each level:

### Results on "the cat sat on the mat"

| Level | Method | Top result | True words | Cosine |
|---|---|---|---|---|
| 1 | Individual centered cosine | mat, cat, sat in top-15 | 3/5 | ~0.15 |
| 2 | Embed all 105 pairs from top-15 | **(cat, sat) is #1** | 2/2 | 0.456 |
| 3 | Embed all 220 triples from top-pair-words | **(mat, cat, sat) is #1** | 3/3 | 0.438 |

### Results on "she walked slowly to the river"

| Level | Top result | True words | Cosine |
|---|---|---|---|
| 1 | river, walked, slowly in top-15 | 3/6 | ~0.15 |
| 2 | **(walked, slowly) is #1** | 2/2 | 0.448 |
| 3 | **(walked, slowly, river) is #1** | 3/3 | **0.695** |

### Results on "Mary had a little lamb"

| Level | Top result | True words | Cosine |
|---|---|---|---|
| 1 | lamb, Mary in top-15 | 2/5 | ~0.10 |
| 2 | (lamb, does) is #1 | 1/2 | 0.319 |
| 3 | (queen, walked, lamb) is #1 | 1/3 | 0.481 |

**The hierarchical probing correctly identifies the content words for 2/3 test sentences.** It fails on "Mary had a little lamb" because "had", "a", "little" are generic words that don't score well individually — only "lamb" and "Mary" are distinctive.

**Compute cost:** ~316 embed API calls total (~15 seconds on Ollama, < $0.01 on paid APIs).

---

## Full Black-Box Attack Pipeline

```
Step 1: Build vocab embeddings          ~89 embed calls (one-time)
Step 2: Compute centroid, center        (local numpy, no API)
Step 3: Level 1 - individual probing    ~89 centered-cosine lookups (precomputed)
Step 4: Level 2 - pair probing          ~105 embed calls
Step 5: Level 3 - triple probing        ~120 embed calls
Step 6: Feed top triple to LLM         1 generate call (Mistral)
Step 7: Embed LLM's candidate          1 embed call
Step 8: Verify by cosine               cos ≈ 1.0 if correct
Step 9: Hill-climb if needed           ~50-200 embed calls
                                       ─────────────────
                                       ~400 embed calls total
```

### End-to-End Results

| Sentence | Content words found | LLM reconstruction | Final cos |
|---|---|---|---|
| the cat sat on the mat | mat, cat, sat | "The cat sat on the mat." | **1.0000 EXACT** |
| she walked slowly to the river | walked, slowly, river | "she walked slowly as children played near the river" | 0.955 (5/6 words) |
| Mary had a little lamb | lamb, Mary | "Mary's mother gets allergy to pork" | 0.48 (failed) |

**1/3 exact recovery, 1/3 near-recovery, 1/3 failure.** Success depends on whether the sentence has enough distinctive content words for the hierarchical probing to find.

---

## Negative Results / Approaches That Failed

- **Greedy residual decomposition**: 1/5 overlap. The embedding isn't a sum, so iterative subtraction fails immediately.
- **Subset selection** (find word group whose centroid matches target): correct words outscore random but wrong word combinations can outscore correct ones. The embedding ≠ centroid of constituent word embeddings.
- **Generate-and-verify with Llama's own generate API**: Llama 3.2 3B Instruct generates assistant-style responses ("I'm happy to help...") instead of content sentences.
- **N estimation**: not recoverable from the normalized embedding by any method tested (repeated tokens, centroid distance, RoPE fingerprint, embedding statistics).

---

## Conclusions

1. **Production embeddings are a SEARCH problem, not a decomposition problem.** The embedding is not a sum and can't be decomposed. But it CAN be verified — any candidate sentence can be checked in one API call.

2. **Hierarchical n-gram probing extracts content words** with ~316 API calls. The signal STRENGTHENS at each level — the correct n-gram consistently scores highest when enough distinctive words exist.

3. **The vulnerability of a sentence depends on its vocabulary distinctiveness.** Sentences with specific content words (names, technical terms, unusual verbs) are more vulnerable than sentences with generic vocabulary.

4. **Fully black-box exact recovery is achievable** on some sentences (demonstrated on "the cat sat on the mat" with cos=1.0000). The pipeline requires only the embed API + any LLM for candidate generation.

5. **The fundamental bottleneck is not N, not normalization, not the metric — it's extracting enough content words from the embedding.** Centered probing + hierarchical n-grams gets 2-3 content words. Sentences needing 4+ content words for reconstruction remain out of reach.

---

## Reproduction

```bash
cd experiments/contextual
python3 llama_probe.py                 # basic probing (5 tests)
python3 llama_hillclimb.py            # thematic probing + hill-climbing
python3 llama_centered_pipeline.py    # centered probing + LLM + hill-climbing
python3 llama_generate_and_verify.py  # generate API attempt (failed)
```

Requires Ollama with `llama3.2:3b` and `mistral:7b`:
```bash
ollama pull llama3.2:3b
ollama pull mistral:7b
```
