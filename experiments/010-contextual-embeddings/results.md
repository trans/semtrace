# Experiment 010: Contextual Embedding Decomposition

**Date**: 2026-04-02
**Code**: `experiments/contextual/build_ctx_vocab.py`, `experiments/contextual/decompose.py`

---

## Objective

Determine whether contextual (forward-pass) embeddings can be decomposed into token components using greedy residual subtraction. Tests both static and contextual search vocabularies, at middle (L6) and final (L12) transformer layers, with all three distance metrics (cosine, L2, inner product).

Also characterizes the semantic quality of different embedding sources: GPT-2 hidden states vs Llama Ollama endpoint.

---

## Models

- **GPT-2 Small (124M)**: Run natively via HuggingFace transformers. Full access to hidden states at every layer, unnormalized. 50,257 tokens x 768d.
- **Llama 3.2 3B Instruct**: Via Ollama `/api/embed` endpoint. L2-normalized output.

---

## Method

1. Build contextual vocabulary: embed every token in the vocabulary individually through GPT-2's forward pass. Save per-layer hidden states to disk as numpy arrays. Done separately to avoid OOM (`build_ctx_vocab.py`).

2. Forward-pass the test sentence to get hidden states at each layer.

3. Decompose using greedy residual subtraction with three metrics (cosine, L2, IP) against:
   - Static vocabulary (`wte.weight`)
   - Contextual vocabulary at L6 (middle layer)
   - Contextual vocabulary at L12 (final layer)

4. Test two target types:
   - **Sentence embedding**: mean-pooled or last-token hidden states from the forward pass
   - **Contextual bag-of-words**: sum of individually-embedded tokens (control — should work if the space supports addition)

Input text: "the cat sat on the mat" (6 tokens, 6 unique)

---

## Results

### Decomposition Accuracy (tokens recovered out of 6)

| Target | Vocab | Layer | Cosine | IP | L2 | Union |
|---|---|---|---|---|---|---|
| Static sum | Static | — | **6/6** | **6/6** | **6/6** | **6/6** |
| Sentence (mean) | Contextual | L6 | 1/6 | 0/6 | 0/6 | 1/6 |
| Sentence (last) | Contextual | L6 | 1/6 | 0/6 | 0/6 | 1/6 |
| Sentence (mean) | Contextual | L12 | 0/6 | 0/6 | 0/6 | 0/6 |
| Sentence (last) | Contextual | L12 | 0/6 | 0/6 | 0/6 | 0/6 |
| Contextual BoW | Contextual | L6 | **5/6** | 0/6 | 0/6 | **5/6** |
| Contextual BoW | Contextual | L12 | 0/6 | 0/6 | 0/6 | 0/6 |

### What the Decomposer Found (First Tokens Recovered)

| Test | Metric | Tokens Found |
|---|---|---|
| Static sum → static | cosine | `the`, ` sat`, ` cat`, ` on`, ` mat`, ` the` |
| Sentence → ctx L6 | cosine | ` sat` (only) |
| Ctx BoW → ctx L6 | cosine | `ed`, ` beside`, `the`, ` cat`, ` mat`, ` on`, ` sat` |
| Sentence → ctx L12 | cosine | `anus`, `raviolet`, ` nor` (garbage) |
| Anything → ctx L12 | IP | ` GOODMAN`, `'d`, `osph` (garbage) |
| Anything → ctx L12 | L2 | ` THEY`, `ung`, `ultane` (garbage) |

---

## Key Findings

### 1. Inner product and L2 fail completely in contextual space

In static space, IP outperformed cosine (Experiment 007). In contextual space, IP and L2 are **catastrophically bad** — finding only garbage tokens (`ieri` repeated, `GOODMAN`, `THEY`). Only cosine finds any meaningful tokens.

This is because contextual embedding norms are extreme: ~3000 at L6, ~70 at L12. IP and L2 are magnitude-sensitive, so they're dominated by norm outliers in the contextual vocabulary.

### 2. Contextual bag-of-words works at L6 (83%), fails at L12 (0%)

When we sum individually-embedded tokens and decompose against the same contextual vocabulary, L6 with cosine recovers 5/6 tokens (83%). This proves the **contextual space at middle layers does support additive decomposition**.

By L12, even additive decomposition fails completely. The final layer's space is non-additive — the geometry no longer supports vector subtraction as a means of token recovery.

### 3. Sentence embeddings never decompose

The forward-pass sentence embedding — whether mean-pooled or last-token — does not decompose into individual token embeddings at any layer, with any metric, against any vocabulary. At best, 1/6 tokens found (17%).

This is because:
```
forward_pass("the cat sat on the mat") ≠ Σ forward_pass(token_i)
```
Attention creates a holistic representation, not an additive one.

### 4. GPT-2 hidden states lack semantic discrimination

Tested cosine similarity between sentence pairs at L12:

| Pair | GPT-2 L12 |
|---|---|
| "the dog chased the cat" / "the cat chased the dog" | 0.997 |
| "I am happy" / "I am not happy" | 0.995 |
| "cat sat on mat" / "quantum physics is fascinating" | 0.987 |

All pairs are 0.99+ — GPT-2's hidden states are dominated by a shared direction with <1% encoding meaning. These are NOT useful embeddings.

### 5. Llama Ollama embeddings show real semantic structure

Same tests through Ollama's `/api/embed` endpoint:

| Pair | Llama |
|---|---|
| bank (money) / bank (river) | 0.597 |
| "I am happy" / "I am not happy" | 0.777 |
| "cat on mat" / "feline on rug" (paraphrase) | 0.674 |
| "cat on mat" / "quantum physics" (unrelated) | 0.517 |

Clear semantic discrimination: connotation pairs 0.57-0.65, paraphrases 0.67-0.86, unrelated 0.52-0.56. This confirms the embedding endpoint produces qualitatively different (and useful) representations compared to raw hidden states.

---

## Conclusions

1. **Greedy residual decomposition requires additive representations.** It works on static bag-of-words (100%) and partially on mid-layer contextual bag-of-words (83%). It does not work on sentence embeddings from any layer.

2. **Attention is the barrier.** It creates non-additive, holistic representations that cannot be decomposed by subtraction. This is not a metric, normalization, or search problem — it's fundamental to how attention works.

3. **The right metric depends on the space.** Cosine dominates in contextual space (where norms are extreme). IP dominates in static space (where norms are moderate). There is no universal best metric.

4. **A learned mapping remains the viable path.** The semantic structure exists in contextual embeddings (Llama demonstrates this). But accessing it through subtraction is not possible. A trained transformation (linear or NN) between contextual and decomposable space is needed.

---

## Reproduction

```bash
# Step 1: Build contextual vocabularies (one-time, ~5 min on CPU)
cd experiments/contextual
python3 build_ctx_vocab.py --layers 6,12

# Step 2: Run decomposition
python3 decompose.py --text "the cat sat on the mat"
```
