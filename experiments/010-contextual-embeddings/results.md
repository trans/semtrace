# Experiment 010: Contextual Embedding Decomposition

**Date**: 2026-04-02
**Code**: `experiments/contextual/build_ctx_vocab.py`, `experiments/contextual/decompose.py`

---

## REVISED INTERPRETATION (2026-04-07)

The numbers in this experiment are floor results, not ceilings. After investigating the per-position structure of GPT-2 hidden states (see `experiments/contextual/attention_trace.py`, `skip_sink.py`, `layer_sweep.py`), we found that the difficulty of contextual decomposition reported here was overwhelmingly an artifact of the **position-0 attention sink**, not a fundamental property of contextual representations.

Specifically:

- **The "contextual vocabulary" used here was sink-corrupted.** `build_ctx_vocab.py` embeds each token by passing it as a single-token sequence — placing every entry at position 0, where the model treats it as the attention sink and inflates its hidden state to ~3000 norm. Every entry in `ctx_vocab_L6.npy` and `ctx_vocab_L11.npy` is therefore a sink-pumped attention dump rather than a representative contextual embedding. The corrected vocabulary uses `[<|endoftext|>, token]` pairs and reads the hidden state at position 1.
- **The "sentence sum" target was sink-dominated.** Position 0 of any forward-passed sentence contributes ~89% of the sum's magnitude (~2700 vs ~70 per other position at L11). Decomposing the sum decomposes the sink, not the sentence content.
- **The "Contextual BoW" 83% control worked for the wrong reason.** It summed sink-corrupted vocab entries and decomposed against the same sink-corrupted vocab — both sides shared the same sink, so the comparison was symmetric and the additive structure was found *inside the sink*, not in the residual stream.

Corrected results on the same sentence ("the cat sat on the mat"), using sink-skip + N-corrected debias:

| Layer | Trailing recovery (5/5 = perfect) |
|---|---|
| L1 | 5/5 |
| L3 | 5/5 |
| L5 | 5/5 |
| L7 | 5/5 |
| L9 | 5/5 |
| L11 | 2/5 |

With `<|endoftext|>` prepended to the input (moving the original first token off the sink position), 6/6 is achievable at L1, L3, L5.

The original conclusions in this file ("attention is the barrier", "a learned mapping remains the viable path") should be read as conditional on the sink contamination. Attention mixing *is* a residual barrier at deep layers, but it is much smaller than this experiment suggested. A learned mapping is no longer required for partial recovery — sink-skip + N-debias is sufficient through L9 of GPT-2 Small.

The original content is preserved below for record.

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

**Note**: In HuggingFace GPT-2, `hidden_states[12]` is post-final-LayerNorm (`ln_f`), unlike `hidden_states[0-11]` which are pre-normalization residual stream states. L11 is the correct "final layer" for comparison; L12 is a different representational object.

| Target | Vocab | Layer | Cosine | IP | L2 | Union |
|---|---|---|---|---|---|---|
| Static sum | Static | — | **6/6** | **6/6** | **6/6** | **6/6** |
| Sentence (mean) | Contextual | L6 | 1/6 | 0/6 | 0/6 | 1/6 |
| Sentence (last) | Contextual | L6 | 1/6 | 0/6 | 0/6 | 1/6 |
| Sentence (mean) | Contextual | L11 (pre-ln_f) | 0/6 | 0/6 | 0/6 | 0/6 |
| Sentence (last) | Contextual | L11 (pre-ln_f) | 0/6 | 0/6 | 0/6 | 0/6 |
| Contextual BoW | Contextual | L6 | **5/6** | 0/6 | 0/6 | **5/6** |
| Contextual BoW | Contextual | L11 (pre-ln_f) | **5/6** | 0/6 | 0/6 | **5/6** |
| Contextual BoW | Contextual | L12 (post-ln_f) | 0/6 | 0/6 | 0/6 | 0/6 |

### What the Decomposer Found (First Tokens Recovered)

| Test | Metric | Tokens Found |
|---|---|---|
| Static sum → static | cosine | `the`, ` sat`, ` cat`, ` on`, ` mat`, ` the` |
| Sentence → ctx L6 | cosine | ` sat` (only) |
| Ctx BoW → ctx L6 | cosine | `ed`, ` beside`, `the`, ` cat`, ` mat`, ` on`, ` sat` |
| Ctx BoW → ctx L11 | cosine | `iling`, ` beside`, `the`, ` cat`, ` mat`, ` on`, ` sat` |
| Anything → ctx L12 | any | garbage (post-ln_f representation) |

---

## Key Findings

### 1. Inner product and L2 fail completely in contextual space

In static space, IP outperformed cosine (Experiment 007). In contextual space, IP and L2 are **catastrophically bad** — finding only garbage tokens (`ieri` repeated, `GOODMAN`, `THEY`). Only cosine finds any meaningful tokens.

This is because contextual embedding norms are extreme: ~3000 at L6, ~70 at L12. IP and L2 are magnitude-sensitive, so they're dominated by norm outliers in the contextual vocabulary.

### 2. Contextual bag-of-words: 83% at L6 AND L11

When we sum individually-embedded tokens and decompose against the same contextual vocabulary, cosine recovers 5/6 tokens (83%) at both L6 and L11. The additive structure persists through the entire pre-ln_f residual stream.

At L12 (post-ln_f), decomposition fails entirely (0/6). The final LayerNorm masks the additive structure by transforming the representation into a prediction-oriented basis. The structure is not destroyed — it is masked by the normalization.

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
