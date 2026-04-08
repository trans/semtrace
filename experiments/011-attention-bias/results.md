# Experiment 011: Attention Bias Subtraction for Contextual Decomposition

**Date**: 2026-04-03
**Code**: `experiments/contextual/linear_map.py`, inline Python tests

---

## REVISED INTERPRETATION (2026-04-07)

This experiment's central observation — that there is a near-constant "bias" component in contextual hidden states accounting for ~99.5% of the energy — is real but **mischaracterized**. The dominant component is not a per-token bias added uniformly to every position. It is the **position-0 attention sink**: a single-position artifact in which the model uses the first token's hidden state as a scratch space for unused attention budget, growing it to ~2700 norm while every other position remains at ~50–70 norm.

What this changes about each finding:

- **"The bias is identical across sentences (cosine 0.9999+)."** True, but for the wrong reason. We were measuring how consistently the model creates a sink at position 0 (very consistently — it is an architectural property, not a sentence property). The sink's magnitude (~2630–2674 in the table) is essentially constant because it is set by the model's learned attention pattern, not by the sentence content.
- **"99.5% of the centered hidden state energy is bias."** This is the sink:non-sink ratio, not a bias:content ratio. With sink removal, the actual per-token bias is ~16 norm at L1 growing linearly to ~60 at L11, and the per-position content is ~50–70 norm. The ratio is much closer to 1:1 than 200:1 for non-sink positions.
- **"Estimated N (norm-ratio) outperforms exact N."** This is **inverted** under the corrected method. Norm-ratio happened to undershoot in a way that was less bad than oversubtracting the sink-contaminated bias. With sink-skip and a clean per-token bias, the *exact* N is the right multiplier, and `N × bias` subtraction works much better than `(||sum||/||bias||) × bias`.
- **The recovery numbers (2/6, 3/6, 4/5, 1/5, 3/6, 5/9).** These are **floor results**. They reflect partial recovery against a sink-contaminated bias model and a sink-corrupted vocabulary. With sink-skip + N-debias, the same sentences should recover at substantially higher rates on the trailing positions. We re-run them below.
- **"Linear mapping 89.9% identity preservation."** Stands. That sub-experiment used per-token contextual embeddings against per-token static embeddings, both with position-0 contamination — but symmetrically, so the linear regression found the right rotation. The result is real and unaffected.
- **"Energy budget: 200:1 bias:signal."** Sink:non-sink, not bias:signal. The corrected ratio is much closer to 1:1.

A re-run with sink-skip + N-debias on the same six sentences is appended at the end of this file.

The original content is preserved below for record.

---

## Objective

Discover and exploit the structure of the attention contribution to contextual embeddings, enabling greedy decomposition of sentence embeddings that previously produced 0% recovery.

---

## Discovery: The Attention Bias is a Constant

For any sentence, the contextual hidden state sum at layer 6 can be decomposed as:

```
contextual_sum = token_content + N × per_token_bias
```

Where:
- `contextual_sum` = sum of per-position hidden states (centered by vocab mean)
- `token_content` = sum of individually-embedded token vectors (centered)
- `per_token_bias` = a constant vector added by attention, per token

The per-token bias was computed by subtracting known token content from the contextual sum. Testing across 8 different reference sentences:

| Reference Sentence | Bias Norm |
|---|---|
| the quick brown fox jumps over the lazy dog | 2631.4 |
| she went to the store to buy some food | 2632.4 |
| he ran down the long road to the old house | 2674.0 |
| they played in the park with the children | 2610.2 |
| we had dinner at a small restaurant | 2672.2 |
| the sun was warm and the sky was blue | 2629.1 |
| it was a dark and stormy night | 2597.7 |
| once upon a time there was a king | 2599.4 |

**Cosine similarity between any two biases: 0.9999+**

The bias is essentially identical regardless of sentence content. It represents a fixed artifact of GPT-2's L6 computation — the "what attention adds to every token" signal.

---

## Key Insight: Estimating N from the Embedding

Since the bias dominates (99.5% of the contextual sum's energy), the number of tokens N can be estimated from the norm:

```
N ≈ ||contextual_sum|| / ||per_token_bias||
```

This estimated N works **better** than the true N because it accounts for sentence-specific variation in bias magnitude.

---

## Method

1. **Precompute**: Average per-token bias from 8 reference sentences at L6.
2. **Center**: Subtract vocabulary mean from all embeddings.
3. **Estimate N**: `N_est = ||centered_sentence_sum|| / ||bias||`
4. **Subtract bias**: `token_signal = centered_sentence_sum - N_est × per_token_bias`
5. **Decompose**: Greedy residual subtraction against centered contextual vocabulary (cosine).

---

## Results

| Text | N | Est N | Recovery | Tokens Found |
|---|---|---|---|---|
| the cat sat on the mat | 6 | 5.6 | **2/6** | sat, cat + mat, The, on, cats, sitting |
| the dog ran in the park | 6 | 5.6 | **3/6** | in, dog, park + running, canine |
| Mary had a little lamb | 5 | 4.5 | **4/5** | had, a, little, lamb (only "Mary" missed) |
| I love cats and dogs | 5 | 4.5 | **1/5** | cats + breeds, loves, dog |
| Four score and seven years ago | 6 | 5.6 | **3/6** | years, seven, score + decade |
| the big red car drove fast down the road | 9 | 9.0 | **5/9** | big, down + driving, drive, downhill |

**Before bias subtraction**: 0% on all sentences.
**After bias subtraction**: 17-80% recovery, with semantically relevant near-misses.

---

## Comparison: Exact N vs Estimated N

| Text | Exact N | Estimated N |
|---|---|---|
| the cat sat on the mat | 0/6 | **2/6** |
| the dog ran in the park | 0/6 | **3/6** |
| Mary had a little lamb | 0/5 | **4/5** |
| Four score and seven years ago | 0/6 | **3/6** |

Estimated N consistently outperforms exact N. The estimate captures sentence-specific variation in bias magnitude that the average bias cannot.

---

## Supporting Findings

### Centering Reveals Static Semantics

Subtracting the vocabulary mean from contextual single-token embeddings reveals semantic structure matching static embeddings:

**Static neighbors of "cat"**: cats (0.69), Cat (0.69), cat (0.60), kitten (0.39)
**Centered contextual neighbors of "cat"**: Cat (0.80), Cat (0.66), rat (0.46), Cats (0.41), pet (0.39)

The contextual space is roughly: `contextual(token) ≈ shared_direction + static_semantics(token)`

### Linear Mapping: 89.9% Identity Preservation

A linear projection W (768×768) learned from all 50,257 paired (contextual, static) embeddings correctly maps 89.9% of contextual tokens to their nearest static counterpart. The transformation between spaces is largely linear for individual tokens.

### Energy Budget

| Component | Norm | % of Total |
|---|---|---|
| Contextual sentence sum (centered) | ~14,808 | 100% |
| Attention bias (N × per_token) | ~14,737 | 99.5% |
| Token content | ~97 | 0.5% |

The token-specific signal is less than 1% of the total energy. This explains why direct decomposition fails — the token signal is buried under a 200:1 bias.

---

## Remaining Challenges

1. **Bias variation**: The per-token bias varies by ~0.005% across sentences. At the 200:1 energy ratio, this variation (~460 norm) swamps the token signal (~97 norm). Better bias estimation could improve results.

2. **Positional bias**: The computed bias includes both attention and positional contributions. Explicitly subtracting wpe before computing the attention bias might improve separation.

3. **GPT-2 Small limitations**: Hidden states lack semantic discrimination (all sentences 0.99+ similar). A larger model or dedicated embedding model may have stronger token signals.

---

## Reproduction

```bash
cd experiments/contextual

# Build contextual vocabularies (one-time)
python3 build_ctx_vocab.py --layers 6,12

# Linear mapping experiment
python3 linear_map.py --text "the cat sat on the mat" --layer 6

# Bias subtraction (inline Python — see experiment notes)
```

---

## Re-run with Sink-Skip + N-Debias (2026-04-07)

Using the corrected method (`<|endoftext|>` prepended, position 0 excluded, `[<|endoftext|>, token]` vocab construction at position 1, `N × bias` debiasing where `bias` is computed from sink-skipped reference sentences). Output captured in `rerun_2026-04-07.txt`.

**Per-token bias norm at L1**: 16.4 (vs 2630 reported above for the sink-contaminated version)
**Per-token bias norm at L6**: 33.3 (vs 2637 reported above)

The bias norm collapsed by ~80–160× once the sink was removed.

### Results at L1 (best layer for sink-skip method)

| Text | Old (L6, contaminated) | New (L1, sink-skip) |
|---|---|---|
| the cat sat on the mat | 2/6 | **6/6** |
| the dog ran in the park | 3/6 | **5/6** |
| Mary had a little lamb | 4/5 | **5/5** |
| I love cats and dogs | 1/5 | **5/5** |
| Four score and seven years ago | 3/6 | **6/6** |
| the big red car drove fast down the road | 5/9 | **8/9** |
| **Total** | **18/37 (49%)** | **35/37 (95%)** |

### Results at L6 (the same layer the original experiment used)

| Text | Old (contaminated bias, norm-ratio N) | New (clean bias, exact N) |
|---|---|---|
| the cat sat on the mat | 2/6 | **4/6** |
| the dog ran in the park | 3/6 | **5/6** |
| Mary had a little lamb | 4/5 | 4/5 |
| I love cats and dogs | 1/5 | **2/5** |
| Four score and seven years ago | 3/6 | **5/6** |
| the big red car drove fast down the road | 5/9 | **6/9** |
| **Total** | **18/37 (49%)** | **26/37 (70%)** |

Even at the original target layer (L6), recovery jumps from 49% to 70% just by using the clean bias and exact N. At L1, where the per-token bias is smallest and the residual structure is freshest, recovery jumps to 95%.

### What this confirms

1. **Sink contamination was the dominant problem.** Removing the sink and using the correct multiplier produces a 2× improvement at L6 and a near-doubling at L1.
2. **Exact N is correct, not norm-ratio.** The "estimated N outperforms exact N" finding from the original experiment is inverted under the corrected method.
3. **L1 is the right layer for sink-skip decomposition.** The per-token bias is smallest there (16 vs 33 at L6) and the additive structure is freshest, making decomposition cleanest.
4. **Misses are still semantic neighbors.** "I love cats and dogs" misses 3/5 at L6 but recovers all 5 at L1; the L6 misses are tokens like ` lovers`, ` Dogs`, ` my` — semantic neighbors of the missing tokens, not garbage. The structure is intact, just slightly noisier at the deeper layer.

### Reproduction

```bash
cd experiments/contextual
python3 rerun_011.py
```
