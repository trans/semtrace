# Experiment 016: Corrected Contextual Decomposition

**Date**: 2026-04-07 – 2026-04-08
**Code**: `experiments/contextual/rerun_011.py`, `experiments/contextual/rerun_012_contextual.py`, `experiments/contextual/rerun_013_quantization.py`, `experiments/contextual/vocab_sanity.py`, `experiments/contextual/method_comparison.py`, `experiments/contextual/per_position_decomp.py`

---

## Objective

Re-run experiments 011, 012, and 013 with the sink-skip correction (experiment 015) and comprehensively evaluate all contextual decomposition methods on the same test set. Also verify that the corrected contextual vocabulary has real semantic structure.

---

## Re-run Results (Sink-Skip + N-Debias, 6 Sentences)

### Experiment 011 (Attention Bias) — L1

| Text | Old (contaminated) | New (sink-skip L1) |
|---|---|---|
| the cat sat on the mat | 2/6 | **6/6** |
| the dog ran in the park | 3/6 | **5/6** |
| Mary had a little lamb | 4/5 | **5/5** |
| I love cats and dogs | 1/5 | **5/5** |
| Four score and seven years ago | 3/6 | **6/6** |
| the big red car drove fast down the road | 5/9 | **8/9** |
| **Total** | **49%** | **95%** |

### Experiment 012 (Coordinate Descent) — Contextual Half, L1

| Text | Old Greedy | Old CD | New Greedy | New CD |
|---|---|---|---|---|
| the cat sat on the mat | 3/6 | 3/6 | 6/6 | **6/6** |
| Mary had a little lamb | 5/5 | 4/5 | 5/5 | **5/5** |
| the dog ran in the park | 3/6 | 3/6 | 5/6 | **6/6** |
| I love cats and dogs | 0/5 | 0/5 | 5/5 | **5/5** |
| **Total** | **50%** | **45%** | **95%** | **100%** |

Coordinate descent on the corrected landscape achieves **100%** on all four sentences. The original "noisy landscape" that trapped CD was the sink artifact.

### Experiment 013 (Quantization) — Contextual Half, L1

| Bits | Old (contaminated L6) | New (sink-skip L1) |
|---|---|---|
| f32 | 6/21 | **13/21** |
| f16 | 4/21 | **13/21** |
| int8 | 0/21 | **13/21** |
| int4 | 0/21 | 4/21 |

The "quantization destroys contextual signal" finding is inverted: int8 and f16 are indistinguishable from f32. Only int4 degrades meaningfully.

---

## Vocabulary Sanity Checks

### Semantic Clustering

Sink-corrected contextual vocab at L11 (nearest neighbors of probe tokens):

```
' dog': 'dog', ' Dog', 'Dog', ' puppy', ' dogs', ' canine', ' kitten', ' cat'
' happy': 'happy', ' joyful', ' unhappy', ' thrilled', ' pleased', ' delighted', ' ecstatic'
' king': ' queen', 'king', ' kings', ' kingdom', ' prince', ' King', ' emperor', ' monarch'
```

The corrected vocab has clean semantic clustering at every layer L1–L11. Arguably cleaner than static `wte` at some layers (no string-similarity false positives like `car` appearing in `cat`'s neighbors).

### Position Invariance

How close is ` cat` at position 1 of `[EOT, cat]` (the vocab entry) to ` cat` at deeper positions in longer sentences?

| Position in sentence | L1 cos to canonical | L6 cos | L11 cos |
|---|---|---|---|
| 1 (canonical) | 1.00 | 1.00 | 1.00 |
| 2 | 0.99 | 0.91 | 0.97 |
| 4 | 0.96 | 0.85 | 0.93 |
| 8 | 0.94 | 0.84 | 0.88 |
| 12 | 0.92 | 0.82 | 0.86 |

Position drift is gentle. The vocab entry is a reasonable proxy for the same token at other positions, with gradual degradation.

### Sentence Discrimination (Post-Sink)

Cosine between sentence-pair sums after sink-skip (L6 trailing positions):

| Pair | Raw (sink-included) | Sink-skipped |
|---|---|---|
| Same words, swapped order | 1.0000 | 0.9975 |
| Synonym swap (mat → rug) | 0.9998 | 0.9872 |
| Unrelated (cat / quantum physics) | **0.9971** | **0.8036** |
| Adjective swap (quick / lazy) | 0.9997 | 0.9662 |

The "GPT-2 hidden states lack semantic discrimination" finding was a sink artifact. With sink-skip, L6 clearly distinguishes unrelated content (cos 0.80 vs synonym pairs at 0.99).

---

## Method Comparison (20 Sentences, 132 Unique Tokens)

Best methods sorted by recovery:

| Method | Recovery |
|---|---|
| Static upper bound (greedy) | 132/132 (100%) |
| L1 int8 low band contextual | 118/132 (89.4%) |
| L1 contextual + N-debias | 117/132 (88.6%) |
| L6 int8 low band contextual | 88/132 (66.7%) |
| L6 contextual + N-debias | 86/132 (65.2%) |
| L1 bridge + bias + CD | 77/132 (58.3%) |
| L11 int8 low band contextual | 60/132 (45.5%) |
| L11 contextual + N-debias | 57/132 (43.2%) |
| L12 bridge + bias + CD | 44/132 (33.3%) |
| L12 contextual decomp | 2/132 (1.5%) |

Union of methods at L11 (contextual + int8 low band + bridge CD): **83/132 (63%)**.

### Per-Position L1 Decomposition

Instead of summing across positions and decomposing the sum, decompose each position independently against the contextual vocab:

| Text | L1 per-position | L6 | L11 | L12 |
|---|---|---|---|---|
| short (6 tokens) | **6/6** | 5/6 | 2/6 | 2/6 |
| medium (24 tokens) | **24/24** | 16/24 | 8/24 | 1/24 |
| gettysburg (62 tokens) | **62/62** | 34/62 | 11/62 | 1/62 |

**Per-position L1 achieves 100% recovery on Gettysburg (62 tokens, in order).** This is white-box only (requires per-position hidden states at L1).

---

## Architecture Clarifications

- **The residual stream is NOT normalized between L0–L11.** Only the inputs to sublayers (attention, MLP) are normalized via ln_1 and ln_2. The final `ln_f` normalizes once at L12.
- **Weight tying**: `wte` and the LM head are the same matrix. Cosine decomposition against `wte` rows is mathematically equivalent to applying the LM head and taking argmax.
- **L12 hidden states are prediction-oriented**, encoding "what comes next" not "what I am." Per-position L12 decomposition gives 40% top-1 via the logit lens (project through `wte.T`), but direct vocab matching fails (1.5%).

---

## Reproduction

```bash
cd experiments/contextual
python3 rerun_011.py                # re-run experiment 011
python3 rerun_012_contextual.py     # re-run 012 contextual half
python3 rerun_013_quantization.py   # re-run 013 contextual half
python3 vocab_sanity.py             # semantic clustering + position invariance
python3 method_comparison.py        # comprehensive method stack-rank
python3 per_position_decomp.py      # per-position decomposition at all layers
```
