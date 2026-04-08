# Experiment 014: Strata Test / Per-Layer Bias and Token Signal Survival

**Date**: 2026-04-03 (original run), 2026-04-07 (revised interpretation)
**Code**: `signal_survival.py`, `identity_vs_prediction.py`, `../contextual/strata_test.py`
**Original output**: `signal_survival.txt`, `identity_vs_prediction.txt`, `medium_decomposition.txt`, `l12_projection.txt`

---

## REVISED INTERPRETATION (2026-04-07)

The data this experiment collected is good. The interpretation needs to be split into two phenomena that were tangled together at the time, both of which the data already shows clearly.

### Finding 1: Position 0 is the attention sink

From `signal_survival.txt` on "the cat sat on the mat":

```
 Layer       the       cat       sat        on       the       mat
   emb    0.2795    0.5157    0.5824    0.4990    0.5482    0.6570
    L1    0.1581    0.1206    0.0992    0.1048    0.1644    0.1412
   L11    0.0855    0.0095   -0.0405    0.1194    0.1955   -0.0392
```

Position 0 (`the`, the first token) behaves fundamentally differently from every other position:

- Its starting cosine to its own static token (0.28) is *lower* than every other position's, even though it is identical content to position 4 (also `the`).
- Its rank to itself stays around 158 from L4 onward — frozen, not degrading like other positions.
- Its cosine to itself stays around 0.08–0.16 throughout — also frozen.

This is the position-0 attention sink. The model is using position 0's hidden state as a scratch space for unused attention budget, overwriting its original token-identity content with sink material that has no per-token specificity. The signal survival data was already showing this in 2026-04-03 — we just didn't recognize what we were looking at until we instrumented the per-block contributions in `attention_trace.py` four days later.

### Finding 2: Late positions become prediction-oriented (the identity vs prediction story)

From `identity_vs_prediction.txt`:

```
 Layer   Avg Identity Rank   Avg Prediction Rank
   emb                   1                     5
    L1                  16                    60
    L6                  81                   575
   L11                 277                   395
   L12               44260                   373
```

Setting aside position 0 (which is the sink), the trailing positions show a smooth degradation of identity (the rank of the position's own token grows from 1 to ~277 across L0–L11) accompanied by a competing growth of prediction (the rank of the *next* token, after applying `wte.T` and softmax). At L12 (post-`ln_f`), the representation has been transformed into a prediction-oriented basis where the identity rank collapses to ~44k while prediction rank stays around 373.

This is consistent with the well-known logit-lens line of work (Geva et al., nostalgebraist 2020) showing that late hidden states encode "what comes next" rather than "what I am." The original framing in this experiment correctly identified the prediction-oriented transition but did not separate it from the position-0 sink phenomenon, so the "average rank" numbers are pulled around by both effects at once.

### Finding 3: GPT-2 Medium decomposition (`medium_decomposition.txt`)

The medium-model decomposition test reproduces the same sink contamination pattern at L6, L17, L23 of GPT-2 Medium. The reported "Bias norm: 3302" / "3416" / "1543" values are sink magnitudes at those layers; the "Sentence (debiased): 5/6 / 1/6 / 1/6" results are floor numbers under the same contaminated method as 011. A re-run with sink-skip on Medium would tell us whether the sink+method scales with model size; we have not yet done this.

### What the per-layer bias measurement (Section 6.6 of the paper) was actually measuring

The bias norms in `strata_test.py` (and reproduced in the paper's Section 6.6 original table) — going from 7.6 at L0 to ~2700 by L3 and plateauing — were measuring sink magnitudes, not per-token bias. The clean per-token bias (after sink-skip) grows monotonically and roughly linearly: 16 → 21 → 28 → 37 → 45 → 60 across L1–L11. The "plateau" was the sink filling up to its asymptotic value; the linear growth is the actual additive sediment from each transformer block.

### Action items

1. The data files (`signal_survival.txt`, `identity_vs_prediction.txt`, `l12_projection.txt`) are correct as raw measurements and do not need re-running. Only the interpretation changes.
2. `medium_decomposition.txt` should be re-run with sink-skip to give a clean measurement of per-token bias growth on GPT-2 Medium and to test whether the L1 → L9 5/5 result on Small generalizes.
3. The strata hypothesis (paper Section 6.5) is now well-defined and can be tested via layer peeling — using the clean per-layer biases (16, 21, 28, 37, 45, 60) to walk a deep hidden state inward through the network. This is the most direct experimental follow-up.

The original outputs are preserved in `signal_survival.txt`, `identity_vs_prediction.txt`, `medium_decomposition.txt`, `l12_projection.txt`.
