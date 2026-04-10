# Experiment 017: Cosine Saturation, L2 as the Correct Metric, and Centering

**Date**: 2026-04-08
**Code**: `experiments/contextual/verify_cosine_baseline.py`, `experiments/contextual/l2_ranking.py`, `experiments/contextual/centering_analysis.py`

---

## Objective

Investigate whether cosine similarity in pooled L12 space is a reliable metric for evaluating inversion quality. Prompted by the observation that gradient-descent "garbage" tokens scored cos 0.997 to target — suspiciously high for completely wrong tokens.

---

## Discovery: Cosine in Pooled L12 is Saturated

### Baseline Measurement

| Comparison | Cosine | L2 distance |
|---|---|---|
| Self (sanity) | 1.000 | 0.00 |
| Two unrelated meaningful sentences | 0.992–0.998 | 16–46 |
| Random 6-token sequence vs target | **mean 0.994, max 0.998** | mean 22–46, min 10–19 |
| Gradient-descent garbage vs target | 0.997 | 14.6 |

**Random 6-token sequences have cosine 0.99+ to any target sentence.** The metric is essentially saturated. 26% of random sequences scored HIGHER cosine than the gradient-descent garbage — meaning the garbage wasn't even at the noise ceiling.

### Root Cause: The Centroid

Pooled L12 embeddings decompose as `centroid + per_sentence_content`:

| Component | Norm | Fraction of total |
|---|---|---|
| Centroid (shared direction, mean of 40 sentences) | 207 | **95%** |
| Per-sentence content (deviation from centroid) | 15.6 | 5% |

All pooled L12 vectors share a 207-norm common direction. Cosine measures the angle between full vectors — when 95% of each vector is the same, the angle is always near 0 (cos near 1), regardless of what the 5% per-sentence part contains.

---

## L2 Distance IS Discriminative

Unlike cosine, L2 distance is sensitive to the per-sentence variation:

| Comparison | L2 distance | Interpretation |
|---|---|---|
| Self | 0.0 | Perfect match |
| Permutations of right bag | 4–8 | Close — right words, wrong order |
| Unrelated meaningful sentences | 16–46 | Moderate distance |
| Random 6-token sequences | 10–25 (mean ~22) | Noise floor |
| Gradient-descent garbage | 14.6 | Between noise and meaningful (87th percentile) |

L2 clearly separates the right answer (0) from permutations (4–8) from random (10–25). Cosine cannot make any of these distinctions.

Key finding from L2 re-evaluation: forward-pass beam search results we previously dismissed as "near-misses" (cos 0.999, looked like noise) actually had L2=7.4 — better than 100% of random sequences tested.

---

## Centering Makes Cosine Useful

Subtracting the centroid from each embedding before computing cosine recovers discriminative power:

| Comparison | Raw cosine | Centered cosine |
|---|---|---|
| Between 40 real sentences (pairwise) | mean 0.997 | mean −0.024 (full range −0.91 to +0.92) |
| Random vs target | mean 0.994 | mean 0.469 |
| Permutations vs target | mean 0.996 | mean **0.847** |
| **Discrimination gap (perm − random)** | **+0.001** | **+0.378** |

Centering improves the discrimination gap by **378×**.

Centered cosine also reveals unexpected semantic structure:

| Sentence pair | Raw cos | Centered cos |
|---|---|---|
| Synonym swap (mat → rug) | 1.000 | 0.996 |
| Word order swap (cat chases dog / dog chases cat) | 1.000 | 0.905 |
| Negation (happy / not happy) | 0.999 | 0.980 |
| Unrelated (cat-on-mat / quantum physics) | 0.997 | 0.762 |
| Different content, same pattern (cat / dog sentences) | 0.998 | **−0.576** |

After centering, the two animal-sentences are *negatively correlated* — the 5% per-sentence content vectors point in opposite directions even though both are grammatically similar short English sentences.

---

## What This Changes

### Re-evaluation of gradient descent

Gradient-descent "garbage" (cos 0.997) was not a dramatic failure of an otherwise-good method — it was a mediocre result (87th percentile by L2, 94th by centered cosine) whose mediocrity was hidden by the saturated metric.

### Re-evaluation of forward-pass beam search

The "wrong order, near-miss" results (cos 0.999) were genuine near-misses, not noise. L2=7.4 is well below the random floor (~13 minimum). The beam search was finding meaningfully good answers.

### Re-evaluation of bridge methods

Bridge + CD at L12 gets 33% bag-match, but L2 evaluation shows 5/20 sentences have recovered-sequence L2 below ALL random baselines. The bridge IS finding structurally meaningful answers; bag-match was an overly strict metric.

### Practical recommendation

**Never use raw cosine for evaluating or ranking in pooled embedding space.** Use L2 distance or centered cosine. Raw cosine is saturated by the centroid and cannot distinguish good from bad answers.

---

## Reproduction

```bash
cd experiments/contextual
python3 verify_cosine_baseline.py  # cosine saturation + baseline
python3 l2_ranking.py              # L2 distance distributions
python3 centering_analysis.py      # centering + discrimination gap
```
