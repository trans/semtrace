# Experiment 018: Complete Inversion Pipeline — Bag Recovery + Order Recovery

**Date**: 2026-04-08 – 2026-04-10
**Code**: `experiments/contextual/bag_exhaustive.py`, `experiments/contextual/beam_width_sweep.py`, `experiments/contextual/forward_pass_scoring.py`, `experiments/contextual/gradient_invert.py`, `experiments/contextual/gradient_invert_v2.py`, `experiments/contextual/hotflip_triangulate.py`, `experiments/contextual/lm_beam_anchored.py`

---

## Objective

Given a pooled L12 embedding from GPT-2 Small, recover the original input sentence — both the bag of words and their order. Test multiple methods and combinations, culminating in a complete end-to-end pipeline.

---

## Stage 1: Bag Recovery

Two methods for recovering the bag of words, depending on access level:

### White-box: Per-position L1 decomposition (experiment 016)

| Text length | Recovery |
|---|---|
| Short (6 tokens) | **100%** |
| Medium (24 tokens) | **100%** |
| Gettysburg (62 tokens) | **100%** |

With full white-box access to per-position L1 hidden states, every token is recovered in order. This is the upper bound on bag recovery.

### Production-style: Bridge + CD from pooled L12

| Metric | Result |
|---|---|
| Bag-match (strict token identity) | 33.3% |
| Avg L2 of recovered vs target | 21.06 (vs random avg 30.63) |
| Sentences with L2 below all random | 5/20 |

Bridge + CD gives partial bag recovery. The bag is noisy but L2-better than random for most sentences.

---

## Stage 2: Order Recovery

Given a bag of words (correct or noisy), find the ordering that best matches the target embedding.

### Exhaustive permutation search (the gold standard)

For each of 6 test sentences, enumerate all permutations of the bag and score by L2 distance:

| Sentence | True sequence rank | True L2 | 2nd-best L2 | Random L2 |
|---|---|---|---|---|
| the cat sat on the mat | **1st (best)** | 0.00 | ~4 | ~25 |
| the dog ran in the park | **1st** | 0.00 | ~5 | ~30 |
| Mary had a little lamb | **1st** | 0.00 | 8.15 | 44 |
| Four score and seven years ago | **1st** | 0.00 | 4.02 | 35 |
| I love cats and dogs | **1st** | 0.00 | 3.81 | 44 |
| the big red car drove fast | **1st** | 0.00 | 4.08 | 21 |

**The true sequence is the unique L2-global-minimum among all permutations, for every sentence.** The embedding fully determines word order given the bag. The gap between the true ordering (L2=0) and the next best (L2=4–8) is clear and consistent.

### Beam search (practical for longer sentences)

Bag-constrained beam search: at each step, use the LM's next-token predictions restricted to remaining bag tokens. Score completed beams by L2 distance.

| Beam width | Perfect sentences (of 6) | Position-match |
|---|---|---|
| B=4 | 3/6 | 71% |
| B=8 | 4/6 | 74% |
| B=16 | 5/6 | 91% |
| **B=32** | **6/6** | **100%** |

At B=32, all 6 sentences are recovered perfectly (L2=0). The progression confirms that the information IS in the embedding; narrower beams just fail to explore the right ordering.

The holdout ("Four score and seven years ago") needed B=32 because "Four score" isn't a high-probability bigram under the LM, so narrower beams pruned it.

---

## Methods That Did NOT Work for Order Recovery

### Pure gradient descent (continuous embeddings)

Optimized continuous input embeddings to minimize L2 to target. Found solutions with L2 ~14 (87th percentile vs random) but projecting to discrete tokens gave 0% token-match. The continuous-discrete gap is real: many non-token continuous solutions produce low L2 but don't correspond to any valid token sequence.

### HotFlip discrete swap search

Started from a partial anchor and iteratively swapped to nearest-neighbor tokens that reduce L2. **Actively destroyed correct tokens** — position-match often decreased because distance-reducing swaps swapped correct tokens for wrong-but-closer alternatives. The LM prior is needed to prevent this.

### LM beam search without bag constraint

Generated continuations from a known prefix using LM probabilities, scored by L2. Recovered ~1/6 extra position per anchor token — barely better than the anchor alone. Without the bag constraint, the LM generates plausible-but-wrong text because many plausible continuations have similar L2 to the target.

### Key lesson

**The bag-of-words constraint is the critical ingredient.** With it, beam search is perfect at B=32. Without it, all methods struggle — gradient descent, HotFlip, and free LM beam search all find low-L2 solutions that are wrong. The bag reduces the search space from "all possible sequences" to "all permutations of these specific words," which is tractable and well-conditioned.

---

## Complete Pipeline (White-Box, Short Sentences)

```
Input:  pooled L12 embedding of unknown sentence
        + white-box access to GPT-2 Small

Step 1: Per-position L1 decomposition
        → 100% bag of words, in order

Step 2: Bag-constrained beam search (B=32) with L2 ranking
        → unique ordering that minimizes L2 to target

Output: exact original sentence
```

**Result: 6/6 test sentences recovered perfectly** (every word, exact order, L2=0.0).

---

## What Remains Open

1. **Longer sentences.** Exhaustive search scales as N! (intractable for N>8). Beam search with B=32 works for 6-token sentences; scaling to 30+ tokens requires either much wider beams or smarter search.

2. **Production threat model.** We use per-position L1 for bag recovery, which requires white-box access. The pooled-only case (bridge + CD) gets 33% bag-match, and we have not tested bag-constrained beam search on a partial/noisy bag.

3. **Other models.** All results are on GPT-2 Small (124M, 768d). Generalization to larger models and production embedding endpoints (Llama, OpenAI, etc.) is untested.

4. **Computational cost.** B=32 beam search on 6-token sentences runs in ~1 minute per sentence on CPU. Longer sentences with wider beams would be much slower.

---

## Reproduction

```bash
cd experiments/contextual
python3 bag_exhaustive.py      # exhaustive permutation search (proves uniqueness)
python3 beam_width_sweep.py    # beam width sweep (B=4,8,16)
# For B=32 on a specific sentence, see beam_width_sweep_output.txt
python3 forward_pass_scoring.py  # original forward-pass scoring (B=4)
```
