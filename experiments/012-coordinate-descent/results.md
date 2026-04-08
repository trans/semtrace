# Experiment 012: Coordinate Descent Token Recovery

**Date**: 2026-04-03
**Code**: `experiments/contextual/coord_descent.py`

---

## REVISED INTERPRETATION (2026-04-07)

This experiment has two halves with different statuses under the sink discovery (see 010, 011 revised notes):

**Static half (Gettysburg, GPT-2 Small): UNAFFECTED.** The headline result — greedy 43.4% → CD 93.7% on Gettysburg — uses static embeddings via `wte.weight` lookups, no transformer forward pass, no attention. This finding stands as written and remains the most significant algorithmic result in the project.

**Contextual half: AFFECTED.** The "bias-subtraction + coordinate descent" results in the contextual table used the same sink-contaminated bias model as experiment 011. The diagnosis ("contextual landscape is noisy due to bias imprecision") was identifying the wrong cause — the noise was overwhelmingly the sink, not bias variation. The "Why It Works on Static, Not Contextual" section's claim that "the bias imprecision (~0.005% variation) creates residual noise of ~460 norm" is real but the noise source is the sink artifact, not per-token bias variation.

The "Bias Structure" speculative section (Primary/Secondary/Token signal) was correctly hypothesizing layered structure but described the wrong layers. With current understanding:
- "Primary bias 99.5%" → position-0 attention sink (a single-position artifact, not a per-token bias)
- "Secondary length-dependent" → the actual per-token bias (16 norm at L1 growing linearly to ~60 at L11)
- "Token signal <0.5%" → the per-position content, which is actually ~50–70 norm per non-sink position (not 0.5% of anything; the small ratio came from comparing against the sink-inflated total)

A re-run of the contextual half with sink-skip + N-debias initialization is appended at the end of this file.

The original content is preserved below for record.

---

## Objective

Test coordinate descent as an optimization-based alternative to greedy decomposition. Instead of making irrevocable decisions at each step, iteratively refine all token positions to minimize total residual.

---

## Algorithm

```
1. Initialize: run greedy decomposition to get starting token set
2. For each iteration:
   a. For each position i (0 to N-1):
      - Compute sum of all OTHER positions: others_sum = Σ tokens[j≠i]
      - Compute ideal vector for position i: ideal = target - others_sum
      - Find nearest token to ideal in vocabulary
      - If it improves total distance, swap it in
   b. If no position improved, stop (converged)
3. Return final token set
```

Each position is optimized while holding all others fixed. This is coordinate descent in the space of token assignments, with the objective function `||target - Σ embeddings[tokens]||`.

---

## Results

### Static Embeddings: GPT-2 Small (768d) on Gettysburg Address

| Method | Unique Recovery | Improvement |
|---|---|---|
| Greedy | 62/143 (43.4%) | — |
| **Coordinate Descent** | **134/143 (93.7%)** | **+50.3 pts** |

Coordinate descent converges in 10 iterations, more than doubling accuracy. **GPT-2 Small with coordinate descent matches GPT-2 XL with plain greedy** (93.7% in both cases).

This is the most significant algorithmic improvement found in this project.

### Contextual Embeddings: Bias Subtraction + Coordinate Descent

| Text | N | Greedy | Coord Descent |
|---|---|---|---|
| the cat sat on the mat | 6 | 3/6 | 3/6 |
| Mary had a little lamb | 5 | 5/5 | 4/5 |
| the dog ran in the park | 6 | 3/6 | 3/6 |
| I love cats and dogs | 5 | 0/5 | 0/5 |

Coordinate descent provides minimal improvement on contextual embeddings. The bias-subtracted target is noisy, creating local minima that prevent coordinate descent from finding better solutions. The contextual landscape has bumps where the residual increases before decreasing again, trapping the optimizer.

---

## Why It Works on Static, Not Contextual

**Static space has a single global minimum at zero residual.** The target IS a sum of token embeddings, so the correct assignment produces exactly zero distance. Every step of coordinate descent gets strictly closer. The landscape is convex.

**Contextual space has a noisy landscape.** After bias subtraction, the target is approximately (but not exactly) a sum of centered contextual embeddings. The bias imprecision (~0.005% variation across sentences) creates residual noise of ~460 norm against a token signal of ~97 norm. This noise creates local minima that trap coordinate descent.

---

## Theoretical Significance

### The Optimization Framing

All decomposition methods are solving the same problem:

```
Find tokens T = {t₁, t₂, ..., tₙ} that minimize ||target - Σ E(tᵢ)||
```

- **Greedy**: makes locally optimal, irrevocable choices
- **Coordinate descent**: iteratively refines all positions toward global optimum
- **Brute force**: evaluates all possible combinations (intractable)

Greedy is fast but makes errors that cascade. Coordinate descent corrects cascading errors by revisiting earlier decisions. Brute force is optimal but computationally impossible.

### Bias Structure

The attention bias may have layered structure — not one flat constant, but strata:

1. **Primary bias** (~99.5% of energy): shared across all sentences, near-constant
2. **Secondary bias** (length/position-dependent): varies with sentence structure
3. **Token signal** (<0.5%): the actual content

Each layer of bias we subtract reveals more signal. Better bias modeling (accounting for sentence length, positional contribution, second-order effects) would smooth the contextual landscape and make coordinate descent more effective.

### The Role of Magnitude

Embedding magnitude carries critical information:
- **Token count N**: estimable from `||embedding|| / ||per_token_bias||`
- **Signal-to-noise ratio**: larger magnitude = more tokens = more signal (but also more bias)
- **Convergence direction**: the residual norm guides both N estimation and decomposition

L2 normalization destroys all of this. Recovery techniques that depend on magnitude — including bias subtraction, N estimation, and the residual norm stopping condition — cannot operate on normalized embeddings without somehow recovering or estimating the original magnitude.

### Local Minima and Search

In static space, the objective landscape is smooth — every correct token substitution strictly reduces the residual. In contextual space, bias imprecision creates a noisy landscape with local minima. This explains why:

- Greedy works perfectly in static space (no cascading errors)
- Coordinate descent dramatically improves static space (corrects cascade errors)
- Both methods plateau in contextual space (trapped by landscape noise)

Further progress on contextual decomposition requires either:
1. Better bias removal (smoother landscape)
2. Stochastic search (simulated annealing, random restarts)
3. The black box API approach (use the model itself as the scoring function)

---

## Reproduction

```bash
cd experiments/contextual

# Build contextual vocabularies (if not already built)
python3 build_ctx_vocab.py --layers 6,12

# Run coordinate descent
python3 coord_descent.py --text "the cat sat on the mat" --mode both

# Run on Gettysburg Address
python3 coord_descent.py --file ../texts/gettysburg.txt --mode static
```

---

## Re-run of Contextual Half with Sink-Skip + N-Debias (2026-04-07)

Coordinate descent on top of greedy decomposition, both initialized with the corrected method (sink-skip, N-corrected debias, `[<|endoftext|>, token]` vocab construction at position 1, decomposition at L1). Output captured in `rerun_2026-04-07.txt`.

| Sentence | Old Greedy | Old CD | New Greedy | New CD |
|---|---|---|---|---|
| the cat sat on the mat | 3/6 | 3/6 | **6/6** | **6/6** |
| Mary had a little lamb | 5/5 | 4/5 | **5/5** | **5/5** |
| the dog ran in the park | 3/6 | 3/6 | **5/6** | **6/6** |
| I love cats and dogs | 0/5 | 0/5 | **5/5** | **5/5** |
| **Total** | **11/22 (50%)** | **10/22 (45%)** | **21/22 (95%)** | **22/22 (100%)** |

Coordinate descent on the sink-corrected landscape achieves **perfect recovery** on all four sentences. Greedy alone reaches 95%; CD finds the one token greedy missed (` ran` in "the dog ran in the park") and pushes it to 100%.

The original interpretation that "the contextual landscape is noisy due to bias imprecision" was diagnosing the wrong cause. With the sink removed, the landscape is clean enough that CD acts exactly as it does in static space — refining the greedy initialization toward a better global minimum. The "0% on I love cats and dogs" result in particular flips to **5/5 with greedy alone and 5/5 with CD**, an outright reversal.

The "Bias Structure" speculation about layered structure was directionally right but had the layers wrong. The corrected layering, confirmed by the per-layer bias growth (Section 6.5 of the paper), is:

1. Position-0 attention sink (~2700 norm) — single-position artifact
2. Per-token bias growing linearly from 16 (L1) to 60 (L11) — actual layer-by-layer additive sediment
3. Per-position content (~50–70 norm) — semantic signal

### Reproduction

```bash
cd experiments/contextual
python3 rerun_012_contextual.py
```
