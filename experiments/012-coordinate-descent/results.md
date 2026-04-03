# Experiment 012: Coordinate Descent Token Recovery

**Date**: 2026-04-03
**Code**: `experiments/contextual/coord_descent.py`

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
