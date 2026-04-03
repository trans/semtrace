# BIGWIN: Full Contextual Embedding Inversion

**Thomas Sawyer**^1 **& Claude Opus 4.6**^2
*^1 Independent Researcher · ^2 Anthropic*

---

## The Goal

Given a contextual embedding vector and black-box access to the embedding API, recover the original text — words AND order.

---

## The Pipeline

### Stage 1: Bias Subtraction → Candidate Tokens

**Status**: Proven (Experiment 011)

- Subtract the constant attention bias (precomputed from reference sentences)
- Estimate N from embedding magnitude: `N ≈ ||embedding|| / ||bias||`
- Greedy decompose the de-biased signal against centered contextual vocabulary
- **Output**: partial bag of words (17-80% of tokens), semantic neighbors of missing tokens

**Requirements**: Pre-built contextual vocabulary, precomputed bias vector, unnormalized embedding.

**Limitation**: Normalized embeddings lose magnitude → N estimation fails. Binary search over N is fallback (log₂ search, ~8 evaluations).

### Stage 2: Coordinate Descent → Refined Bag

**Status**: Proven on static (Experiment 012), untested on contextual at scale

- Initialize from Stage 1's candidate tokens
- Iteratively optimize each position: hold others fixed, find the token that minimizes `||target - Σ tokens||`
- Converges in ~10 iterations
- **Output**: refined bag of words with significantly higher accuracy

**On static embeddings**: 43% → 94% on Gettysburg (GPT-2 Small). Matches XL greedy.

**On contextual**: limited by landscape noise from imperfect bias subtraction. Better bias modeling (layered subtraction per attention layer) would smooth the landscape.

### Stage 3: Order Recovery via API → Full Text

**Status**: Conceptual

- Take the refined bag of N tokens from Stage 2
- Generate candidate orderings
- Feed each through the embedding API
- Score: `similarity(API(candidate), target_embedding)`
- The ordering whose embedding most closely matches the target wins

**Search strategy**:
- N! orderings is intractable for N > ~10
- But most orderings are grammatically nonsensical
- Constrain with:
  - Language model scoring (perplexity) to prune implausible orderings
  - Beam search: start with highest-confidence tokens in likely positions, expand
  - Positional affinity: `dot(E(token), P(position))` as a prior
  - Coordinate descent on ORDER: swap pairs of tokens, score via API, keep improvements

**API calls**: For N=10 with beam width 5 and 10 iterations of coordinate descent: ~500 calls. Tractable.

---

## What Each Stage Contributes

| Stage | Input | Output | Accuracy (est.) |
|---|---|---|---|
| 1: Bias subtraction | Raw embedding | ~50% of tokens + neighbors | Proven |
| 2: Coordinate descent | Stage 1 candidates | ~90% of tokens (bag) | Proven on static |
| 3: Order search | Stage 2 bag | Ordered sentence | Conceptual |

---

## Open Problems

### Normalized Embeddings

Most embedding APIs L2-normalize their output. This destroys:
- Token count N (magnitude encodes it)
- Signal-to-noise ratio (bias dominates after normalization)
- Residual norm (stopping condition for decomposition)

**Mitigations**:
- Binary search over N (~8 evaluations)
- External N estimation (document metadata, typical sentence lengths)
- Train a magnitude predictor from (text, magnitude) pairs

### Layered Bias Structure

The attention bias likely has strata — one per transformer layer + normalization. We computed the aggregate. Peeling layers individually might yield:
- Cleaner token signal after subtraction
- Smoother optimization landscape for coordinate descent
- Better N estimation

We have GPT-2's per-layer hidden states to test this.

### Scaling to Long Text

Short sentences (5-10 tokens): all three stages are tractable.
Medium text (50-100 tokens): Stage 1+2 work, Stage 3 order search becomes hard.
Long text (500+ tokens): Stage 1+2 recover partial bag, Stage 3 may need to operate on chunks.

### The Pretty Solution

The closed-form inverse of a transformer — `solve attention(tokens) = embedding for tokens` — would make all three stages unnecessary. Attention involves softmax (non-invertible in general), layer normalization, GELU activations, and residual connections. No known analytical inverse exists. If one were found, it would be a major result in deep learning theory.

---

## Implementation Plan

### Phase A: Improve Bias Subtraction
1. Compute per-layer bias (using GPT-2 container, all 12 layers)
2. Subtract layered biases sequentially
3. Test if the cleaned signal improves coordinate descent results

### Phase B: Contextual Coordinate Descent
1. Run coordinate descent on bias-subtracted contextual embeddings
2. Test on the full test suite (Gettysburg, Mary, Tale of Two Cities)
3. Compare to static decomposition ceiling

### Phase C: Black Box Order Recovery
1. Implement API-based scoring: `score = cosine(embed(candidate), target)`
2. Start with known bag (cheating) to test order recovery in isolation
3. Implement beam search over orderings with LM perplexity pruning
4. Connect to Ollama for live scoring

### Phase D: Full Pipeline Integration
1. Stage 1 → Stage 2 → Stage 3 end-to-end
2. Test on normalized embeddings with binary search for N
3. Benchmark against known text for accuracy measurement

---

## Security Implications

If this pipeline works:
- **Vector databases** storing embeddings of sensitive text are vulnerable to content recovery
- **RAG systems** that expose embeddings leak their knowledge base
- **Embedding-based search** reveals query content to the embedding provider
- The defense is straightforward: don't expose raw embeddings. But many systems do.

---

## Why This Matters Beyond Security

- **Reasoning over embeddings**: operate on continuous representations without decoding to text
- **Embedding-only storage**: store the vector, recover approximate text on demand
- **Semantic compression**: fixed-size representation regardless of text length
- **Interpretability**: read what the model "thinks" at any point in its computation
- **Cross-model translation**: map between embedding spaces to transfer knowledge

---

*The model remembers. The question is whether we can read its memory.*
