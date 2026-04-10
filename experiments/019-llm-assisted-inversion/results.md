# Experiment 019: LLM-Assisted Inversion and Leave-One-Out Diagnostics

**Date**: 2026-04-10
**Code**: `experiments/contextual/beam_width_sweep.py`, `experiments/contextual/mary_full_pipeline.py`, `experiments/contextual/hybrid_beam.py`, `experiments/contextual/llm_plus_refine.py`, `experiments/contextual/leave_one_out.py`

---

## Objective

Push beyond the 6-token beam search result (6/6 perfect at B=32) to longer sentences. Test whether LLM-assisted ordering and leave-one-out diagnostics can close the gap between bag recovery (100%) and order recovery (which degrades with sentence length).

---

## Beam Width Sweep (Short Sentences, 6 Tokens)

Bag-constrained beam search with L2 ranking on 6 test sentences:

| Beam width | Perfect sentences | Position-match |
|---|---|---|
| B=4 | 3/6 | 71% |
| B=8 | 4/6 | 74% |
| B=16 | 5/6 | 91% |
| B=32 | **6/6** | **100%** |

At B=32, all 6 short sentences are recovered perfectly (L2=0). The holdout ("Four score and seven years ago") needed B=32 because "Four score" isn't a high-probability bigram under the LM.

---

## Scaling to 23 Tokens (Mary Had a Little Lamb)

### Plain beam search

| Beam width | Best L2 | Position-match | Random L2 |
|---|---|---|---|
| B=8 | 16.16 | 3/23 | 37–57 |
| B=16 | 12.44 | 3/23 | |
| B=32 | 8.60 | 1/23 | |

L2 improves with beam width (16 → 12 → 8.6) but position-match stays low. The search space at 23 tokens (21! unique permutations) is too vast for beam search alone.

### LLM perplexity scoring (GPT-2 Small)

Scored 2000 random permutations by GPT-2's log probability:

- True sequence log-prob: -112.41
- Best of 2000 random: -163.14
- Random mean: -190.08
- **True sequence rank: 0/2000** (better than ALL random permutations)

GPT-2's perplexity CAN discriminate the right ordering — the signal is strong (50 log-prob unit gap). But 2000 samples out of 21! ≈ 5×10^19 is insufficient to find it by random sampling.

### LLM-assisted ordering via Mistral 7B

Asked Mistral 7B (via Ollama) to rearrange shuffled words into a sentence:

| Sentence type | Mistral result |
|---|---|
| Famous (Mary had a little lamb) | **Perfect** — recognized the nursery rhyme |
| Non-famous (the old man fished...) | All words present, minor reordering ("slowly" moved) |
| Non-famous (snow fell quietly...) | All words present, reordered into valid alternative |
| Non-famous (scientists discovered...) | All words present, minor additions ("The") |

Llama 3.2 3B failed at this task (hallucinated meta-text). **Mistral 7B at the same task followed the constraint well**, getting all words present with minor stylistic reorderings on non-famous sentences.

### Combined pipeline: Mistral → swap refinement

| Sentence | Tokens | Mistral L2 | After swap refinement | True L2 | Random L2 |
|---|---|---|---|---|---|
| old man fished | 13 | 12.64 | **2.96** | 0 | ~40 |
| snow fell quietly | 12 | 28.48 | **6.05** | 0 | ~45 |
| Mary had a lamb | 23 | 24.76 | **3.48** | 0 | ~57 |

Swap refinement (pairwise token swaps scored by L2) closes most of the gap from Mistral's initial ordering. For the 23-token Mary rhyme, L2 goes from 24.8 → 3.5, which is **94% of the way from random to perfect**.

---

## Leave-One-Out Diagnostics

### On the TRUE sequence

Removing each token and measuring L2 change reveals **per-position importance**:

| Most important (removal hurts most) | Delta |
|---|---|
| ' had' at pos 1 | +13.8 |
| ' a' at pos 2 | +8.9 |
| 'Mary' at pos 0 | +8.3 |
| ' its' at pos 5 | +7.2 |

| Least important (removal hurts least) | Delta |
|---|---|
| ' the' at pos 17 | +1.2 |
| ' go' at pos 22 | +1.3 |
| ' was' at pos 19 | +1.5 |

Content words contribute more to the embedding's identity than function words. All deltas are positive for the true sequence (every token is correctly placed).

### On a SHUFFLED sequence

Some positions have **negative deltas** — removing the token IMPROVES L2:

| Position | Token | Delta | Meaning |
|---|---|---|---|
| 10 | 'Mary' | **-8.53** | Actively pushing embedding AWAY from target |
| 2 | ' as' | -1.83 | Wrong position |
| 16 | ' go' | -0.60 | Wrong position |

**Negative delta = the token at this position is harmful.** It's in the wrong place and its presence there is making the embedding worse. This correctly identifies the most-misplaced tokens.

The position-wise refinement used this signal: swapping the worst offender (position 10, delta -8.5) first gave a 10-unit L2 drop in one step (31.96 → 21.98).

### Implications for the pipeline

Leave-one-out is most valuable as a **diagnostic layer** feeding into targeted search:

1. Mistral gives initial ordering
2. Leave-one-out identifies the 3–5 most suspicious positions (negative or near-zero delta)
3. Targeted search on just those positions (try all bag tokens there)
4. Swap refinement polishes the result

For **noisy bags** (where some words are wrong), leave-one-out has an additional use: wrong words that don't belong should show strongly negative deltas regardless of position, enabling intruder detection before ordering.

---

## Complete Pipeline Summary

| Step | Method | Input | Output | Cost |
|---|---|---|---|---|
| 1. Bag recovery | L1 per-position decomposition | Pooled L12 embedding + white-box L1 access | 100% bag of words | 1 forward pass |
| 2. Initial ordering | Ask Mistral 7B (or any LLM) | Shuffled bag of words | Near-correct ordering | 1 LLM call (~2s) |
| 3. Diagnostics | Leave-one-out | Candidate ordering + target embedding | Per-position confidence map | N forward passes |
| 4. Targeted search | Try bag tokens at suspicious positions | Confidence map + bag | Improved ordering | ~100 forward passes |
| 5. Polish | Pairwise swap refinement | Current best ordering | Final ordering | ~N² forward passes |

**Results on test sentences:**
- Short (6 tokens): perfect recovery (L2=0) at Step 2 + Step 5
- Medium (12–13 tokens): L2 ≈ 3–6 after Step 5 (94–97% of way from random to perfect)
- Long (23 tokens): L2 ≈ 3.5 after Step 5 (94% of way from random to perfect)

---

## What Remains Open

1. **Full combined pipeline** (Mistral → leave-one-out → targeted search → swap) not yet implemented as a single end-to-end system. Individual components tested separately.
2. **Longer sentences** (Gettysburg-length, 62 tokens) untested with LLM-assisted ordering.
3. **Noisy bag** scenario: when L1 per-position isn't available and the bag comes from the bridge (~33% accuracy), leave-one-out intruder detection is untested.
4. **Larger LLMs** for step 2: GPT-4 / Claude-class models would likely produce better initial orderings, especially for non-famous text.
5. **The production threat model** (pooled-only, no L1 access) remains the hardest case.

---

## Reproduction

```bash
cd experiments/contextual
python3 beam_width_sweep.py        # beam width sweep on short sentences
python3 mary_full_pipeline.py      # 23-token Mary pipeline
python3 hybrid_beam.py             # hybrid beam with L2 scoring
python3 llm_plus_refine.py         # Mistral + swap refinement
python3 leave_one_out.py           # leave-one-out diagnostics
```

Requires Ollama with `mistral:7b` for the LLM-assisted ordering step:
```bash
ollama pull mistral:7b
```
