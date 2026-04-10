# Experiment 015: Attention Sink Discovery and Sink-Skip Method

**Date**: 2026-04-07
**Code**: `experiments/contextual/attention_trace.py`, `experiments/contextual/skip_sink.py`, `experiments/contextual/layer_sweep.py`

---

## Objective

Investigate why contextual decomposition was failing at deep layers by instrumenting the per-position, per-block contributions of GPT-2 Small's transformer blocks. Identify the dominant artifact and develop a correction.

---

## Discovery: The Position-0 Attention Sink

By placing hooks on every block component (ln_1, attention output, MLP output) for a 6-token input ("the cat sat on the mat"), we found that **position 0 is repurposed as an attention sink**:

| Block | pos 0 (`the`) norm | pos 1–5 norm (avg) |
|---|---|---|
| input to L1 | 137 | 57 |
| input to L2 | 645 | 59 |
| input to L3 | 2573 | 67 |
| input to L4 | 2765 | 70 |

Position 0 alone grows from 137 to 2765 — a 20× increase. Every other position stays at ~50–75. The growth is driven by the MLP adding ~500 → ~2300 → ~200 norm of content to position 0 specifically, while adding only ~10–16 to other positions.

The attention pattern confirms: every other position attends heavily to position 0. By block 2, position 5 puts 34% of its attention on position 0, and position 1 puts 94%.

This is the well-documented **attention sink** phenomenon (Xiao et al., 2023), rediscovered empirically in the embedding-inversion context.

## Consequences for Prior Results

Three findings from experiments 010–014 were sink artifacts:

1. **The "near-constant 99.5% bias"** (experiment 011) was the position-0 sink dump, not a per-token bias. Sink consistency (cos 0.9999+ across sentences) reflected the architectural property that every sentence creates the same sink.

2. **The "contextual vocabulary"** built by passing each token individually through the model was sink-corrupted: every entry was at position 0, receiving the full sink treatment.

3. **The "GPT-2 hidden states lack semantic discrimination"** (experiment 010) was sink-driven. With sink removal, GPT-2 has real semantic discrimination (Section confirmed in experiment 016).

## The Sink-Skip Method

Two corrections neutralize the sink:

**Vocabulary correction.** Build the contextual vocabulary by passing `[<|endoftext|>, token]` pairs and reading position 1 (not position 0).

**Target correction.** Exclude position 0 from the target sentence sum. Optionally prepend `<|endoftext|>` to move the original first token off the sink position.

## Results: Per-Layer Decomposition with Sink-Skip + N-Debias

Test sentence: "the cat sat on the mat" (5 trailing tokens after sink-skip, or 6 with EOT prepend).

With EOT prepend, decomposition at every layer L1–L11:

| Layer | Bias norm (clean) | Recovery (N-debias) |
|---|---|---|
| L1 | 16 | 6/6 |
| L3 | 21 | 6/6 |
| L5 | 28 | 6/6 |
| L7 | 37 | 5/6 |
| L9 | 45 | 5/6 |
| L11 | 60 | 4/6 |

Compare to pre-sink bias norms: 116 (L1) → 2266 (L3) → 2678 (L11). The sink inflated the apparent bias by ~50×.

On 20 sentences (132 unique tokens), L1 sink-skip + N-debias gives **117/132 (88.6%)** bag-of-words recovery.

## Reproduction

```bash
cd experiments/contextual
python3 attention_trace.py     # per-position per-block trace
python3 skip_sink.py           # sink-skip + N-debias decomposition
python3 layer_sweep.py         # every-other-layer sweep
```
