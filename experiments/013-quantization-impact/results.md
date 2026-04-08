# Experiment 013: Quantization Impact on Decomposition

**Date**: 2026-04-03 (original run), 2026-04-07 (revised interpretation)
**Code**: `experiments/contextual/quantization_test.py`
**Original output**: `output.txt`

---

## REVISED INTERPRETATION (2026-04-07)

This experiment has two halves with different statuses under the sink discovery (see 010, 011 revised notes):

**Static half: UNAFFECTED.**

| Bits | Recovery |
|---|---|
| 32 | 18/21 |
| 16 | 18/21 |
| 8 | 18/21 |
| 4 | 19/21 |

Static decomposition uses the same quantized values for both target and vocabulary, so quantization errors cancel. This result stands and is consistent with the dimensionality-threshold story (Section 3.3 of the paper).

**Contextual half: AFFECTED.**

The original output reported:

| Bits | Recovery | Signal Norm | Signal % |
|---|---|---|---|
| 32 | 6/21 | 554.0 | 0.85% |
| 16 | 4/21 | 554.0 | 0.85% |
| 8 | 0/21 | 555.8 | 0.85% |
| 4 | 0/21 | 587.4 | 0.90% |

Two characterizations need correcting:

1. **"Signal % = 0.85%."** This is the ratio of post-bias-subtraction residual to total hidden state magnitude — but the "total" was sink-dominated. With the corrected understanding, the per-position content is ~50–70 norm and not "0.85%" of anything in particular. The 0.85% was sink:non-sink, not signal:total.

2. **"Quantization destroys the contextual signal."** The empirical observation (f32 → 6/21, int8 → 0/21) is real, but the explanation needs updating. Under the original model the story was "the token signal lives in the low-order bits, exactly where quantization removes precision." Under the corrected model, the relevant claim is "the per-layer additive corrections (the strata, see paper Section 6.5) are small directional differences that quantization disrupts, and the bias-subtraction step is sensitive to these differences."

The headline conclusion **"quantized models may be more resistant to contextual embedding inversion"** is plausibly still true but needs to be re-tested against sink-skip + N-debias decomposition rather than against the contaminated bias model used here. A re-run is appended at the end of this file.

The original output is preserved in `output.txt`.

---

## Re-run with Sink-Skip + N-Debias (2026-04-07)

Quantization sweep using the corrected decomposition method (sink-skip, N-debias, `[<|endoftext|>, token]` vocab construction at position 1, decomposition at L1). Same Mary-had-a-little-lamb 23-token text, 21 unique trailing tokens after EOT prepend. Output captured in `rerun_2026-04-07.txt`.

### Static decomposition (control — same as original)

| Bits | Recovery |
|---|---|
| 32 | 18/21 |
| 16 | 18/21 |
| 8 | 18/21 |
| 4 | 19/21 |

Unchanged. Static decomposition is unaffected by quantization because both target and vocabulary are quantized identically and the errors cancel.

### Contextual decomposition

| Bits | Old (L6, contaminated bias, norm-ratio N) | New (L1, sink-skip, N-debias) |
|---|---|---|
| 32 | 6/21 | **13/21** |
| 16 | 4/21 | **13/21** |
| 8 | 0/21 | **13/21** |
| 4 | 0/21 | 4/21 |

The original headline finding "**quantization destroys the contextual signal**" is essentially **inverted** under the corrected method:

- The f32 baseline jumps from 6/21 to 13/21 (more than doubled) — the original method was leaving most of the signal on the table
- f16 and int8 are now indistinguishable from f32 — no degradation at all from precision reduction down to 8 bits
- Only int4 substantially degrades, dropping to 4/21

The earlier story ("the token signal lives in the low-order bits, exactly where quantization removes precision") was describing a fragile bias-subtraction working against a sink-contaminated representation. The signal it was tracking — the residual after subtracting a near-2700-norm sink-driven "bias" from a near-2700-norm sink-dominated target — was indeed precision-sensitive, because both sides of that subtraction were huge and nearly equal. The corrected method subtracts a 16-norm bias from a ~1180-norm target (a 70× ratio, not 1.001×), and that subtraction is robust to int8 precision loss.

The "quantized models are more resistant to contextual embedding inversion" claim is now substantially weaker. It survives only at int4, where the per-layer additive structure is actually disrupted by the precision loss (and the bias norm itself shifts from 16.39 to 17.39, indicating the quantization is now perturbing the per-token bias direction in a way it does not at int8 or above).

### What this confirms

1. **The "0.85% signal" framing was sink:non-sink, not signal:total.** The corrected signal-to-target ratio is much closer to 1:70, not 1:200.
2. **int8 inference does not protect against sink-skip embedding inversion.** Defenders relying on quantization as a privacy mitigation should re-evaluate.
3. **int4 still degrades substantially.** Aggressive quantization (4 bits or below) does provide meaningful resistance, but this is a less practical operating point for production inference.
4. **The decomposition method matters more than the precision.** Switching from contaminated-bias to sink-skip more than doubled f32 recovery — far larger than any precision-induced change.

### Reproduction

```bash
cd experiments/contextual
python3 rerun_013_quantization.py
```
