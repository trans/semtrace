#!/usr/bin/env python3
"""
Quantization Impact Test

Tests whether quantization destroys the token signal in contextual
embeddings after bias subtraction. Same model (GPT-2 Small), same
text, same algorithm — only the precision of the hidden states changes.

Usage:
  python3 quantization_test.py [--text "the cat sat on the mat"]
"""

import argparse
import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer


def decompose_greedy(target, embeddings, max_steps):
    residual = target.copy()
    recovered = []
    prev_norm = float("inf")
    for _ in range(max_steps):
        r_norm = np.linalg.norm(residual)
        if r_norm < 0.001 or r_norm > prev_norm:
            break
        prev_norm = r_norm
        norms = np.linalg.norm(embeddings, axis=1)
        norms[norms < 1e-10] = 1.0
        sims = embeddings @ residual / (norms * r_norm)
        best = np.argmax(sims)
        recovered.append(int(best))
        residual = residual - embeddings[best]
    return recovered


def simulate_quantize(arr, bits):
    """Simulate quantization by reducing precision."""
    if bits == 32:
        return arr
    elif bits == 16:
        return arr.astype(np.float16).astype(np.float32)
    elif bits == 8:
        # Per-channel int8 simulation
        scale = np.abs(arr).max(axis=-1, keepdims=True)
        scale[scale < 1e-10] = 1.0
        quantized = np.round(arr / scale * 127).clip(-128, 127).astype(np.int8)
        return quantized.astype(np.float32) * scale / 127
    elif bits == 4:
        # Per-channel int4 simulation
        scale = np.abs(arr).max(axis=-1, keepdims=True)
        scale[scale < 1e-10] = 1.0
        quantized = np.round(arr / scale * 7).clip(-8, 7).astype(np.int8)
        return quantized.astype(np.float32) * scale / 7
    else:
        raise ValueError(f"Unsupported bits: {bits}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="the cat sat on the mat")
    args = parser.parse_args()

    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True)
    model.eval()

    wte = model.wte.weight.detach().numpy()
    vocab_size, dims = wte.shape

    tokens = tokenizer.encode(args.text)
    token_strs = [tokenizer.decode([t]) for t in tokens]
    unique = set(tokens)
    n, u = len(tokens), len(unique)

    print(f"Text: {args.text!r}")
    print(f"Tokens ({n}): {token_strs}, Unique: {u}")

    # Forward pass at full precision
    with torch.no_grad():
        outputs = model(torch.tensor([tokens]))

    # Get L6 hidden states (where we know decomposition partially works)
    h6 = outputs.hidden_states[6].squeeze(0).numpy()

    # Build contextual vocab at L6 (full precision)
    print("\nBuilding contextual vocab at L6 (f32)...")
    ctx_vocab_f32 = np.load("ctx_vocab_L6.npy")
    ctx_mean = ctx_vocab_f32.mean(axis=0)

    # Compute bias from reference sentences
    refs = [
        "the quick brown fox jumps over the lazy dog",
        "she went to the store to buy some food",
        "he ran down the long road to the old house",
        "they played in the park with the children",
    ]
    biases = []
    for ref in refs:
        rtoks = tokenizer.encode(ref)
        with torch.no_grad():
            out = model(torch.tensor([rtoks]))
        rh6 = out.hidden_states[6].squeeze(0).numpy()
        rh6_c = rh6 - ctx_mean
        ctx_centered_ref = ctx_vocab_f32[rtoks] - ctx_mean
        biases.append((rh6_c.sum(axis=0) - ctx_centered_ref.sum(axis=0)) / len(rtoks))
    avg_bias = np.mean(biases, axis=0)
    bias_norm = np.linalg.norm(avg_bias)

    print(f"Bias norm: {bias_norm:.1f}")

    # ================================================================
    # Part 1: Static decomposition at different quantization levels
    # ================================================================
    print(f"\n{'='*60}")
    print("PART 1: Static decomposition (sum of static embeddings)")
    print(f"  Target and vocab quantized to same precision")
    print(f"  {'Bits':>6s}  {'Recovery':>10s}  {'Max Error':>10s}")
    print(f"  {'-'*30}")

    static_sum_f32 = wte[tokens].sum(axis=0)

    for bits in [32, 16, 8, 4]:
        wte_q = simulate_quantize(wte, bits)
        target_q = wte_q[tokens].sum(axis=0)
        rec = decompose_greedy(target_q, wte_q, n + 10)
        hits = len(unique & set(rec))
        max_err = np.max(np.abs(wte_q - wte))
        print(f"  {bits:>4d}b  {hits:>8d}/{u}  {max_err:>10.4f}")

    # ================================================================
    # Part 2: Contextual decomposition after bias subtraction
    # ================================================================
    print(f"\n{'='*60}")
    print("PART 2: Contextual decomposition (bias-subtracted hidden states)")
    print(f"  Hidden states quantized, then bias subtracted")
    print(f"  {'Bits':>6s}  {'Recovery':>10s}  {'Signal Norm':>12s}  {'Tokens'}")
    print(f"  {'-'*55}")

    h6_centered = h6 - ctx_mean
    sentence_sum = h6_centered.sum(axis=0)
    est_n = np.linalg.norm(sentence_sum) / bias_norm

    for bits in [32, 16, 8, 4]:
        # Quantize the hidden states
        h6_q = simulate_quantize(h6, bits)
        h6_q_centered = h6_q - ctx_mean
        sentence_sum_q = h6_q_centered.sum(axis=0)

        # Estimate N and subtract bias
        est_n_q = np.linalg.norm(sentence_sum_q) / bias_norm
        fixed_q = sentence_sum_q - est_n_q * avg_bias
        signal_norm = np.linalg.norm(fixed_q)

        # Also quantize the contextual vocab
        ctx_vocab_q = simulate_quantize(ctx_vocab_f32, bits)
        ctx_centered_q = ctx_vocab_q - simulate_quantize(ctx_mean.reshape(1, -1), bits).squeeze()

        rec = decompose_greedy(fixed_q, ctx_centered_q, n + 10)
        hits = len(unique & set(rec))
        toks = [tokenizer.decode([r]) for r in rec[:6]]
        print(f"  {bits:>4d}b  {hits:>8d}/{u}  {signal_norm:>12.1f}  {toks}")

    # ================================================================
    # Part 3: What happens to the signal energy at each precision?
    # ================================================================
    print(f"\n{'='*60}")
    print("PART 3: Signal-to-noise ratio at each precision")
    print(f"  After bias subtraction, how much token signal remains?")
    print(f"  {'Bits':>6s}  {'Total Norm':>11s}  {'Bias Norm':>10s}  {'Signal Norm':>12s}  {'Signal %':>9s}")
    print(f"  {'-'*55}")

    for bits in [32, 16, 8, 4]:
        h6_q = simulate_quantize(h6, bits)
        h6_q_centered = h6_q - ctx_mean
        ss_q = h6_q_centered.sum(axis=0)
        bias_component = est_n * avg_bias
        signal = ss_q - bias_component
        total = np.linalg.norm(ss_q)
        sig = np.linalg.norm(signal)
        pct = 100 * sig / total if total > 0 else 0
        print(f"  {bits:>4d}b  {total:>11.1f}  {np.linalg.norm(bias_component):>10.1f}  {sig:>12.1f}  {pct:>8.1f}%")


if __name__ == "__main__":
    main()
