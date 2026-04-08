#!/usr/bin/env python3
"""Layer sweep: N-corrected debias + bit-band decomposition at every other layer.

For each layer L in {1, 3, 5, 7, 9, 11}:
  1. Build contextual vocab at L
  2. Compute per-token bias from reference sentences
  3. Decompose target sentence against:
     - full vocab, debiased with N as multiplier
     - low band (int8 quant) vocab, debiased
     - high band (residual) vocab, debiased

Usage:
  python3 layer_sweep.py [--text "the cat sat on the mat"]
"""
import argparse
import gc
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
        best = int(np.argmax(sims))
        recovered.append(best)
        residual = residual - embeddings[best]
    return recovered


def get_layer_hiddens(model, tokens, layer, device):
    with torch.no_grad():
        out = model(torch.tensor([tokens]).to(device))
    return out.hidden_states[layer].squeeze(0).cpu().numpy()


def build_layer_vocab(model, layer, device, batch=512):
    vocab_size = model.wte.weight.shape[0]
    chunks = []
    for start in range(0, vocab_size, batch):
        end = min(start + batch, vocab_size)
        ids = torch.arange(start, end).unsqueeze(1).to(device)
        with torch.no_grad():
            out = model(ids)
        chunks.append(out.hidden_states[layer].squeeze(1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def int8_quantize(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        scale = np.max(np.abs(x)) / 127.0
        if scale == 0:
            return x.copy()
        return np.round(x / scale).clip(-127, 127) * scale
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        out[i] = int8_quantize(x[i])
    return out


def hits(target, vocab, unique, n):
    rec = decompose_greedy(target, vocab, n + 10)
    return len(unique & set(rec)), rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="the cat sat on the mat")
    parser.add_argument("--layers", default="1,3,5,7,9,11")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    layers_to_test = [int(x) for x in args.layers.split(",")]

    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True).to(args.device)
    model.eval()

    tokens = tokenizer.encode(args.text)
    unique = set(tokens)
    n, u = len(tokens), len(unique)
    print(f"Text: {args.text!r}  ({n} tokens, {u} unique)")

    refs = [
        "the quick brown fox jumps over the lazy dog",
        "she went to the store to buy some food",
        "he ran down the long road to the old house",
        "they played in the park with the children",
    ]

    # Pre-compute reference token sequences
    ref_toks = [tokenizer.encode(r) for r in refs]

    # Pre-compute the target sentence hiddens at all layers (one forward pass)
    with torch.no_grad():
        all_target = model(torch.tensor([tokens]).to(args.device)).hidden_states
    target_hs = {L: all_target[L].squeeze(0).cpu().numpy() for L in layers_to_test}

    # Pre-compute reference hiddens at all layers (one pass per ref)
    ref_hs = {L: [] for L in layers_to_test}
    for rt in ref_toks:
        with torch.no_grad():
            rh = model(torch.tensor([rt]).to(args.device)).hidden_states
        for L in layers_to_test:
            ref_hs[L].append(rh[L].squeeze(0).cpu().numpy())

    print(f"\n{'L':>3s}  {'BiasN':>8s}  {'FullDeb':>9s}  {'LowDeb':>9s}  {'HighDeb':>9s}  {'tokens (full deb)'}")
    print("-" * 90)

    results = {}
    for L in layers_to_test:
        # Build vocab at this layer
        vocab = build_layer_vocab(model, L, args.device)
        low_vocab = int8_quantize(vocab)
        high_vocab = vocab - low_vocab

        # Compute biases (per-token) for full, low, high
        full_b, low_b, high_b = [], [], []
        for rt, rh in zip(ref_toks, ref_hs[L]):
            full_b.append((rh.sum(axis=0) - vocab[rt].sum(axis=0)) / len(rt))
            rh_low = int8_quantize(rh)
            rh_high = rh - rh_low
            low_b.append((rh_low.sum(axis=0) - low_vocab[rt].sum(axis=0)) / len(rt))
            high_b.append((rh_high.sum(axis=0) - high_vocab[rt].sum(axis=0)) / len(rt))
        full_bias = np.mean(full_b, axis=0)
        low_bias = np.mean(low_b, axis=0)
        high_bias = np.mean(high_b, axis=0)

        # Target sums
        ths = target_hs[L]
        full_sum = ths.sum(axis=0)
        ths_low = int8_quantize(ths)
        low_sum = ths_low.sum(axis=0)
        high_sum = (ths - ths_low).sum(axis=0)

        # Debias with N as multiplier
        full_d = full_sum - n * full_bias
        low_d = low_sum - n * low_bias
        high_d = high_sum - n * high_bias

        h_full, rec_full = hits(full_d, vocab, unique, n)
        h_low, _ = hits(low_d, low_vocab, unique, n)
        h_high, _ = hits(high_d, high_vocab, unique, n)

        toks_full = [tokenizer.decode([r]) for r in rec_full[:8]]
        bn = np.linalg.norm(full_bias)
        print(f"  L{L:>2d}  {bn:>8.1f}  {h_full:>4d}/{u}    {h_low:>4d}/{u}    {h_high:>4d}/{u}    {toks_full}")
        results[L] = (h_full, h_low, h_high)

        del vocab, low_vocab, high_vocab
        gc.collect()

    print(f"\nSummary (hits / {u}):")
    print(f"  {'Layer':>6s}  {'Full+deb':>10s}  {'Low+deb':>10s}  {'High+deb':>10s}")
    for L in layers_to_test:
        f, lo, hi = results[L]
        print(f"  L{L:<5d}  {f:>10d}  {lo:>10d}  {hi:>10d}")


if __name__ == "__main__":
    main()
