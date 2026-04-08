#!/usr/bin/env python3
"""Decompose the high band of contextual hidden states against the STATIC vocab.

If the high-precision residual preserves the original token identity, it might
align with the static (wte.weight) embeddings directly — bypassing the need for
per-layer contextual vocabularies entirely.
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

    static_vocab = model.wte.weight.detach().cpu().numpy()
    print(f"Static vocab: {static_vocab.shape}, avg norm {np.mean(np.linalg.norm(static_vocab, axis=1)):.2f}")

    tokens = tokenizer.encode(args.text)
    unique = set(tokens)
    n, u = len(tokens), len(unique)
    print(f"Text: {args.text!r}  ({n} tokens, {u} unique)")

    # Forward pass once, get all hiddens
    with torch.no_grad():
        out = model(torch.tensor([tokens]).to(args.device))
    all_hs = out.hidden_states

    refs = [
        "the quick brown fox jumps over the lazy dog",
        "she went to the store to buy some food",
        "he ran down the long road to the old house",
        "they played in the park with the children",
    ]
    ref_toks = [tokenizer.encode(r) for r in refs]

    # Per-layer biases: high-band(forward(ref)) - static(ref), per token
    # Also per-layer biases: full(forward(ref)) - static(ref), per token
    print("Computing per-layer high→static and full→static biases from refs...")
    biases_high = {}
    biases_full = {}
    for L in layers_to_test:
        hb, fb = [], []
        for rt in ref_toks:
            with torch.no_grad():
                rh = model(torch.tensor([rt]).to(args.device)).hidden_states[L].squeeze(0).cpu().numpy()
            rh_high = rh - int8_quantize(rh)
            static_sum = static_vocab[rt].sum(axis=0)
            hb.append((rh_high.sum(axis=0) - static_sum) / len(rt))
            fb.append((rh.sum(axis=0) - static_sum) / len(rt))
        biases_high[L] = np.mean(hb, axis=0)
        biases_full[L] = np.mean(fb, axis=0)

    print(f"\n{'L':>3s}  {'HiBiasN':>9s}  {'FullBiasN':>10s}  {'Hi+deb':>8s}  {'Full+deb':>9s}  tokens (high+deb vs static)")
    print("-" * 110)

    for L in layers_to_test:
        hs = all_hs[L].squeeze(0).cpu().numpy()
        hs_low = int8_quantize(hs)
        hs_high = hs - hs_low

        high_sum = hs_high.sum(axis=0)
        full_sum = hs.sum(axis=0)

        # Debias with N as multiplier
        high_d = high_sum - n * biases_high[L]
        full_d = full_sum - n * biases_full[L]

        rec_high = decompose_greedy(high_d, static_vocab, n + 10)
        h_high = len(unique & set(rec_high))
        toks_high = [tokenizer.decode([r]) for r in rec_high[:8]]

        rec_full = decompose_greedy(full_d, static_vocab, n + 10)
        h_full = len(unique & set(rec_full))

        bhn = np.linalg.norm(biases_high[L])
        bfn = np.linalg.norm(biases_full[L])
        print(f"  L{L:>2d}  {bhn:>9.2f}  {bfn:>10.1f}  {h_high:>4d}/{u}    {h_full:>4d}/{u}     {toks_high}")


if __name__ == "__main__":
    main()
