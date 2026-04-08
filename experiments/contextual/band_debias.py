#!/usr/bin/env python3
"""Bit-band split + per-layer bias subtraction.

Hypothesis: the int8-quantized "low band" carries the static-token signal,
while the small-magnitude "high band" carries non-decomposable contextual
attention contributions. If true, then debiasing the LOW band should jump
decomposition substantially, while debiasing the HIGH band should not.

Test at L11 of GPT-2 Small (last pre-ln_f layer).

Usage:
  python3 band_debias.py [--text "the cat sat on the mat"] [--layer 11]
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
    """Per-vector symmetric int8 quant. Returns the dequantized approximation."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        scale = np.max(np.abs(x)) / 127.0
        if scale == 0:
            return x.copy()
        q = np.round(x / scale).clip(-127, 127)
        return q * scale
    # 2D: per-row
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        out[i] = int8_quantize(x[i])
    return out


def compute_bias(model, tokenizer, device, layer, ctx_vocab, refs):
    biases = []
    for ref in refs:
        rtoks = tokenizer.encode(ref)
        hs = get_layer_hiddens(model, rtoks, layer, device)
        sentence_sum = hs.sum(axis=0)
        token_sum = ctx_vocab[rtoks].sum(axis=0)
        biases.append((sentence_sum - token_sum) / len(rtoks))
    return np.mean(biases, axis=0)


def try_decompose(label, target, vocab, unique, n, tokenizer):
    rec = decompose_greedy(target, vocab, n + 10)
    hits = len(unique & set(rec))
    toks = [tokenizer.decode([r]) for r in rec[:8]]
    print(f"    {label:36s} norm={np.linalg.norm(target):>8.1f}  {hits}/{len(unique)}  {toks}")
    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="the cat sat on the mat")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True).to(args.device)
    model.eval()

    tokens = tokenizer.encode(args.text)
    token_strs = [tokenizer.decode([t]) for t in tokens]
    unique = set(tokens)
    n, u = len(tokens), len(unique)
    print(f"Text: {args.text!r}")
    print(f"Tokens ({n}): {token_strs}, Unique: {u}")
    print(f"Target layer: L{args.layer}")

    # Build contextual vocab at the target layer
    print(f"\nBuilding contextual vocab at L{args.layer}...")
    ctx_vocab = build_layer_vocab(model, args.layer, args.device)
    print(f"  shape={ctx_vocab.shape}, avg norm={np.mean(np.linalg.norm(ctx_vocab, axis=1)):.1f}")

    # Compute bias from reference sentences
    refs = [
        "the quick brown fox jumps over the lazy dog",
        "she went to the store to buy some food",
        "he ran down the long road to the old house",
        "they played in the park with the children",
    ]
    print(f"\nComputing per-layer bias from {len(refs)} reference sentences...")
    bias = compute_bias(model, tokenizer, args.device, args.layer, ctx_vocab, refs)
    bias_norm = np.linalg.norm(bias)
    print(f"  bias norm: {bias_norm:.1f}")

    # Get target sentence hidden states
    hs = get_layer_hiddens(model, tokens, args.layer, args.device)
    sentence_sum = hs.sum(axis=0)
    print(f"  sentence_sum norm: {np.linalg.norm(sentence_sum):.1f}")

    # Bit band split: low = int8 quantized approx, high = residual
    print(f"\nBit-band splitting (per-row int8 quant)...")
    low_hs = int8_quantize(hs)
    high_hs = hs - low_hs
    low_sum = low_hs.sum(axis=0)
    high_sum = high_hs.sum(axis=0)
    print(f"  low_sum norm:  {np.linalg.norm(low_sum):.2f}")
    print(f"  high_sum norm: {np.linalg.norm(high_sum):.4f}")

    # Also quantize the vocab so we compare like-with-like
    print(f"\nQuantizing contextual vocab...")
    low_vocab = int8_quantize(ctx_vocab)
    high_vocab = ctx_vocab - low_vocab
    print(f"  low_vocab avg norm:  {np.mean(np.linalg.norm(low_vocab, axis=1)):.1f}")
    print(f"  high_vocab avg norm: {np.mean(np.linalg.norm(high_vocab, axis=1)):.4f}")

    # Bias for the low band (computed against low vocab)
    print(f"\nComputing band-specific biases...")
    low_bias = compute_bias(model, tokenizer, args.device, args.layer, low_vocab, refs)
    # For the low band we need low-only sentence hiddens; recompute properly:
    # Actually low_bias above used full hs minus low_vocab tokens — not right.
    # Recompute using quantized refs:
    low_biases = []
    for ref in refs:
        rtoks = tokenizer.encode(ref)
        rhs = get_layer_hiddens(model, rtoks, args.layer, args.device)
        rhs_low = int8_quantize(rhs)
        low_biases.append((rhs_low.sum(axis=0) - low_vocab[rtoks].sum(axis=0)) / len(rtoks))
    low_bias = np.mean(low_biases, axis=0)
    print(f"  low_bias norm: {np.linalg.norm(low_bias):.1f}")

    # ================================================================
    print(f"\n{'='*70}")
    print(f"DECOMPOSITION RESULTS @ L{args.layer}")
    print(f"{'='*70}")

    print(f"\n  Against FULL contextual vocab:")
    try_decompose("full, no debias", sentence_sum, ctx_vocab, unique, n, tokenizer)

    est = np.linalg.norm(sentence_sum) / bias_norm
    debiased = sentence_sum - est * bias
    try_decompose(f"full, debiased (k={est:.1f})", debiased, ctx_vocab, unique, n, tokenizer)

    print(f"\n  Against LOW band vocab (int8 approx):")
    try_decompose("low, no debias", low_sum, low_vocab, unique, n, tokenizer)

    lb_norm = np.linalg.norm(low_bias)
    est_l = np.linalg.norm(low_sum) / lb_norm
    low_debiased = low_sum - est_l * low_bias
    try_decompose(f"low, debiased (k={est_l:.1f})", low_debiased, low_vocab, unique, n, tokenizer)

    print(f"\n  Against HIGH band vocab (f32 - int8 residual):")
    try_decompose("high, no debias", high_sum, high_vocab, unique, n, tokenizer)

    # high band bias
    high_biases = []
    for ref in refs:
        rtoks = tokenizer.encode(ref)
        rhs = get_layer_hiddens(model, rtoks, args.layer, args.device)
        rhs_high = rhs - int8_quantize(rhs)
        high_biases.append((rhs_high.sum(axis=0) - high_vocab[rtoks].sum(axis=0)) / len(rtoks))
    high_bias = np.mean(high_biases, axis=0)
    print(f"  high_bias norm: {np.linalg.norm(high_bias):.4f}")
    hb_norm = np.linalg.norm(high_bias)
    if hb_norm > 1e-10:
        est_h = np.linalg.norm(high_sum) / hb_norm
        try_decompose(f"high, debiased (k={est_h:.1f})", high_sum - est_h * high_bias, high_vocab, unique, n, tokenizer)
    try_decompose(f"high, debiased N=6", high_sum - n * high_bias, high_vocab, unique, n, tokenizer)

    # Try with N as multiplier instead of norm-ratio
    print(f"\n  Using N={n} as bias multiplier (instead of norm-ratio):")
    try_decompose(f"full, debiased N={n}", sentence_sum - n * bias, ctx_vocab, unique, n, tokenizer)
    try_decompose(f"low, debiased N={n}", low_sum - n * low_bias, low_vocab, unique, n, tokenizer)

    # Cross-check: low band against full vocab (does the cleaned-up signal still align?)
    print(f"\n  Cross: LOW sentence vs FULL vocab:")
    try_decompose("low_sum vs full vocab", low_sum, ctx_vocab, unique, n, tokenizer)
    try_decompose("low debiased vs full vocab", low_sum - est_l * low_bias, ctx_vocab, unique, n, tokenizer)


if __name__ == "__main__":
    main()
