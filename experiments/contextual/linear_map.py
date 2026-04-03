#!/usr/bin/env python3
"""
Linear Mapping: Contextual → Static Embedding Space

Learns a linear projection W such that: static ≈ W @ contextual
Then applies it to a sentence embedding and attempts greedy decomposition.

Usage:
  python3 linear_map.py --text "the cat sat on the mat" [--layer 6]
"""

import argparse
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer
import torch


def decompose_greedy(target, embeddings, max_steps, metric="cosine"):
    residual = target.copy()
    recovered = []
    prev_norm = float("inf")
    for step in range(max_steps):
        r_norm = np.linalg.norm(residual)
        if r_norm < 0.001 or r_norm > prev_norm:
            break
        prev_norm = r_norm
        if metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1)
            norms[norms < 1e-10] = 1.0
            sims = embeddings @ residual / (norms * r_norm)
            best = np.argmax(sims)
        elif metric == "ip":
            best = np.argmax(embeddings @ residual)
        elif metric == "l2":
            best = np.argmin(np.sum((embeddings - residual) ** 2, axis=1))
        recovered.append(int(best))
        residual = residual - embeddings[best]
    return recovered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="the cat sat on the mat")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print(f"Loading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True).to(args.device)
    model.eval()

    # Load pre-built vocabs
    static = np.load("static_vocab.npy")
    ctx = np.load(f"ctx_vocab_L{args.layer}.npy")
    vocab_size, dims = static.shape
    print(f"Static: {static.shape}, Contextual L{args.layer}: {ctx.shape}")

    # ================================================================
    # Learn linear mapping: static = W @ contextual
    # Using least squares: W = static.T @ ctx @ (ctx.T @ ctx)^-1
    # Or equivalently: W = np.linalg.lstsq(ctx, static)
    # ================================================================
    print(f"\nFitting linear mapping (contextual L{args.layer} → static)...")
    print(f"  Training on all {vocab_size} token pairs...")

    # lstsq: find W such that ctx @ W ≈ static  (W is dims x dims)
    W, residuals, rank, sv = np.linalg.lstsq(ctx, static, rcond=None)
    print(f"  W shape: {W.shape}, rank: {rank}")

    # Check fit quality on training data
    predicted = ctx @ W
    errors = np.linalg.norm(predicted - static, axis=1)
    print(f"  Reconstruction error: mean={errors.mean():.4f}, median={np.median(errors):.4f}, max={errors.max():.4f}")

    # Check: does the mapping preserve token identity?
    # For each token, is the nearest static vector to the mapped contextual the same token?
    print(f"  Checking identity preservation (mapped contextual → nearest static)...")
    correct = 0
    for i in range(min(vocab_size, 10000)):  # check first 10k
        mapped = ctx[i] @ W
        dists = np.sum((static - mapped) ** 2, axis=1)
        nearest = np.argmin(dists)
        if nearest == i:
            correct += 1
    pct = 100 * correct / min(vocab_size, 10000)
    print(f"  Identity preserved: {correct}/{min(vocab_size, 10000)} ({pct:.1f}%)")

    # ================================================================
    # Now: forward pass the sentence, get contextual embedding, map it,
    # then decompose against static vocab
    # ================================================================
    tokens = tokenizer.encode(args.text)
    token_strs = [tokenizer.decode([t]) for t in tokens]
    unique_tokens = set(tokens)
    n, u = len(tokens), len(unique_tokens)
    max_steps = n + 10

    print(f"\nText: {args.text!r}")
    print(f"Tokens ({n}): {token_strs}, Unique: {u}")

    with torch.no_grad():
        outputs = model(torch.tensor([tokens]).to(args.device))
    h = outputs.hidden_states[args.layer].squeeze(0).cpu().numpy()

    # Test multiple approaches
    print(f"\n{'='*60}")
    print(f"DECOMPOSITION RESULTS (Layer {args.layer})")

    targets = {
        "mean-pool (raw)": h.mean(axis=0),
        "last-token (raw)": h[-1],
        "mean-pool (mapped)": h.mean(axis=0) @ W,
        "last-token (mapped)": h[-1] @ W,
    }

    # Also: map each per-position hidden state, then sum the mapped versions
    mapped_per_pos = np.array([h[i] @ W for i in range(n)])
    targets["sum of mapped positions"] = mapped_per_pos.sum(axis=0)

    # Control: static bag-of-words
    targets["static BoW (control)"] = static[tokens].sum(axis=0)

    # Control: contextual BoW mapped
    ctx_bow = ctx[tokens].sum(axis=0)
    targets["ctx BoW mapped"] = ctx_bow @ W

    for name, target in targets.items():
        all_recovered = set()
        best_hits = 0
        best_metric = ""
        for metric in ["cosine", "ip", "l2"]:
            rec = decompose_greedy(target, static, max_steps, metric)
            hits = len(unique_tokens & set(rec))
            all_recovered.update(rec)
            if hits > best_hits:
                best_hits = hits
                best_metric = metric
                best_toks = [tokenizer.decode([r]) for r in rec[:8]]
        union_hits = len(unique_tokens & all_recovered)
        print(f"\n  {name}:")
        print(f"    best: {best_metric} {best_hits}/{u} ({100*best_hits/u:.0f}%)  union: {union_hits}/{u} ({100*union_hits/u:.0f}%)")
        print(f"    tokens: {best_toks}")


if __name__ == "__main__":
    main()
