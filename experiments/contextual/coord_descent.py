#!/usr/bin/env python3
"""
Coordinate Descent Token Recovery

Iteratively optimizes each token position: hold all other positions fixed,
find the token that minimizes total distance to target. Repeat until
convergence. Initialized from greedy decomposition.

Usage:
  python3 coord_descent.py --text "the cat sat on the mat" [--mode static|contextual]
"""

import argparse
import os
import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer


def nearest_k(target, vocab, k=1):
    t_norm = np.linalg.norm(target)
    if t_norm < 1e-10:
        return np.array([0])
    norms = np.linalg.norm(vocab, axis=1)
    norms[norms < 1e-10] = 1.0
    sims = vocab @ target / (norms * t_norm)
    return np.argsort(-sims)[:k]


def greedy_decompose(target, vocab, n_tokens):
    residual = target.copy()
    tokens = []
    prev_norm = float("inf")
    for _ in range(n_tokens + 10):
        r_norm = np.linalg.norm(residual)
        if r_norm < 0.001 or r_norm > prev_norm:
            break
        prev_norm = r_norm
        best = nearest_k(residual, vocab, k=1)[0]
        tokens.append(int(best))
        residual = residual - vocab[best]
    return tokens


def coord_descent(target, vocab, n_tokens, max_iters=20):
    # Initialize with greedy
    current = greedy_decompose(target, vocab, n_tokens)

    # Pad if greedy stopped early
    while len(current) < n_tokens:
        ideal = target - vocab[current].sum(axis=0)
        best = nearest_k(ideal, vocab, k=1)[0]
        current.append(int(best))

    # Trim if greedy overshot
    current = current[:n_tokens]

    best_dist = np.linalg.norm(target - vocab[current].sum(axis=0))

    for iteration in range(max_iters):
        improved = False
        for pos in range(n_tokens):
            others = [current[j] for j in range(n_tokens) if j != pos]
            others_sum = vocab[others].sum(axis=0)
            ideal = target - others_sum
            best_for_pos = nearest_k(ideal, vocab, k=1)[0]

            if best_for_pos != current[pos]:
                new_tokens = list(current)
                new_tokens[pos] = int(best_for_pos)
                new_dist = np.linalg.norm(target - vocab[new_tokens].sum(axis=0))
                if new_dist < best_dist:
                    current[pos] = int(best_for_pos)
                    best_dist = new_dist
                    improved = True

        if not improved:
            break

    return current, best_dist, iteration + 1


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="the cat sat on the mat")
    parser.add_argument("--mode", default="both", choices=["static", "contextual", "both"])
    parser.add_argument("--file", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    text = open(args.file).read().strip() if args.file else args.text

    print("Loading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True).to(args.device)
    model.eval()

    wte = model.wte.weight.detach().cpu().numpy()
    vocab_size, dims = wte.shape

    tokens = tokenizer.encode(text)
    token_strs = [tokenizer.decode([t]) for t in tokens]
    unique = set(tokens)
    n, u = len(tokens), len(unique)

    print(f"Text: {text[:60]!r}{'...' if len(text) > 60 else ''}")
    print(f"Tokens: {n}, Unique: {u}")

    if args.mode in ("static", "both"):
        print(f"\n{'='*60}")
        print("STATIC: Coordinate Descent")
        target = wte[tokens].sum(axis=0)

        greedy = greedy_decompose(target, wte, n)
        greedy_hits = len(unique & set(greedy))

        cd_result, cd_dist, cd_iters = coord_descent(target, wte, len(greedy), max_iters=15)
        cd_hits = len(unique & set(cd_result))

        print(f"  Greedy:      {greedy_hits}/{u} ({100*greedy_hits/u:.1f}%)")
        print(f"  Coord Desc:  {cd_hits}/{u} ({100*cd_hits/u:.1f}%), {cd_iters} iters, dist={cd_dist:.4f}")

    if args.mode in ("contextual", "both"):
        ctx_L6 = np.load("ctx_vocab_L6.npy")
        ctx_mean = ctx_L6.mean(axis=0)
        ctx_centered = ctx_L6 - ctx_mean

        # Compute bias
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
                out = model(torch.tensor([rtoks]).to(args.device))
            h6 = out.hidden_states[6].squeeze(0).cpu().numpy()
            h6_c = h6 - ctx_mean
            biases.append((h6_c.sum(axis=0) - ctx_centered[rtoks].sum(axis=0)) / len(rtoks))
        avg_bias = np.mean(biases, axis=0)

        # Forward pass on target
        with torch.no_grad():
            out = model(torch.tensor([tokens]).to(args.device))
        h6 = out.hidden_states[6].squeeze(0).cpu().numpy()
        h6_c = h6 - ctx_mean
        sentence_sum = h6_c.sum(axis=0)

        est_n = np.linalg.norm(sentence_sum) / np.linalg.norm(avg_bias)
        fixed = sentence_sum - est_n * avg_bias

        print(f"\n{'='*60}")
        print(f"CONTEXTUAL L6: Bias Subtraction + Coordinate Descent (est N={est_n:.1f})")

        greedy = greedy_decompose(fixed, ctx_centered, n)
        greedy_hits = len(unique & set(greedy))

        cd_result, cd_dist, cd_iters = coord_descent(fixed, ctx_centered, n, max_iters=15)
        cd_hits = len(unique & set(cd_result))
        cd_toks = [tokenizer.decode([t]) for t in cd_result]

        print(f"  Greedy:      {greedy_hits}/{u} ({100*greedy_hits/u:.1f}%)")
        print(f"  Coord Desc:  {cd_hits}/{u} ({100*cd_hits/u:.1f}%), {cd_iters} iters, dist={cd_dist:.2f}")
        print(f"  Tokens: {cd_toks}")


if __name__ == "__main__":
    main()
