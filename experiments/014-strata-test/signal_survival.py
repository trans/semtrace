#!/usr/bin/env python3
"""Per-position token signal survival across all transformer layers.

Measures cosine similarity and rank of the correct static token at each
layer's hidden state, for each position in the input sentence.

Usage: python3 signal_survival.py [--text "the cat sat on the mat"]
"""

import argparse
import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="the cat sat on the mat")
    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True)
    model.eval()

    wte = model.wte.weight.detach().numpy()
    tokens = tokenizer.encode(args.text)
    token_strs = [tokenizer.decode([t]) for t in tokens]
    n = len(tokens)

    with torch.no_grad():
        out = model(torch.tensor([tokens]))

    print("=== Per-position token signal survival across layers ===")
    print(f"Text: {token_strs}")
    print()

    print(f"{'Layer':>6s}  ", end="")
    for i in range(n):
        print(f"{token_strs[i]:>8s}  ", end="")
    print("  Avg")
    print("-" * (10 + n * 10 + 6))

    for layer_idx in range(13):
        h = out.hidden_states[layer_idx].squeeze(0).numpy()
        sims = []
        for i in range(n):
            hv = h[i]
            tv = wte[tokens[i]]
            sim = np.dot(hv, tv) / (np.linalg.norm(hv) * np.linalg.norm(tv))
            sims.append(sim)
        avg_sim = np.mean(sims)
        lname = "emb" if layer_idx == 0 else f"L{layer_idx}"
        print(f"{lname:>6s}  ", end="")
        for s in sims:
            print(f"{s:>8.4f}  ", end="")
        print(f"  {avg_sim:.4f}")

    print()
    print("=== Rank of correct token (out of 50,257) ===")
    print(f"{'Layer':>6s}  ", end="")
    for i in range(n):
        print(f"{token_strs[i]:>8s}  ", end="")
    print("  Avg")
    print("-" * (10 + n * 10 + 6))

    for layer_idx in range(13):
        h = out.hidden_states[layer_idx].squeeze(0).numpy()
        ranks = []
        for i in range(n):
            hv = h[i]
            tv = wte[tokens[i]]
            sim = np.dot(hv, tv) / (np.linalg.norm(hv) * np.linalg.norm(tv))
            all_sims = wte @ hv / (np.linalg.norm(wte, axis=1) * np.linalg.norm(hv))
            rank = int(np.sum(all_sims > sim) + 1)
            ranks.append(rank)
        avg_rank = np.mean(ranks)
        lname = "emb" if layer_idx == 0 else f"L{layer_idx}"
        print(f"{lname:>6s}  ", end="")
        for r in ranks:
            print(f"{r:>8d}  ", end="")
        print(f"  {avg_rank:.0f}")


if __name__ == "__main__":
    main()
