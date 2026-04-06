#!/usr/bin/env python3
"""Identity rank vs prediction rank at every transformer layer.

At each layer, measures:
- Identity rank: cosine similarity rank of the correct static token
- Prediction rank: logit rank of the current token after applying
  the output projection (LayerNorm + wte.T)

Usage: python3 identity_vs_prediction.py [--text "the cat sat on the mat"]
"""

import argparse
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="the cat sat on the mat")
    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    wte = model.transformer.wte.weight.detach().numpy()
    ln_f = model.transformer.ln_f
    tokens = tokenizer.encode(args.text)
    token_strs = [tokenizer.decode([t]) for t in tokens]
    n = len(tokens)

    with torch.no_grad():
        out = model(torch.tensor([tokens]), output_hidden_states=True)

    print("=== Identity rank vs Prediction rank at every layer ===")
    print(f"Text: {token_strs}")
    print()
    print(f"{'Layer':>6s}  {'Avg Identity Rank':>18s}  {'Avg Prediction Rank':>20s}")
    print("-" * 50)

    for layer_idx in range(13):
        h = out.hidden_states[layer_idx].squeeze(0).detach()
        h_np = h.numpy()

        # Identity: cosine to own static token
        identity_ranks = []
        for i in range(n):
            hv = h_np[i]
            tv = wte[tokens[i]]
            sim = np.dot(hv, tv) / (np.linalg.norm(hv) * np.linalg.norm(tv))
            all_sims = wte @ hv / (np.linalg.norm(wte, axis=1) * np.linalg.norm(hv))
            identity_ranks.append(int(np.sum(all_sims > sim) + 1))

        # Prediction: apply LayerNorm + wte.T, check self-rank
        h_normed = ln_f(h).detach().numpy()
        logits = h_normed @ wte.T
        pred_ranks = []
        for i in range(n):
            own_logit = logits[i][tokens[i]]
            rank = int(np.sum(logits[i] > own_logit) + 1)
            pred_ranks.append(rank)

        lname = "emb" if layer_idx == 0 else f"L{layer_idx}"
        print(f"{lname:>6s}  {np.mean(identity_ranks):>18.0f}  {np.mean(pred_ranks):>20.0f}")


if __name__ == "__main__":
    main()
