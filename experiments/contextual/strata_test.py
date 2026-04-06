#!/usr/bin/env python3
"""
Strata Test: Per-layer bias measurement and multi-layer subtraction.

1. Compute the attention bias at each transformer layer
2. Measure norm, direction, and consistency across sentences
3. Test whether peeling layers sequentially improves token recovery

Usage:
  python3 strata_test.py [--text "the cat sat on the mat"]
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


def get_hidden_states(model, tokens, device):
    """Get hidden states at every layer for a token sequence."""
    with torch.no_grad():
        out = model(torch.tensor([tokens]).to(device))
    return [h.squeeze(0).cpu().numpy() for h in out.hidden_states]


def compute_per_layer_bias(model, tokenizer, device, refs, ctx_vocabs):
    """Compute the per-layer bias: what attention adds at each layer.

    At each layer L, for each reference sentence:
      bias_L = (sentence_hidden_sum_L - sum_of_individual_token_hiddens_L) / N

    Average across reference sentences.
    """
    biases = []

    for layer_idx in range(len(ctx_vocabs)):
        layer_biases = []
        for ref in refs:
            rtoks = tokenizer.encode(ref)
            n = len(rtoks)
            hs = get_hidden_states(model, rtoks, device)

            # Sum of sentence hidden states at this layer
            sentence_sum = hs[layer_idx].sum(axis=0)

            # Sum of individual token embeddings at this layer
            token_sum = ctx_vocabs[layer_idx][rtoks].sum(axis=0)

            # Per-token bias at this layer
            layer_biases.append((sentence_sum - token_sum) / n)

        avg_bias = np.mean(layer_biases, axis=0)
        biases.append(avg_bias)

    return biases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="the cat sat on the mat")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True).to(args.device)
    model.eval()

    vocab_size = model.wte.weight.shape[0]
    num_layers = model.config.n_layer  # 12

    tokens = tokenizer.encode(args.text)
    token_strs = [tokenizer.decode([t]) for t in tokens]
    unique = set(tokens)
    n, u = len(tokens), len(unique)

    print(f"Text: {args.text!r}")
    print(f"Tokens ({n}): {token_strs}, Unique: {u}")
    print(f"Layers: {num_layers}")

    # Reference sentences for bias computation
    refs = [
        "the quick brown fox jumps over the lazy dog",
        "she went to the store to buy some food",
        "he ran down the long road to the old house",
        "they played in the park with the children",
    ]

    # ================================================================
    # Part 1: Build per-layer contextual vocabularies
    # ================================================================
    print(f"\nBuilding per-layer contextual vocabularies...")
    ctx_vocabs = []
    for layer_idx in range(num_layers + 1):  # 0=embed, 1-12=layers
        all_embs = []
        for start in range(0, vocab_size, 512):
            end = min(start + 512, vocab_size)
            ids = torch.arange(start, end).unsqueeze(1).to(args.device)
            with torch.no_grad():
                out = model(ids)
            h = out.hidden_states[layer_idx].squeeze(1).cpu().numpy()
            all_embs.append(h)
        vocab = np.concatenate(all_embs, axis=0)
        ctx_vocabs.append(vocab)
        print(f"  Layer {layer_idx:>2d}: avg norm {np.mean(np.linalg.norm(vocab, axis=1)):>8.1f}")

    # ================================================================
    # Part 2: Compute per-layer biases
    # ================================================================
    print(f"\nComputing per-layer biases from {len(refs)} reference sentences...")
    biases = compute_per_layer_bias(model, tokenizer, args.device, refs, ctx_vocabs)

    print(f"\n{'Layer':>6s}  {'Bias Norm':>10s}  {'Consistency':>12s}")
    print(f"{'-'*32}")

    # Also check consistency across refs for each layer
    for layer_idx in range(num_layers + 1):
        bias_norm = np.linalg.norm(biases[layer_idx])

        # Check consistency: compute per-ref biases and measure cosine between them
        ref_biases = []
        for ref in refs:
            rtoks = tokenizer.encode(ref)
            hs = get_hidden_states(model, rtoks, args.device)
            sentence_sum = hs[layer_idx].sum(axis=0)
            token_sum = ctx_vocabs[layer_idx][rtoks].sum(axis=0)
            ref_biases.append((sentence_sum - token_sum) / len(rtoks))

        # Average pairwise cosine
        cos_sims = []
        for i in range(len(ref_biases)):
            for j in range(i+1, len(ref_biases)):
                a, b = ref_biases[i], ref_biases[j]
                cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                cos_sims.append(cos)
        avg_cos = np.mean(cos_sims)

        print(f"  L{layer_idx:>2d}    {bias_norm:>10.1f}  {avg_cos:>12.6f}")

    # ================================================================
    # Part 3: Decompose with aggregate vs per-layer subtraction
    # ================================================================
    print(f"\n{'='*60}")
    print("DECOMPOSITION COMPARISON")

    # Get sentence hidden states
    hs = get_hidden_states(model, tokens, args.device)

    # Method A: No bias subtraction (baseline)
    last_layer = num_layers
    ctx_mean = ctx_vocabs[last_layer].mean(axis=0)
    ctx_centered = ctx_vocabs[last_layer] - ctx_mean
    sentence_sum = (hs[last_layer] - ctx_mean).sum(axis=0)

    rec_none = decompose_greedy(sentence_sum, ctx_centered, n + 10)
    hits_none = len(unique & set(rec_none))
    print(f"\n  No bias subtraction (L{last_layer}): {hits_none}/{u}")

    # Method B: Aggregate bias subtraction (what we've been doing)
    agg_bias = biases[last_layer]
    est_n = np.linalg.norm(sentence_sum) / np.linalg.norm(agg_bias)
    fixed_agg = sentence_sum - est_n * agg_bias

    rec_agg = decompose_greedy(fixed_agg, ctx_centered, n + 10)
    hits_agg = len(unique & set(rec_agg))
    toks_agg = [tokenizer.decode([r]) for r in rec_agg[:8]]
    print(f"  Aggregate bias subtraction (L{last_layer}): {hits_agg}/{u}  {toks_agg}")

    # Method C: Per-layer sequential subtraction
    # Start with the sentence hidden state sum at the last layer
    # Subtract bias layer by layer from outermost inward
    print(f"\n  Per-layer sequential subtraction:")

    residual = sentence_sum.copy()
    for peel_layer in range(last_layer, -1, -1):
        bias = biases[peel_layer]
        bias_norm = np.linalg.norm(bias)
        if bias_norm < 1e-10:
            continue

        # Estimate how much of this layer's bias to subtract
        # Project residual onto bias direction
        projection = np.dot(residual, bias) / (bias_norm * bias_norm)
        residual = residual - projection * bias

        r_norm = np.linalg.norm(residual)

        # Try decomposing at this point
        rec = decompose_greedy(residual, ctx_centered, n + 10)
        hits = len(unique & set(rec))
        toks = [tokenizer.decode([r]) for r in rec[:6]]
        print(f"    After removing L{peel_layer:>2d} bias: norm={r_norm:>8.1f}  hits={hits}/{u}  {toks}")

    # Method D: Try decomposing at each individual layer
    print(f"\n  Decomposition at each layer (with that layer's bias subtracted):")
    for layer_idx in [0, 3, 6, 9, 12]:
        layer_mean = ctx_vocabs[layer_idx].mean(axis=0)
        layer_centered = ctx_vocabs[layer_idx] - layer_mean
        layer_sum = (hs[layer_idx] - layer_mean).sum(axis=0)

        # Subtract this layer's bias
        bias = biases[layer_idx]
        bias_norm = np.linalg.norm(bias)
        if bias_norm > 1e-10:
            est = np.linalg.norm(layer_sum) / bias_norm
            fixed = layer_sum - est * bias
        else:
            fixed = layer_sum

        rec = decompose_greedy(fixed, layer_centered, n + 10)
        hits = len(unique & set(rec))
        toks = [tokenizer.decode([r]) for r in rec[:8]]
        print(f"    L{layer_idx:>2d}: {hits}/{u}  {toks}")


if __name__ == "__main__":
    main()
