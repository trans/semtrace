#!/usr/bin/env python3
"""
Contextual Embedding Decomposition

Tests decomposition at middle and last transformer layers against both
static and contextual vocabularies, with cosine, L2, and inner product.

Usage:
  python3 run.py [--text "the cat sat on the mat"] [--model gpt2] [--device cpu]
"""

import argparse
import torch
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer


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
            dists = np.sum((embeddings - residual) ** 2, axis=1)
            best = np.argmin(dists)
        recovered.append(int(best))
        residual = residual - embeddings[best]
    return recovered


def embed_single_tokens(model, vocab_size, device, layer):
    all_embs = []
    for start in range(0, vocab_size, 512):
        end = min(start + vocab_size, vocab_size)
        ids = torch.arange(start, end).unsqueeze(1).to(device)
        with torch.no_grad():
            out = model(ids)
        all_embs.append(out.hidden_states[layer].squeeze(1).cpu().numpy())
    return np.concatenate(all_embs, axis=0)


def run_tests(target, vocab, unique_tokens, max_steps, tokenizer):
    results = {}
    all_recovered = set()
    for metric in ["cosine", "ip", "l2"]:
        rec = decompose_greedy(target, vocab, max_steps, metric)
        hits = len(unique_tokens & set(rec))
        results[metric] = hits
        all_recovered.update(rec)
        toks = [tokenizer.decode([r]) for r in rec[:8]]
        results[f"{metric}_toks"] = toks
    results["union"] = len(unique_tokens & all_recovered)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="the cat sat on the mat")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {args.model} on {device}...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    model = GPT2Model.from_pretrained(args.model, output_hidden_states=True).to(device)
    model.eval()

    wte = model.wte.weight.detach().cpu().numpy()
    vocab_size, dims = wte.shape
    tokens = tokenizer.encode(args.text)
    token_strs = [tokenizer.decode([t]) for t in tokens]
    unique_tokens = set(tokens)
    n, u = len(tokens), len(unique_tokens)
    num_layers = model.config.n_layer
    mid = num_layers // 2

    print(f"Embeddings: {vocab_size} x {dims}d, Layers: {num_layers}")
    print(f"Text: {args.text!r}")
    print(f"Tokens ({n}): {token_strs}, Unique: {u}")

    # Forward pass
    with torch.no_grad():
        outputs = model(torch.tensor([tokens]).to(device))
    hidden_states = [h.squeeze(0).cpu().numpy() for h in outputs.hidden_states]

    max_steps = n + 10

    # Part 1: Static baseline
    print(f"\n{'='*60}")
    print("STATIC BASELINE: sum of static embeddings → static vocab")
    static_sum = wte[tokens].sum(axis=0)
    r = run_tests(static_sum, wte, unique_tokens, max_steps, tokenizer)
    print(f"  cos={r['cosine']}/{u}  ip={r['ip']}/{u}  l2={r['l2']}/{u}  union={r['union']}/{u}")

    # Part 2: Sentence → Static vocab
    print(f"\n{'='*60}")
    print("SENTENCE → STATIC VOCAB")
    for li, lname in [(0, "embed"), (mid, f"L{mid}"), (num_layers, f"L{num_layers}")]:
        h = hidden_states[li]
        for pname, pooled in [("mean", h.mean(0)), ("last", h[-1])]:
            r = run_tests(pooled, wte, unique_tokens, max_steps, tokenizer)
            print(f"  {lname:>5s} {pname:>4s}: cos={r['cosine']}/{u}  ip={r['ip']}/{u}  l2={r['l2']}/{u}  union={r['union']}/{u}")

    # Part 3: Build contextual vocab at mid and last layer
    for li, lname in [(mid, f"L{mid}"), (num_layers, f"L{num_layers}")]:
        print(f"\n{'='*60}")
        print(f"Building contextual vocab at {lname}...", flush=True)
        ctx_vocab = embed_single_tokens(model, vocab_size, device, layer=li)
        print(f"  Shape: {ctx_vocab.shape}, avg norm: {np.mean(np.linalg.norm(ctx_vocab, axis=1)):.1f}")

        # Part 3a: Sentence → Contextual vocab
        print(f"\nSENTENCE → CONTEXTUAL VOCAB ({lname})")
        h = hidden_states[li]
        for pname, pooled in [("mean", h.mean(0)), ("last", h[-1])]:
            r = run_tests(pooled, ctx_vocab, unique_tokens, max_steps, tokenizer)
            print(f"  {pname:>4s}: cos={r['cosine']}/{u}  ip={r['ip']}/{u}  l2={r['l2']}/{u}  union={r['union']}/{u}")
            best = max(["cosine", "ip", "l2"], key=lambda m: r[m])
            print(f"        best ({best}): {r[best+'_toks']}")

        # Part 3b: Contextual bag-of-words → Contextual vocab (control)
        print(f"\nCONTEXTUAL BAG-OF-WORDS → CONTEXTUAL VOCAB ({lname})")
        ctx_sum = ctx_vocab[tokens].sum(axis=0)
        r = run_tests(ctx_sum, ctx_vocab, unique_tokens, max_steps, tokenizer)
        print(f"  cos={r['cosine']}/{u}  ip={r['ip']}/{u}  l2={r['l2']}/{u}  union={r['union']}/{u}")
        best = max(["cosine", "ip", "l2"], key=lambda m: r[m])
        print(f"  best ({best}): {r[best+'_toks']}")


if __name__ == "__main__":
    main()
