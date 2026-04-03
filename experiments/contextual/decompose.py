#!/usr/bin/env python3
"""Decompose a sentence embedding against pre-built vocabularies.
Requires build_ctx_vocab.py to have been run first.

Usage:
  python3 decompose.py --text "the cat sat on the mat" [--vocabdir .]
"""
import argparse
import numpy as np
import torch
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
            best = np.argmin(np.sum((embeddings - residual) ** 2, axis=1))
        recovered.append(int(best))
        residual = residual - embeddings[best]
    return recovered


def test_all(target, vocab, unique_tokens, max_steps, tokenizer):
    all_recovered = set()
    for metric in ["cosine", "ip", "l2"]:
        rec = decompose_greedy(target, vocab, max_steps, metric)
        hits = len(unique_tokens & set(rec))
        all_recovered.update(rec)
        toks = [tokenizer.decode([r]) for r in rec[:10]]
        print(f"    {metric:8s}: {hits}/{len(unique_tokens)} ({100*hits/len(unique_tokens):.0f}%)  {toks}")
    union_hits = len(unique_tokens & all_recovered)
    print(f"    {'union':8s}: {union_hits}/{len(unique_tokens)} ({100*union_hits/len(unique_tokens):.0f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="the cat sat on the mat")
    parser.add_argument("--vocabdir", default=".")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print(f"Loading model for forward pass...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True).to(args.device)
    model.eval()

    tokens = tokenizer.encode(args.text)
    token_strs = [tokenizer.decode([t]) for t in tokens]
    unique_tokens = set(tokens)
    n, u = len(tokens), len(unique_tokens)
    max_steps = n + 10

    print(f"Text: {args.text!r}")
    print(f"Tokens ({n}): {token_strs}, Unique: {u}")

    # Forward pass
    with torch.no_grad():
        outputs = model(torch.tensor([tokens]).to(args.device))
    hidden_states = {
        "L6": outputs.hidden_states[6].squeeze(0).cpu().numpy(),
        "L12": outputs.hidden_states[12].squeeze(0).cpu().numpy(),
    }

    # Free model memory
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    # Load pre-built vocabs
    print(f"\nLoading vocabularies from {args.vocabdir}/...")
    static_vocab = np.load(f"{args.vocabdir}/static_vocab.npy")
    ctx_L6 = np.load(f"{args.vocabdir}/ctx_vocab_L6.npy")
    ctx_L12 = np.load(f"{args.vocabdir}/ctx_vocab_L12.npy")
    print(f"  static: {static_vocab.shape}, L6: {ctx_L6.shape}, L12: {ctx_L12.shape}")

    # Part 1: Static baseline
    print(f"\n{'='*60}")
    print("STATIC BASELINE: sum of static → static vocab")
    static_sum = static_vocab[tokens].sum(axis=0)
    test_all(static_sum, static_vocab, unique_tokens, max_steps, tokenizer)

    # Part 2: Sentence → contextual vocab (THE KEY TEST)
    for lname, hs, ctx in [("L6", hidden_states["L6"], ctx_L6), ("L12", hidden_states["L12"], ctx_L12)]:
        print(f"\n{'='*60}")
        print(f"SENTENCE → CONTEXTUAL VOCAB ({lname})")
        for pname, pooled in [("mean-pool", hs.mean(axis=0)), ("last-token", hs[-1])]:
            print(f"  {pname}:")
            test_all(pooled, ctx, unique_tokens, max_steps, tokenizer)

    # Part 3: Contextual bag-of-words → contextual vocab (control)
    for lname, ctx in [("L6", ctx_L6), ("L12", ctx_L12)]:
        print(f"\n{'='*60}")
        print(f"CONTEXTUAL BAG-OF-WORDS → CONTEXTUAL VOCAB ({lname})")
        ctx_sum = ctx[tokens].sum(axis=0)
        test_all(ctx_sum, ctx, unique_tokens, max_steps, tokenizer)


if __name__ == "__main__":
    main()
