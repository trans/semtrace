#!/usr/bin/env python3
"""Layer peeling: test the strata hypothesis directly.

The strata hypothesis (paper Section 6.5) predicts that each transformer block
adds a thin additive sediment (the per-token bias) to the residual stream, and
that subtracting these biases in order should let us walk a deep hidden state
inward through the network — approximately reconstructing earlier-layer states.

This experiment:
  1. Forward-passes a sentence and captures all hidden states
  2. Computes per-token biases at every layer (sink-skipped, EOT-prefixed vocab)
  3. Builds the corrected vocab at every layer
  4. Starts with the L11 trailing-position sum
  5. Subtracts per-layer biases in order: -N*bias[11], then -N*bias[10], ...
  6. After each subtraction, decomposes against the corresponding earlier-layer vocab
  7. Also directly compares the peeled vector to the actual earlier-layer trailing sum

Three things we want to know:
  A. Does layer peeling improve recovery vs decomposing at L11 directly?
  B. Does each peeling step reduce the distance to the actual earlier-layer state?
  C. Where (if anywhere) does peeling break down?

Usage:
  python3 layer_peel.py [--text "the cat sat on the mat"]
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


def build_vocab_pos1(model, layer, device, prefix_token, batch=256):
    vocab_size = model.wte.weight.shape[0]
    chunks = []
    for start in range(0, vocab_size, batch):
        end = min(start + batch, vocab_size)
        ids = torch.tensor([[prefix_token, t] for t in range(start, end)]).to(device)
        with torch.no_grad():
            out = model(ids)
        h = out.hidden_states[layer][:, 1, :].cpu().numpy()
        chunks.append(h)
    return np.concatenate(chunks, axis=0)


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="the cat sat on the mat")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True).to(args.device)
    model.eval()
    eot = tokenizer.eos_token_id

    # Encode test sentence with EOT prepended
    raw_tokens = tokenizer.encode(args.text)
    tokens = [eot] + raw_tokens
    trailing_tokens = tokens[1:]
    unique = set(trailing_tokens)
    n_t = len(trailing_tokens)
    u = len(unique)
    print(f"Text: {args.text!r}")
    print(f"Trailing tokens ({n_t}): {[tokenizer.decode([t]) for t in trailing_tokens]}")
    print(f"Unique trailing: {u}")

    # Forward pass — get all hidden states
    with torch.no_grad():
        out = model(torch.tensor([tokens]).to(args.device))
    all_hs = [h.squeeze(0).cpu().numpy() for h in out.hidden_states]
    # all_hs[L] is shape (n_tok, 768) for L = 0..12
    # We care about L = 1..11 (residual stream blocks, no embedding, no post-ln_f)

    layers = list(range(1, 12))  # L1..L11

    # Reference sentences for bias computation
    refs = [
        "the quick brown fox jumps over the lazy dog",
        "she went to the store to buy some food",
        "he ran down the long road to the old house",
        "they played in the park with the children",
    ]
    ref_token_lists = [tokenizer.encode(r) for r in refs]

    # Pre-compute reference hiddens at every layer (one forward pass per ref)
    print("\nForward-passing reference sentences...")
    ref_hiddens = {}
    for ri, rt in enumerate(ref_token_lists):
        with torch.no_grad():
            rh = model(torch.tensor([rt]).to(args.device)).hidden_states
        for L in layers:
            ref_hiddens.setdefault(L, []).append(rh[L].squeeze(0).cpu().numpy())

    # Build per-layer corrected vocabs and per-layer biases
    print(f"\nBuilding corrected vocabs and biases at layers {layers}...")
    vocabs = {}
    biases = {}
    for L in layers:
        v = build_vocab_pos1(model, L, args.device, eot)
        vocabs[L] = v

        # Per-token bias at this layer (sink-skipped)
        bias_accum = []
        for rt, rh in zip(ref_token_lists, ref_hiddens[L]):
            trailing = rh[1:]
            n_r = trailing.shape[0]
            bias_accum.append((trailing.sum(axis=0) - v[rt[1:]].sum(axis=0)) / n_r)
        biases[L] = np.mean(bias_accum, axis=0)
        print(f"  L{L:2d}: vocab norm avg {np.mean(np.linalg.norm(v, axis=1)):.1f}, bias norm {np.linalg.norm(biases[L]):.2f}")

    # Trailing-position sums at every layer (the "true" earlier-layer states)
    layer_sums = {L: all_hs[L][1:].sum(axis=0) for L in layers}

    # ==================================================================
    # PART 1: Direct decomposition at each layer (baseline)
    # ==================================================================
    print(f"\n{'='*78}")
    print("PART 1: Direct decomposition at each layer (sink-skip + N-debias)")
    print(f"{'='*78}")
    print(f"  {'Layer':>6s}  {'BiasNorm':>9s}  {'TargetNorm':>11s}  {'Recovery':>10s}")
    print("  " + "-" * 50)

    direct_results = {}
    for L in layers:
        target = layer_sums[L]
        debiased = target - n_t * biases[L]
        rec = decompose_greedy(debiased, vocabs[L], n_t + 10)
        hits = len(unique & set(rec))
        direct_results[L] = hits
        print(f"  L{L:>2d}    {np.linalg.norm(biases[L]):>9.2f}  {np.linalg.norm(target):>11.2f}  {hits:>4d}/{u:<4d}")

    # ==================================================================
    # PART 2: Layer peeling — start at L11, peel inward
    # ==================================================================
    print(f"\n{'='*78}")
    print("PART 2: Layer peeling — start at L11, subtract biases inward")
    print(f"{'='*78}")
    print("  At each step, we subtract N * bias[L] from the running residual,")
    print("  then decompose against the L-1 vocab and compare to the true L-1 sum.")
    print()
    print(f"  {'Step':>6s}  {'After':>8s}  {'PeeledNorm':>11s}  {'TrueNorm':>10s}  {'cos(peel,true)':>15s}  {'DistRatio':>10s}  {'DecodeAt':>10s}  {'Hits':>6s}")
    print("  " + "-" * 100)

    # Start at L11
    residual = layer_sums[11].copy()
    print(f"  start  {'L11':>8s}  {np.linalg.norm(residual):>11.2f}  {np.linalg.norm(layer_sums[11]):>10.2f}  {1.0:>15.4f}  {0.0:>10.4f}  {'L11':>10s}  ")

    # Peel each layer's bias and decompose against the lower-layer vocab
    for L in range(11, 1, -1):
        # Subtract the bias of layer L (the sediment that block L deposited)
        residual = residual - n_t * biases[L]

        # Compare to the true L-1 sum
        true_lower = layer_sums[L - 1]
        c = cos(residual, true_lower)
        peel_norm = np.linalg.norm(residual)
        true_norm = np.linalg.norm(true_lower)
        # Distance ratio: ||residual - true_lower|| / ||true_lower||
        dist = np.linalg.norm(residual - true_lower) / true_norm

        # Decompose the peeled residual against the L-1 vocab
        # Note: the residual still contains the L-1 (and earlier) per-token biases,
        # so we should debias it with the L-1 bias before decoding
        decode_target = residual - n_t * biases[L - 1]
        rec = decompose_greedy(decode_target, vocabs[L - 1], n_t + 10)
        hits = len(unique & set(rec))

        print(f"  -L{L:<2d}   L{L-1:>2d}      {peel_norm:>11.2f}  {true_norm:>10.2f}  {c:>15.4f}  {dist:>10.4f}  {'L'+str(L-1):>10s}  {hits:>4d}/{u:<4d}")

    # ==================================================================
    # PART 3: Direct comparison: how close is peel(L11→L1) to true L1?
    # ==================================================================
    print(f"\n{'='*78}")
    print("PART 3: Distance between peeled and true earlier-layer states (no peel intermediary)")
    print(f"{'='*78}")
    print("  Single-shot peel: starting from L11, subtract biases L11..L+1 in one go,")
    print("  compare directly to the true layer L sum.")
    print()
    print(f"  {'TargetL':>8s}  {'PeeledNorm':>11s}  {'TrueNorm':>10s}  {'cos(peel,true)':>15s}  {'DistRatio':>10s}")
    print("  " + "-" * 75)

    for target_L in range(10, 0, -1):
        residual = layer_sums[11].copy()
        for L in range(11, target_L, -1):
            residual = residual - n_t * biases[L]
        true_target = layer_sums[target_L]
        c = cos(residual, true_target)
        dist = np.linalg.norm(residual - true_target) / np.linalg.norm(true_target)
        print(f"  L{target_L:>2d}      {np.linalg.norm(residual):>11.2f}  {np.linalg.norm(true_target):>10.2f}  {c:>15.4f}  {dist:>10.4f}")

    del vocabs
    gc.collect()


if __name__ == "__main__":
    main()
