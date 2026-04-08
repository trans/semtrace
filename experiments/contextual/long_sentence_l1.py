#!/usr/bin/env python3
"""Contextual decomposition on long-text inputs.

The static decomposition baseline (experiment 001) was Gettysburg Address,
~143 unique tokens, recovered at 93%+ via greedy + coordinate descent on
GPT-2 Small. We've never tested the corrected contextual method (sink-skip
+ N-debias) on text that long.

Three test texts in increasing length, all decomposed at L1, L6, L11
against the corrected contextual vocab. We use the low band (debias→split)
since that consistently slightly outperforms the full vector.
"""
import gc
import os
import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer


GETTYSBURG = (
    "Four score and seven years ago our fathers brought forth on this continent, "
    "a new nation, conceived in Liberty, and dedicated to the proposition that "
    "all men are created equal. Now we are engaged in a great civil war, testing "
    "whether that nation, or any nation so conceived and so dedicated, can long endure."
)

MEDIUM = (
    "the quick brown fox jumps over the lazy dog while the cat watches from the "
    "windowsill and the children play in the garden under the warm afternoon sun"
)

SHORT = "the cat sat on the mat and the dog ran around the yard"


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


def coord_descent(target, vocab, n_tokens, max_iters=15):
    """Coord descent over token assignments minimizing ||target - sum(vocab[t])||."""
    current = decompose_greedy(target, vocab, n_tokens)
    while len(current) < n_tokens:
        ideal = target - vocab[current].sum(axis=0)
        n = np.linalg.norm(vocab, axis=1)
        n[n < 1e-10] = 1.0
        sims = vocab @ ideal / n
        current.append(int(np.argmax(sims)))
    current = current[:n_tokens]

    best_dist = np.linalg.norm(target - vocab[current].sum(axis=0))
    for _ in range(max_iters):
        improved = False
        for pos in range(n_tokens):
            others = [current[j] for j in range(n_tokens) if j != pos]
            others_sum = vocab[others].sum(axis=0)
            ideal = target - others_sum
            n = np.linalg.norm(vocab, axis=1)
            n[n < 1e-10] = 1.0
            sims = vocab @ ideal / n
            best_for_pos = int(np.argmax(sims))
            if best_for_pos != current[pos]:
                new_tokens = list(current)
                new_tokens[pos] = best_for_pos
                new_dist = np.linalg.norm(target - vocab[new_tokens].sum(axis=0))
                if new_dist < best_dist:
                    current[pos] = best_for_pos
                    best_dist = new_dist
                    improved = True
        if not improved:
            break
    return current


def int8_quantize(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        scale = np.max(np.abs(x)) / 127.0
        if scale < 1e-12:
            return x.copy()
        return np.round(x / scale).clip(-127, 127) * scale
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        out[i] = int8_quantize(x[i])
    return out


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


def main():
    device = "cpu"
    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True).to(device)
    model.eval()
    eot = tokenizer.eos_token_id

    texts = [
        ("short", SHORT),
        ("medium", MEDIUM),
        ("gettysburg", GETTYSBURG),
    ]
    layers = [1, 6, 11]
    refs = [
        "the quick brown fox jumps over the lazy dog",
        "she went to the store to buy some food",
        "he ran down the long road to the old house",
        "they played in the park with the children",
    ]
    ref_token_lists = [tokenizer.encode(r) for r in refs]
    ref_hs_per_layer = {L: [] for L in layers}
    for rt in ref_token_lists:
        with torch.no_grad():
            out = model(torch.tensor([rt]).to(device))
        for L in layers:
            ref_hs_per_layer[L].append(out.hidden_states[L].squeeze(0).cpu().numpy())

    # Forward-pass each test text
    print("Forward-passing test texts...")
    test_data = []
    for name, text in texts:
        tt = [eot] + tokenizer.encode(text)
        with torch.no_grad():
            out = model(torch.tensor([tt]).to(device))
        hs = [h.squeeze(0).cpu().numpy() for h in out.hidden_states]
        n_t = len(tt) - 1
        u_t = len(set(tt[1:]))
        print(f"  {name:12s}: {n_t} tokens, {u_t} unique")
        test_data.append((name, tt, hs))

    for L in layers:
        print(f"\n{'='*78}")
        print(f"LAYER L{L}")
        print(f"{'='*78}")

        v = build_vocab_pos1(model, L, device, eot)
        v_low = int8_quantize(v)

        bias_accum = []
        for rt, rh in zip(ref_token_lists, ref_hs_per_layer[L]):
            trailing = rh[1:]
            n_r = trailing.shape[0]
            bias_accum.append((trailing.sum(axis=0) - v[rt[1:]].sum(axis=0)) / n_r)
        bias = np.mean(bias_accum, axis=0)
        print(f"  Per-token bias norm: {np.linalg.norm(bias):.2f}")

        for name, tt, hs_per_layer in test_data:
            trailing_tokens = tt[1:]
            unique = set(trailing_tokens)
            u = len(unique)
            n_t = len(trailing_tokens)

            target = hs_per_layer[L][1:].sum(axis=0)
            debiased = target - n_t * bias
            debiased_low = int8_quantize(debiased)

            # Greedy on full debiased
            rec_full_g = decompose_greedy(debiased, v, n_t + 10)
            hits_full_g = len(unique & set(rec_full_g))

            # Greedy on debiased low band
            rec_low_g = decompose_greedy(debiased_low, v_low, n_t + 10)
            hits_low_g = len(unique & set(rec_low_g))

            # Coord descent on full debiased
            rec_full_cd = coord_descent(debiased, v, n_t)
            hits_full_cd = len(unique & set(rec_full_cd))

            # Coord descent on low band
            rec_low_cd = coord_descent(debiased_low, v_low, n_t)
            hits_low_cd = len(unique & set(rec_low_cd))

            print(f"  {name:12s} (n={n_t}, unique={u}):")
            print(f"    greedy full: {hits_full_g}/{u} ({100*hits_full_g/u:.0f}%)")
            print(f"    greedy low : {hits_low_g}/{u} ({100*hits_low_g/u:.0f}%)")
            print(f"    CD full    : {hits_full_cd}/{u} ({100*hits_full_cd/u:.0f}%)")
            print(f"    CD low     : {hits_low_cd}/{u} ({100*hits_low_cd/u:.0f}%)")

        del v, v_low
        gc.collect()


if __name__ == "__main__":
    main()
