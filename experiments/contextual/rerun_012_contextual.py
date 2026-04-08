#!/usr/bin/env python3
"""Re-run experiment 012 (contextual half) with sink-skip + N-debias initialization.

Tests coordinate descent on top of the corrected greedy decomposition for the
four sentences from the original 012 contextual table:
  - the cat sat on the mat
  - Mary had a little lamb
  - the dog ran in the park
  - I love cats and dogs

For each sentence, runs greedy and CD against the corrected vocab (built with
[<|endoftext|>, token] pairs at position 1) using sink-skipped target sums and
N-corrected debias.
"""
import gc
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
    current = greedy_decompose(target, vocab, n_tokens)
    while len(current) < n_tokens:
        ideal = target - vocab[current].sum(axis=0)
        best = nearest_k(ideal, vocab, k=1)[0]
        current.append(int(best))
    current = current[:n_tokens]

    best_dist = np.linalg.norm(target - vocab[current].sum(axis=0))
    iters_used = 0
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
        iters_used = iteration + 1
        if not improved:
            break
    return current, best_dist, iters_used


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
    sentences = [
        "the cat sat on the mat",
        "Mary had a little lamb",
        "the dog ran in the park",
        "I love cats and dogs",
    ]
    L = 1  # Best layer for sink-skip method
    refs = [
        "the quick brown fox jumps over the lazy dog",
        "she went to the store to buy some food",
        "he ran down the long road to the old house",
        "they played in the park with the children",
    ]

    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True).to(device)
    model.eval()
    eot = tokenizer.eos_token_id

    print(f"Building corrected vocab at L{L}...")
    vocab = build_vocab_pos1(model, L, device, eot)

    # Compute per-token bias from refs (sink-skipped)
    bias_accum = []
    for ref in refs:
        rt = tokenizer.encode(ref)
        with torch.no_grad():
            rh = model(torch.tensor([rt]).to(device)).hidden_states[L].squeeze(0).cpu().numpy()
        trailing = rh[1:]
        n_t = trailing.shape[0]
        bias_accum.append((trailing.sum(axis=0) - vocab[rt[1:]].sum(axis=0)) / n_t)
    bias = np.mean(bias_accum, axis=0)
    print(f"Per-token bias norm: {np.linalg.norm(bias):.2f}")

    print(f"\n{'='*70}")
    print(f"CONTEXTUAL DECOMPOSITION + COORD DESCENT (L{L}, sink-skip + N-debias)")
    print(f"{'='*70}")
    print(f"\n{'Sentence':45s}  {'unique':>6s}  {'Greedy':>10s}  {'CD':>10s}  CD tokens")
    print("-" * 130)

    for sent in sentences:
        tt = [eot] + tokenizer.encode(sent)
        with torch.no_grad():
            hs = model(torch.tensor([tt]).to(device)).hidden_states[L].squeeze(0).cpu().numpy()
        trailing_target = hs[1:].sum(axis=0)
        trailing_tokens = tt[1:]
        unique = set(trailing_tokens)
        n_t = len(trailing_tokens)
        u = len(unique)

        debiased = trailing_target - n_t * bias

        greedy_recs = greedy_decompose(debiased, vocab, n_t)
        greedy_hits = len(unique & set(greedy_recs))

        cd_result, cd_dist, cd_iters = coord_descent(debiased, vocab, n_t, max_iters=20)
        cd_hits = len(unique & set(cd_result))
        cd_toks = [tokenizer.decode([t]) for t in cd_result]

        print(f"  {sent[:43]:45s}  {u:>6d}  {greedy_hits:>4d}/{u:<4d}    {cd_hits:>4d}/{u:<4d}    {cd_toks}")

    del vocab
    gc.collect()


if __name__ == "__main__":
    main()
