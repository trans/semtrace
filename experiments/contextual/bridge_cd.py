#!/usr/bin/env python3
"""Bridge + Coordinate Descent.

Take the linear-bridged contextual sum and apply coordinate descent in static
space. The bridge gives us an approximation of the static sum; CD then refines
the token assignment by iteratively swapping each position's token for whichever
one minimizes the residual to the bridged target.

This combines what we know works:
  - Sink-skip + bias subtraction (clean contextual sum)
  - Linear bridge to static space (W from least-squares fit)
  - Coordinate descent in static space (the 43% → 93% improvement we know from
    the static Gettysburg result)

Tested at L1, L6, L11, L12.
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
    for _ in range(max_iters):
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
    return current


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

    wte = model.wte.weight.detach().cpu().numpy()

    test_sentences = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "Mary had a little lamb",
        "Four score and seven years ago",
        "I love cats and dogs",
        "the big red car drove fast down the road",
    ]
    layers = [1, 6, 11, 12]

    refs = [
        "the quick brown fox jumps over the lazy dog",
        "she went to the store to buy some food",
        "he ran down the long road to the old house",
        "they played in the park with the children",
    ]
    ref_token_lists = [tokenizer.encode(r) for r in refs]

    print("Forward-passing test sentences...")
    test_data = []
    for sent in test_sentences:
        tt = [eot] + tokenizer.encode(sent)
        with torch.no_grad():
            out = model(torch.tensor([tt]).to(device))
        hs = [h.squeeze(0).cpu().numpy() for h in out.hidden_states]
        test_data.append((sent, tt, hs))

    print("Forward-passing reference sentences...")
    ref_hs_per_layer = {L: [] for L in layers}
    for rt in ref_token_lists:
        with torch.no_grad():
            out = model(torch.tensor([rt]).to(device))
        for L in layers:
            ref_hs_per_layer[L].append(out.hidden_states[L].squeeze(0).cpu().numpy())

    for L in layers:
        print(f"\n{'='*78}")
        print(f"LAYER L{L}")
        print(f"{'='*78}")

        v = build_vocab_pos1(model, L, device, eot)

        bias_accum = []
        for rt, rh in zip(ref_token_lists, ref_hs_per_layer[L]):
            trailing = rh[1:]
            n_r = trailing.shape[0]
            bias_accum.append((trailing.sum(axis=0) - v[rt[1:]].sum(axis=0)) / n_r)
        bias = np.mean(bias_accum, axis=0)

        W, _, _, _ = np.linalg.lstsq(v, wte, rcond=None)

        print()
        print(f"  {'Sentence':45s}  {'unique':>6s}  {'bridge':>9s}  {'bridge+CD':>11s}  CD tokens")
        print("  " + "-" * 115)

        for sent, tt, hs_per_layer in test_data:
            trailing_tokens = tt[1:]
            unique = set(trailing_tokens)
            u = len(unique)
            n_t = len(trailing_tokens)

            ctx_sum = hs_per_layer[L][1:].sum(axis=0)
            # Variation A from bridge_decompose_v2: bias-subtract before bridge
            biased = (ctx_sum - n_t * bias) @ W

            # Greedy bridge baseline
            greedy_rec = greedy_decompose(biased, wte, n_t)
            greedy_hits = len(unique & set(greedy_rec))

            # Coord descent on bridged target
            cd_rec = coord_descent(biased, wte, n_t, max_iters=20)
            cd_hits = len(unique & set(cd_rec))
            cd_toks = [tokenizer.decode([r]) for r in cd_rec]

            print(f"  {sent[:43]:45s}  {u:>6d}  {greedy_hits:>4d}/{u:<4d}    {cd_hits:>4d}/{u:<4d}     {cd_toks}")

        del v, W
        gc.collect()


if __name__ == "__main__":
    main()
