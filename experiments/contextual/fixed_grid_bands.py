#!/usr/bin/env python3
"""Test fixed-grid (decimal-style) band split, where the grid step is a
fixed constant rather than per-vector adaptive (like int8).

Per-vector int8 is scale-invariant in proportional terms, so scaling the
target up doesn't change the band split. A fixed grid IS scale-sensitive,
so scaling matters and the bands could behave differently.

For each fixed grid step, split the target and decompose the high band.
"""
import gc
import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer


SENTENCES = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "Mary had a little lamb",
    "Four score and seven years ago",
    "I love cats and dogs",
    "the big red car drove fast down the road",
    "she walked slowly to the river",
    "the old man fished from the boat",
    "children played with the colorful kite",
    "morning sun warmed the sleepy village",
    "snow fell quietly on the mountain pass",
    "he opened the book and began to read",
    "the wizard cast a powerful spell",
    "rain washed over the dusty street",
    "music drifted from the open window",
    "a small bird sang in the tall tree",
    "the chef prepared a delicious meal",
    "fire crackled in the stone fireplace",
    "the runner crossed the finish line first",
    "stars filled the dark night sky",
]


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


def fixed_grid_split(x, step):
    """Split x at a fixed grid step (independent of x's magnitude)."""
    low = np.round(x / step) * step
    high = x - low
    return low, high


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

    L = 1
    refs = [
        "the quick brown fox jumps over the lazy dog",
        "she went to the store to buy some food",
        "he ran down the long road to the old house",
        "they played in the park with the children",
    ]
    ref_token_lists = [tokenizer.encode(r) for r in refs]

    print(f"Building vocab + bias at L{L}...")
    v = build_vocab_pos1(model, L, device, eot)

    bias_accum = []
    for rt in ref_token_lists:
        with torch.no_grad():
            rh = model(torch.tensor([rt]).to(device)).hidden_states[L].squeeze(0).cpu().numpy()
        trailing = rh[1:]
        n_r = trailing.shape[0]
        bias_accum.append((trailing.sum(axis=0) - v[rt[1:]].sum(axis=0)) / n_r)
    bias = np.mean(bias_accum, axis=0)

    test_data = []
    for sent in SENTENCES:
        tt = [eot] + tokenizer.encode(sent)
        with torch.no_grad():
            out = model(torch.tensor([tt]).to(device))
        hs = [h.squeeze(0).cpu().numpy() for h in out.hidden_states]
        test_data.append((sent, tt, hs))
    total_unique = sum(len(set(tt[1:])) for _, tt, _ in test_data)

    # Try fixed-grid splits at various step sizes
    # Vocab values at L1 typically range from -10 to +10. Steps to try:
    grid_steps = [0.01, 0.1, 1.0, 5.0]

    print(f"\nFixed-grid band split at L{L}, decompose high band against quantized vocab:")
    print(f"  vocab norm avg: {np.mean(np.linalg.norm(v, axis=1)):.2f}")
    print(f"  vocab value range: [{v.min():.3f}, {v.max():.3f}]")
    print(f"  bias norm: {np.linalg.norm(bias):.2f}")
    print()

    for step in grid_steps:
        v_low, v_high = fixed_grid_split(v, step)
        v_high_norm_avg = np.mean(np.linalg.norm(v_high, axis=1))
        v_low_norm_avg = np.mean(np.linalg.norm(v_low, axis=1))

        hits_low_total = 0
        hits_high_total = 0
        hits_full_total = 0  # baseline

        for sent, tt, hs_per_layer in test_data:
            trailing_tokens = tt[1:]
            unique = set(trailing_tokens)
            n_t = len(trailing_tokens)

            target_full = hs_per_layer[L][1:].sum(axis=0)
            debiased = target_full - n_t * bias

            tg_low, tg_high = fixed_grid_split(debiased, step)

            rec_full = decompose_greedy(debiased, v, n_t + 10)
            rec_low = decompose_greedy(tg_low, v_low, n_t + 10)
            rec_high = decompose_greedy(tg_high, v_high, n_t + 10)

            hits_full_total += len(unique & set(rec_full))
            hits_low_total += len(unique & set(rec_low))
            hits_high_total += len(unique & set(rec_high))

        print(f"  step={step}:")
        print(f"    vocab low avg norm: {v_low_norm_avg:.3f}, vocab high avg norm: {v_high_norm_avg:.3f}")
        print(f"    full   {hits_full_total}/{total_unique}  ({100*hits_full_total/total_unique:.1f}%)")
        print(f"    low    {hits_low_total}/{total_unique}  ({100*hits_low_total/total_unique:.1f}%)")
        print(f"    high   {hits_high_total}/{total_unique}  ({100*hits_high_total/total_unique:.1f}%)")
        print()

        del v_low, v_high
        gc.collect()


if __name__ == "__main__":
    main()
