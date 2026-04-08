#!/usr/bin/env python3
"""Verify that the 97% L1 pooled-vector result is robust, not an artifact
of the arbitrary `n_t * 100` scaling I used in pooled_and_more.py.

The greedy decomposer is scale-sensitive after the first iteration: cosine
is scale-invariant for picking the best vocab entry, but the residual
arithmetic (subtracting that entry from the target) depends on the target's
magnitude. Scale the target up by 100, and the residual is dominated by
the target direction. Scale it down to 1, and the residual is dominated by
the subtraction.

This script tests several scaling strategies on the L2-normalized mean-pool
target at L1 (where we got 97%) and reports whether the result is stable.

Strategies tested:
  1. scale = 1.0 (no rescaling after L2 normalize)
  2. scale = avg vocab norm (~63 at L1)
  3. scale = avg vocab norm * sqrt(N) (matches expected magnitude of N
     random unit vectors summed)
  4. scale = avg vocab norm * N (matches expected magnitude of N aligned
     vectors summed, the upper bound)
  5. scale = N * 100 (what I originally used)
  6. scale = original mean-pool magnitude (i.e., undo the L2 normalize) --
     this should give the same result as not normalizing in the first place
  7. scale-free top-N by cosine: pick the top-N vocab entries by single-step
     cosine to the target, no iterative residual subtraction

Also tests at L6, L11, L12 for comparison.
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


def topn_cosine(target, embeddings, n):
    """Scale-free: pick top-N by cosine to target. No iterative subtraction."""
    norms = np.linalg.norm(embeddings, axis=1)
    norms[norms < 1e-10] = 1.0
    t_norm = np.linalg.norm(target)
    if t_norm < 1e-10:
        return list(range(n))
    sims = embeddings @ target / (norms * t_norm)
    return list(np.argsort(-sims)[:n])


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

    layers = [1, 6, 11, 12]

    test_data = []
    for sent in SENTENCES:
        tt = [eot] + tokenizer.encode(sent)
        with torch.no_grad():
            out = model(torch.tensor([tt]).to(device))
        hs = [h.squeeze(0).cpu().numpy() for h in out.hidden_states]
        test_data.append((sent, tt, hs))
    total_unique = sum(len(set(tt[1:])) for _, tt, _ in test_data)

    for L in layers:
        print(f"\n{'='*78}")
        print(f"LAYER L{L}")
        print(f"{'='*78}")

        v = build_vocab_pos1(model, L, device, eot)
        avg_vocab_norm = float(np.mean(np.linalg.norm(v, axis=1)))
        print(f"  avg vocab norm: {avg_vocab_norm:.2f}")

        # All scaling strategies
        scales = {
            "scale=1.0":                 lambda mp, n_t, mp_norm: 1.0,
            "scale=vocab_avg":           lambda mp, n_t, mp_norm: avg_vocab_norm,
            "scale=vocab_avg*sqrt(N)":   lambda mp, n_t, mp_norm: avg_vocab_norm * np.sqrt(n_t),
            "scale=vocab_avg*N":         lambda mp, n_t, mp_norm: avg_vocab_norm * n_t,
            "scale=N*100 (original)":    lambda mp, n_t, mp_norm: n_t * 100.0,
            "scale=mp_norm (undo norm)": lambda mp, n_t, mp_norm: mp_norm,
            "scale=mp_norm*N":           lambda mp, n_t, mp_norm: mp_norm * n_t,
        }

        results_per_method = {}
        for label in scales:
            results_per_method[label] = 0
        results_per_method["topN cosine (scale-free)"] = 0
        results_per_method["sum (no normalize, baseline)"] = 0

        for sent, tt, hs_per_layer in test_data:
            trailing_tokens = tt[1:]
            unique = set(trailing_tokens)
            n_t = len(trailing_tokens)
            trailing = hs_per_layer[L][1:]

            mean_pooled = trailing.mean(axis=0)
            mp_norm = float(np.linalg.norm(mean_pooled))
            normalized = mean_pooled / max(mp_norm, 1e-12)

            # Greedy with each scale
            for label, scale_fn in scales.items():
                s = scale_fn(mean_pooled, n_t, mp_norm)
                target = normalized * s
                rec = decompose_greedy(target, v, n_t + 10)
                results_per_method[label] += len(unique & set(rec))

            # Scale-free top-N
            rec = topn_cosine(mean_pooled, v, n_t)
            results_per_method["topN cosine (scale-free)"] += len(unique & set(rec))

            # Baseline: no normalization at all (just sum)
            sum_target = trailing.sum(axis=0)
            rec = decompose_greedy(sum_target, v, n_t + 10)
            results_per_method["sum (no normalize, baseline)"] += len(unique & set(rec))

        # Print sorted
        sorted_results = sorted(results_per_method.items(), key=lambda x: -x[1])
        for label, hits in sorted_results:
            pct = 100 * hits / total_unique
            print(f"    {label:40s} {hits:>4d}/{total_unique:<4d}  {pct:>5.1f}%")

        del v
        gc.collect()


if __name__ == "__main__":
    main()
