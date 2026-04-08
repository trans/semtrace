#!/usr/bin/env python3
"""Verified bit-band decomposition test on sink-skipped data.

Two methodological issues with bands_post_sink.py I'm correcting:

  1. I tested only one split order (debias-then-split). The original
     pre-sink experiment did split-then-no-debias. To compare apples to
     apples I need both.

  2. I tested 6 sentences and called the small low-vs-full difference
     "noise" without actually checking. Need more sentences AND a
     per-sentence breakdown to see if the small gain is consistent.

This script tests on 20 sentences across L1, L6, L11 with three variants:

  V_raw   = no debias, split target into low/high directly (matches original)
  V_deb   = subtract N*bias from target FIRST, then split (what I did before)
  V_full  = full vector with N*bias debias and no banding (the standard method)

For each variant we report per-sentence hits and aggregate stats. We also
flag whether the low band consistently beats / matches / loses to the full
method, so we can tell signal from noise.
"""
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


def main():
    device = "cpu"
    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True).to(device)
    model.eval()
    eot = tokenizer.eos_token_id

    layers = [1, 6, 11]
    refs = [
        "the quick brown fox jumps over the lazy dog",
        "she went to the store to buy some food",
        "he ran down the long road to the old house",
        "they played in the park with the children",
    ]
    ref_token_lists = [tokenizer.encode(r) for r in refs]

    # Pre-compute reference forward passes
    print(f"Forward-passing {len(refs)} reference sentences and {len(SENTENCES)} test sentences...")
    ref_hs_per_layer = {L: [] for L in layers}
    for rt in ref_token_lists:
        with torch.no_grad():
            out = model(torch.tensor([rt]).to(device))
        for L in layers:
            ref_hs_per_layer[L].append(out.hidden_states[L].squeeze(0).cpu().numpy())

    test_data = []
    for sent in SENTENCES:
        tt = [eot] + tokenizer.encode(sent)
        with torch.no_grad():
            out = model(torch.tensor([tt]).to(device))
        hs = [h.squeeze(0).cpu().numpy() for h in out.hidden_states]
        test_data.append((sent, tt, hs))

    for L in layers:
        print(f"\n{'='*78}")
        print(f"LAYER L{L}")
        print(f"{'='*78}")

        v = build_vocab_pos1(model, L, device, eot)
        v_low = int8_quantize(v)
        v_high = v - v_low

        bias_accum = []
        for rt, rh in zip(ref_token_lists, ref_hs_per_layer[L]):
            trailing = rh[1:]
            n_r = trailing.shape[0]
            bias_accum.append((trailing.sum(axis=0) - v[rt[1:]].sum(axis=0)) / n_r)
        bias = np.mean(bias_accum, axis=0)
        print(f"  Per-token bias norm: {np.linalg.norm(bias):.2f}")
        print(f"  Vocab norms: full {np.mean(np.linalg.norm(v, axis=1)):.1f}, "
              f"low {np.mean(np.linalg.norm(v_low, axis=1)):.1f}, "
              f"high {np.mean(np.linalg.norm(v_high, axis=1)):.3f}")
        print()

        # Aggregate counters
        agg = {
            'full_deb':       0,  # full vector + N*bias debias
            'raw_low':        0,  # split before debias, low band raw
            'raw_high':       0,  # split before debias, high band raw
            'deb_low':        0,  # debias first, then split, low band
            'deb_high':       0,  # debias first, then split, high band
            'unique_total':   0,
        }
        # Per-sentence wins for low vs full
        wins = {'low_beats_full': 0, 'low_equals_full': 0, 'low_loses_full': 0}

        print(f"  {'Sentence':45s}  {'u':>3s}  {'fullD':>5s}  {'rawL':>5s}  {'rawH':>5s}  {'debL':>5s}  {'debH':>5s}")
        print("  " + "-" * 90)

        for sent, tt, hs_per_layer in test_data:
            trailing_tokens = tt[1:]
            unique = set(trailing_tokens)
            u = len(unique)
            n_t = len(trailing_tokens)
            agg['unique_total'] += u

            target_full = hs_per_layer[L][1:].sum(axis=0)

            # Variant 1: full vector + N*bias debias (the standard method)
            full_deb = target_full - n_t * bias
            rec = decompose_greedy(full_deb, v, n_t + 10)
            hits_full = len(unique & set(rec))
            agg['full_deb'] += hits_full

            # Variant 2: split RAW target (no debias yet), decompose against banded vocab
            raw_low = int8_quantize(target_full)
            raw_high = target_full - raw_low
            rec_rL = decompose_greedy(raw_low, v_low, n_t + 10)
            rec_rH = decompose_greedy(raw_high, v_high, n_t + 10)
            hits_rL = len(unique & set(rec_rL))
            hits_rH = len(unique & set(rec_rH))
            agg['raw_low'] += hits_rL
            agg['raw_high'] += hits_rH

            # Variant 3: debias first, then split (what I did before)
            deb_low = int8_quantize(full_deb)
            deb_high = full_deb - deb_low
            rec_dL = decompose_greedy(deb_low, v_low, n_t + 10)
            rec_dH = decompose_greedy(deb_high, v_high, n_t + 10)
            hits_dL = len(unique & set(rec_dL))
            hits_dH = len(unique & set(rec_dH))
            agg['deb_low'] += hits_dL
            agg['deb_high'] += hits_dH

            # Track low-vs-full (debiased low band, since that's the question)
            if hits_dL > hits_full:
                wins['low_beats_full'] += 1
            elif hits_dL == hits_full:
                wins['low_equals_full'] += 1
            else:
                wins['low_loses_full'] += 1

            print(f"  {sent[:43]:45s}  {u:>3d}  {hits_full:>3d}/{u:<2d} {hits_rL:>3d}/{u:<2d} {hits_rH:>3d}/{u:<2d} {hits_dL:>3d}/{u:<2d} {hits_dH:>3d}/{u:<2d}")

        # Aggregate report
        u_total = agg['unique_total']
        print()
        print(f"  AGGREGATE (over {len(SENTENCES)} sentences, {u_total} unique tokens total):")
        print(f"    full + N*debias    : {agg['full_deb']:>3d}/{u_total} ({100*agg['full_deb']/u_total:.1f}%)")
        print(f"    raw split, low band: {agg['raw_low']:>3d}/{u_total} ({100*agg['raw_low']/u_total:.1f}%)")
        print(f"    raw split, high    : {agg['raw_high']:>3d}/{u_total} ({100*agg['raw_high']/u_total:.1f}%)")
        print(f"    deb-then-split, low: {agg['deb_low']:>3d}/{u_total} ({100*agg['deb_low']/u_total:.1f}%)")
        print(f"    deb-then-split,high: {agg['deb_high']:>3d}/{u_total} ({100*agg['deb_high']/u_total:.1f}%)")
        print(f"    low-vs-full per sentence: low_beats={wins['low_beats_full']}, low_equals={wins['low_equals_full']}, low_loses={wins['low_loses_full']}")

        del v, v_low, v_high
        gc.collect()


if __name__ == "__main__":
    main()
