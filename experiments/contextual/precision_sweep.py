#!/usr/bin/env python3
"""Precision sweep: contextual decomposition recovery as a function of
the quantization level applied to BOTH the target and the vocabulary
(decompose-then-debias method).

The question: int8 gave a small improvement over f32. Is int8 the peak,
or is the peak at finer precision (f16, int10, int12) or coarser precision
(int6, int4)? A bell shape would suggest there's a 'noise filtering' sweet
spot somewhere in the precision range.

Quantization levels tested:
  - f32 (baseline, no quantization)
  - f16 (half-precision float)
  - int N for N in {12, 10, 8, 6, 4, 2, 1}  (per-vector symmetric)

Both target and vocab are quantized at the same level. Per-token bias is
computed in f32 and subtracted before quantization.
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


def quantize_int_n(x, bits):
    """Per-vector symmetric int-N quantization with `bits` levels of magnitude."""
    if bits >= 32:
        return x.copy()
    levels = (1 << (bits - 1)) - 1  # e.g. bits=8 -> 127
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        scale = np.max(np.abs(x)) / max(levels, 1)
        if scale < 1e-12:
            return x.copy()
        return np.round(x / scale).clip(-levels - 1, levels) * scale
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        out[i] = quantize_int_n(x[i], bits)
    return out


def quantize_f16(x):
    return x.astype(np.float16).astype(np.float32)


def quantize(x, mode):
    if mode == "f32":
        return x.copy()
    if mode == "f16":
        return quantize_f16(x)
    if mode.startswith("int"):
        bits = int(mode[3:])
        return quantize_int_n(x, bits)
    raise ValueError(mode)


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

    modes = ["f32", "f16", "int12", "int10", "int8", "int6", "int4", "int2", "int1"]

    for L in layers:
        print(f"\n{'='*78}")
        print(f"LAYER L{L}")
        print(f"{'='*78}")

        v_f32 = build_vocab_pos1(model, L, device, eot)
        bias_accum = []
        for rt, rh in zip(ref_token_lists, ref_hs_per_layer[L]):
            trailing = rh[1:]
            n_r = trailing.shape[0]
            bias_accum.append((trailing.sum(axis=0) - v_f32[rt[1:]].sum(axis=0)) / n_r)
        bias = np.mean(bias_accum, axis=0)
        print(f"  Per-token bias norm: {np.linalg.norm(bias):.2f}")

        # Pre-compute test target debiased sums (in f32) for each sentence
        target_debiased = []
        for sent, tt, hs_per_layer in test_data:
            n_t = len(tt) - 1
            tg = hs_per_layer[L][1:].sum(axis=0) - n_t * bias
            target_debiased.append((sent, tt, n_t, tg))

        # For each mode: quantize vocab once, then quantize each target and decompose
        print(f"\n  {'mode':>6s}  {'recovery':>12s}  {'%':>6s}  notes")
        print("  " + "-" * 50)

        for mode in modes:
            v_q = quantize(v_f32, mode)

            total_hits = 0
            total_unique = 0
            for sent, tt, n_t, tg in target_debiased:
                tg_q = quantize(tg, mode)
                rec = decompose_greedy(tg_q, v_q, n_t + 10)
                unique = set(tt[1:])
                u = len(unique)
                hits = len(unique & set(rec))
                total_hits += hits
                total_unique += u

            pct = 100 * total_hits / total_unique
            print(f"  {mode:>6s}  {total_hits:>4d}/{total_unique:<5d}  {pct:>5.1f}%")

            del v_q
            gc.collect()

        del v_f32
        gc.collect()


if __name__ == "__main__":
    main()
