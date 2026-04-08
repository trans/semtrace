#!/usr/bin/env python3
"""Bit-band decomposition on sink-skipped contextual data.

The original bit-band experiment (way back) was on sink-contaminated hidden
states. We found that at L11, the high band (f32 - int8 residual) carried a
small persistent ~2/6 signal while the low band collapsed to 0/6 due to norm
outliers in the sink-corrupted vocabulary.

After the sink discovery, we never re-ran this. With sink-skip + N-debias the
representations are clean, the vocabulary is built correctly, and the question
"where in precision space does the per-token signal live" can finally be asked
without the sink dominating both bands.

Method per layer:
  1. Build sink-corrected vocab (one entry per token, position 1 of [EOT, t])
  2. Compute per-token bias from reference sentences (sink-skipped)
  3. For each test sentence:
     a. Forward-pass with EOT prepend
     b. Sum trailing positions to get target
     c. Subtract N * bias to get debiased target
     d. int8-quantize the debiased target -> low_target; remainder -> high_target
     e. int8-quantize the vocab -> low_vocab; remainder -> high_vocab
     f. Decompose: full+debias / low+debias / high+debias against matching vocab
  4. Report hits per band, per sentence, per layer
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
    """Per-vector symmetric int8 quantization. Returns dequantized approx."""
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

    sentences = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "Mary had a little lamb",
        "Four score and seven years ago",
        "I love cats and dogs",
        "the big red car drove fast down the road",
    ]
    layers = [1, 6, 11]
    refs = [
        "the quick brown fox jumps over the lazy dog",
        "she went to the store to buy some food",
        "he ran down the long road to the old house",
        "they played in the park with the children",
    ]

    # Pre-compute reference token lists and forward passes
    ref_token_lists = [tokenizer.encode(r) for r in refs]
    ref_hs_per_layer = {L: [] for L in layers}
    for rt in ref_token_lists:
        with torch.no_grad():
            out = model(torch.tensor([rt]).to(device))
        for L in layers:
            ref_hs_per_layer[L].append(out.hidden_states[L].squeeze(0).cpu().numpy())

    # Pre-compute test sentence forward passes
    print("Forward-passing test sentences...")
    test_data = []
    for sent in sentences:
        tt = [eot] + tokenizer.encode(sent)
        with torch.no_grad():
            out = model(torch.tensor([tt]).to(device))
        hs = [h.squeeze(0).cpu().numpy() for h in out.hidden_states]
        test_data.append((sent, tt, hs))

    for L in layers:
        print(f"\n{'='*78}")
        print(f"LAYER L{L}")
        print(f"{'='*78}")

        print(f"  Building corrected vocab at L{L}...")
        v = build_vocab_pos1(model, L, device, eot)

        # Compute clean per-token bias
        bias_accum = []
        for rt, rh in zip(ref_token_lists, ref_hs_per_layer[L]):
            trailing = rh[1:]
            n_r = trailing.shape[0]
            bias_accum.append((trailing.sum(axis=0) - v[rt[1:]].sum(axis=0)) / n_r)
        bias = np.mean(bias_accum, axis=0)
        print(f"  Per-token bias norm: {np.linalg.norm(bias):.2f}")

        # Bit-band split the vocabulary
        v_low = int8_quantize(v)
        v_high = v - v_low
        print(f"  Vocab norms: full {np.mean(np.linalg.norm(v, axis=1)):.2f}, "
              f"low {np.mean(np.linalg.norm(v_low, axis=1)):.2f}, "
              f"high {np.mean(np.linalg.norm(v_high, axis=1)):.4f}")

        print()
        print(f"  {'Sentence':45s}  {'unique':>6s}  {'full':>6s}  {'low':>6s}  {'high':>6s}  {'union':>6s}  bands_high_tokens")
        print("  " + "-" * 130)

        for sent, tt, hs_per_layer in test_data:
            trailing_tokens = tt[1:]
            unique = set(trailing_tokens)
            u = len(unique)
            n_t = len(trailing_tokens)

            # Sum and debias the target (full precision)
            target_full = hs_per_layer[L][1:].sum(axis=0)
            debiased_full = target_full - n_t * bias

            # Bit-band split the debiased target
            debiased_low = int8_quantize(debiased_full)
            debiased_high = debiased_full - debiased_low

            # Decompose against matching bands
            rec_full = decompose_greedy(debiased_full, v, n_t + 10)
            hits_full = len(unique & set(rec_full))

            rec_low = decompose_greedy(debiased_low, v_low, n_t + 10)
            hits_low = len(unique & set(rec_low))

            rec_high = decompose_greedy(debiased_high, v_high, n_t + 10)
            hits_high = len(unique & set(rec_high))

            # Union of all bands
            union = set(rec_full) | set(rec_low) | set(rec_high)
            hits_union = len(unique & union)

            high_toks = [tokenizer.decode([r]) for r in rec_high[:8]]
            print(f"  {sent[:43]:45s}  {u:>6d}  {hits_full:>3d}/{u:<2d}  {hits_low:>3d}/{u:<2d}  {hits_high:>3d}/{u:<2d}  {hits_union:>3d}/{u:<2d}   {high_toks}")

        del v, v_low, v_high
        gc.collect()


if __name__ == "__main__":
    main()
