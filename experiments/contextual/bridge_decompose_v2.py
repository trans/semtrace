#!/usr/bin/env python3
"""Bridge decomposition with per-token bias subtracted before applying W.

Variation A: instead of `ctx_sum @ W`, do `(ctx_sum - N * bias[L]) @ W`.
The bias subtraction removes the position-shift drift and per-token bias
content; W then only has to do the contextual→static rotation on a cleaner
input.

We also try a second variation:
  - Variation B: train W on bias-subtracted vocab pairs:
      (v - bias[L]) @ W' ≈ wte
    Then apply W' to (ctx_sum - N * bias[L]).

Both should be at least as good as the unbiased bridge from bridge_decompose.py
and might be substantially better.
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

    print("Forward-passing reference sentences (for bias)...")
    ref_hs_per_layer = {}
    for L in layers:
        ref_hs_per_layer[L] = []
    for rt in ref_token_lists:
        with torch.no_grad():
            out = model(torch.tensor([rt]).to(device))
        for L in layers:
            ref_hs_per_layer[L].append(out.hidden_states[L].squeeze(0).cpu().numpy())

    for L in layers:
        print(f"\n{'='*78}")
        print(f"LAYER L{L}")
        print(f"{'='*78}")

        print(f"  Building corrected vocab at L{L}...")
        v = build_vocab_pos1(model, L, device, eot)

        # Compute per-token bias from refs (sink-skipped)
        bias_accum = []
        for rt, rh in zip(ref_token_lists, ref_hs_per_layer[L]):
            trailing = rh[1:]
            n_r = trailing.shape[0]
            bias_accum.append((trailing.sum(axis=0) - v[rt[1:]].sum(axis=0)) / n_r)
        bias = np.mean(bias_accum, axis=0)
        print(f"  Per-token bias norm: {np.linalg.norm(bias):.2f}")

        # Variation A: standard W (no bias in training), bias-subtracted input at apply time
        print(f"  Fitting W_A (standard, on raw vocab)...")
        W_A, _, _, _ = np.linalg.lstsq(v, wte, rcond=None)

        # Variation B: W trained on bias-subtracted pairs
        # We need a "per-token bias" for individual vocab entries — but bias is
        # defined as a sum-level quantity. The natural single-token analog is
        # `bias` itself (the per-token offset). So debiased_v[t] = v[t] - bias.
        print(f"  Fitting W_B (on bias-subtracted vocab)...")
        v_minus_bias = v - bias  # broadcast: subtract bias from every row
        W_B, _, _, _ = np.linalg.lstsq(v_minus_bias, wte, rcond=None)

        print()
        print(f"  {'Sentence':45s}  {'unique':>6s}  {'unbiased':>9s}  {'A:bias→W':>10s}  {'B:both':>8s}  best tokens")
        print("  " + "-" * 115)

        for sent, tt, hs_per_layer in test_data:
            trailing_tokens = tt[1:]
            unique = set(trailing_tokens)
            u = len(unique)
            n_t = len(trailing_tokens)

            ctx_sum = hs_per_layer[L][1:].sum(axis=0)

            # Unbiased bridge (the v1 result)
            unbiased = ctx_sum @ W_A
            rec_un = decompose_greedy(unbiased, wte, n_t + 10)
            hits_un = len(unique & set(rec_un))

            # Variation A: subtract bias, then apply W_A
            biased_A = (ctx_sum - n_t * bias) @ W_A
            rec_A = decompose_greedy(biased_A, wte, n_t + 10)
            hits_A = len(unique & set(rec_A))

            # Variation B: subtract bias, then apply W_B (which was trained on debiased pairs)
            biased_B = (ctx_sum - n_t * bias) @ W_B
            rec_B = decompose_greedy(biased_B, wte, n_t + 10)
            hits_B = len(unique & set(rec_B))

            best_toks = [tokenizer.decode([r]) for r in rec_B[:8]]
            print(f"  {sent[:43]:45s}  {u:>6d}  {hits_un:>4d}/{u:<4d}    {hits_A:>4d}/{u:<4d}     {hits_B:>4d}/{u:<4d}   {best_toks}")

        del v, v_minus_bias, W_A, W_B
        gc.collect()


if __name__ == "__main__":
    main()
