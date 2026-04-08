#!/usr/bin/env python3
"""Linear-bridge sentence decomposition.

We have a 768x768 linear map W_L that takes the sink-corrected contextual
vocabulary at layer L to the static wte vocabulary, with 99.9% identity
preservation at L1 and 86.7% at L12.

By linearity:
  (sum of contextual_i) @ W = sum of (contextual_i @ W) ≈ sum of (static_i)

So in principle, mapping a sentence's sink-skipped contextual sum through W
should produce something approximating the static sum of those tokens, which
we can then decompose against wte using the static-decomposition methods that
already work well (greedy + coord descent).

Three things will limit this in practice:
  1. Per-token reconstruction error of W (~0.5 relative error, even at L1)
  2. Position drift: the hidden state of `cat at position 7` differs from
     the vocab entry `cat at position 1 of [EOT, cat]` (cos ~0.92 at L1)
  3. Per-token bias accumulation across positions

This experiment measures all of them by comparing:
  - Direct decomposition of contextual sum against wte (Part A baseline)
  - Bridged decomposition: (contextual sum @ W) decomposed against wte
  - Static upper bound: decomposing the actual wte[tokens].sum() against wte
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

    # Forward pass each test sentence once and grab all hidden states
    print("Forward-passing test sentences...")
    test_data = []  # list of (sent, tokens, hidden_states_per_layer)
    for sent in test_sentences:
        tt = [eot] + tokenizer.encode(sent)
        with torch.no_grad():
            out = model(torch.tensor([tt]).to(device))
        hs = [h.squeeze(0).cpu().numpy() for h in out.hidden_states]
        test_data.append((sent, tt, hs))

    # ==================================================================
    # Static upper bound: decompose the actual wte[tokens].sum()
    # ==================================================================
    print(f"\n{'='*78}")
    print("UPPER BOUND: decompose actual static sum (wte[tokens].sum()) against wte")
    print(f"{'='*78}")
    print(f"  {'Sentence':45s}  {'unique':>6s}  {'recovery':>10s}")
    print("  " + "-" * 70)

    for sent, tt, _ in test_data:
        trailing = tt[1:]
        unique = set(trailing)
        u = len(unique)
        n_t = len(trailing)
        target = wte[trailing].sum(axis=0)
        rec = decompose_greedy(target, wte, n_t + 10)
        hits = len(unique & set(rec))
        print(f"  {sent[:43]:45s}  {u:>6d}  {hits:>4d}/{u:<4d}")

    # ==================================================================
    # For each layer: train W, then test bridged decomposition
    # ==================================================================
    for L in layers:
        print(f"\n{'='*78}")
        print(f"LAYER L{L}: train W on (corrected_vocab, wte) pairs, test bridge")
        print(f"{'='*78}")

        print(f"  Building corrected vocab at L{L}...")
        v = build_vocab_pos1(model, L, device, eot)

        print(f"  Fitting W (768x768, lstsq)...")
        W, _, rank, _ = np.linalg.lstsq(v, wte, rcond=None)
        # Verify identity preservation on a small sample
        n_check = 1000
        rng = np.random.default_rng(42)
        check_ids = rng.choice(v.shape[0], n_check, replace=False)
        mapped = v[check_ids] @ W
        wte_norms = np.linalg.norm(wte, axis=1)
        wte_norms[wte_norms < 1e-10] = 1.0
        wte_normed = wte / wte_norms[:, None]
        correct = 0
        for i, tid in enumerate(check_ids):
            m = mapped[i]
            mn = np.linalg.norm(m)
            if mn < 1e-10:
                continue
            sims = wte_normed @ (m / mn)
            if int(np.argmax(sims)) == int(tid):
                correct += 1
        print(f"  W identity preservation (sample {n_check}): {correct/n_check*100:.1f}%")

        # Now test sentence bridging
        print()
        print(f"  {'Sentence':45s}  {'unique':>6s}  {'direct':>9s}  {'bridged':>9s}  bridged tokens")
        print("  " + "-" * 110)

        for sent, tt, hs_per_layer in test_data:
            trailing_tokens = tt[1:]
            unique = set(trailing_tokens)
            u = len(unique)
            n_t = len(trailing_tokens)

            # Sink-skipped contextual sum at this layer
            ctx_sum = hs_per_layer[L][1:].sum(axis=0)

            # Direct decomposition against wte (Part A repeat)
            direct_rec = decompose_greedy(ctx_sum, wte, n_t + 10)
            direct_hits = len(unique & set(direct_rec))

            # Bridged: map through W, then decompose against wte
            bridged = ctx_sum @ W
            bridged_rec = decompose_greedy(bridged, wte, n_t + 10)
            bridged_hits = len(unique & set(bridged_rec))
            bridged_toks = [tokenizer.decode([r]) for r in bridged_rec[:8]]

            print(f"  {sent[:43]:45s}  {u:>6d}  {direct_hits:>4d}/{u:<4d}    {bridged_hits:>4d}/{u:<4d}    {bridged_toks}")

        del v, W, mapped
        gc.collect()


if __name__ == "__main__":
    main()
