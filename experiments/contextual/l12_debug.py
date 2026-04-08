#!/usr/bin/env python3
"""Debug L12 contextual decomposition more carefully.

l12_contextual.py reported 0/37 across all sentences with sink-skip + N-debias.
But vocab_sanity.py showed the L12 vocab has clean semantic clustering. These
should not both be true unless I am doing something wrong.

Possibilities to check:
  1. Maybe sum is the wrong reduction at L12 (LayerNorm normalizes per-vector,
     so summing N normalized vectors loses something that summing pre-norm
     vectors does not). Try mean instead.
  2. Maybe the bias computation is wrong at L12 specifically.
  3. Maybe there is a sign or normalization bug I haven't noticed.
  4. Maybe the issue is that at L12 the search needs to be against a normalized
     vocab (since vocab entries are themselves ln_f'd and have similar norms,
     cosine vs L2 vs IP could matter).

Try several variations.
"""
import gc
import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer


def decompose_greedy(target, embeddings, max_steps, metric="cosine"):
    residual = target.copy()
    recovered = []
    prev_norm = float("inf")
    for _ in range(max_steps):
        r_norm = np.linalg.norm(residual)
        if r_norm < 0.001 or r_norm > prev_norm:
            break
        prev_norm = r_norm
        if metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1)
            norms[norms < 1e-10] = 1.0
            sims = embeddings @ residual / (norms * r_norm)
            best = int(np.argmax(sims))
        elif metric == "ip":
            best = int(np.argmax(embeddings @ residual))
        elif metric == "l2":
            best = int(np.argmin(np.sum((embeddings - residual) ** 2, axis=1)))
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
    L = 12
    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True).to(device)
    model.eval()
    eot = tokenizer.eos_token_id

    print(f"Building corrected vocab at L{L}...")
    vocab = build_vocab_pos1(model, L, device, eot)
    vocab_norms = np.linalg.norm(vocab, axis=1)
    print(f"  vocab shape: {vocab.shape}")
    print(f"  vocab norms: avg {np.mean(vocab_norms):.2f}, std {np.std(vocab_norms):.2f}, "
          f"min {np.min(vocab_norms):.2f}, max {np.max(vocab_norms):.2f}")

    # Sanity check: nearest neighbors in the vocab for ' cat'
    cat_id = tokenizer.encode(" cat")[0]
    q = vocab[cat_id]
    sims = vocab @ q / (vocab_norms * np.linalg.norm(q))
    sims[cat_id] = -np.inf
    top = np.argsort(-sims)[:6]
    nbrs = [(tokenizer.decode([int(t)]), float(sims[t])) for t in top]
    print(f"\n  Sanity: nearest neighbors of ' cat' in L12 vocab:")
    for n, s in nbrs:
        print(f"    {n!r}: {s:.4f}")

    refs = [
        "the quick brown fox jumps over the lazy dog",
        "she went to the store to buy some food",
        "he ran down the long road to the old house",
        "they played in the park with the children",
    ]

    sentences = [
        "the cat sat on the mat",
        "Mary had a little lamb",
        "the dog ran in the park",
    ]

    # ----------------------------------------------------------------
    # Compute bias the standard way
    # ----------------------------------------------------------------
    bias_accum = []
    for ref in refs:
        rt = tokenizer.encode(ref)
        with torch.no_grad():
            rh = model(torch.tensor([rt]).to(device)).hidden_states[L].squeeze(0).cpu().numpy()
        rt_trailing = rh[1:]
        n_r = rt_trailing.shape[0]
        bias_accum.append((rt_trailing.sum(axis=0) - vocab[rt[1:]].sum(axis=0)) / n_r)
    bias = np.mean(bias_accum, axis=0)
    print(f"\nPer-token sum-bias norm at L12: {np.linalg.norm(bias):.2f}")

    # Mean-bias: average of per-position differences (not sum of differences / N)
    # This should equal bias above mathematically. Just verifying.
    mean_bias_accum = []
    for ref in refs:
        rt = tokenizer.encode(ref)
        with torch.no_grad():
            rh = model(torch.tensor([rt]).to(device)).hidden_states[L].squeeze(0).cpu().numpy()
        rt_trailing = rh[1:]
        # Per-position bias: hidden[i] - vocab[token_i]
        per_pos_bias = rt_trailing - vocab[rt[1:]]
        mean_bias_accum.append(per_pos_bias.mean(axis=0))
    mean_bias = np.mean(mean_bias_accum, axis=0)
    print(f"Per-position-then-mean bias norm at L12: {np.linalg.norm(mean_bias):.2f}")
    print(f"Cosine between the two bias estimates: "
          f"{float(np.dot(bias, mean_bias) / (np.linalg.norm(bias) * np.linalg.norm(mean_bias))):.6f}")

    # ----------------------------------------------------------------
    # Test 1: Sum-based decomposition (current method)
    # ----------------------------------------------------------------
    print(f"\n{'='*78}")
    print("TEST 1: Sum-based decomposition (sum trailing positions, debias by N*bias)")
    print(f"{'='*78}")
    print(f"  {'Sentence':40s}  {'metric':>8s}  {'method':>15s}  recovery")

    for sent in sentences:
        tt = [eot] + tokenizer.encode(sent)
        with torch.no_grad():
            hs = model(torch.tensor([tt]).to(device)).hidden_states[L].squeeze(0).cpu().numpy()
        trailing_target = hs[1:].sum(axis=0)
        trailing_tokens = tt[1:]
        unique = set(trailing_tokens)
        u = len(unique)
        n_t = len(trailing_tokens)

        for metric in ["cosine", "ip", "l2"]:
            for label, target in [
                ("no_debias", trailing_target),
                ("N*bias", trailing_target - n_t * bias),
            ]:
                rec = decompose_greedy(target, vocab, n_t + 10, metric)
                hits = len(unique & set(rec))
                if hits > 0:
                    toks = [tokenizer.decode([r]) for r in rec[:6]]
                    print(f"  {sent[:38]:40s}  {metric:>8s}  {label:>15s}  {hits}/{u}  {toks}")
                else:
                    print(f"  {sent[:38]:40s}  {metric:>8s}  {label:>15s}  {hits}/{u}")

    # ----------------------------------------------------------------
    # Test 2: Mean-based decomposition (mean trailing positions, no N scaling)
    # ----------------------------------------------------------------
    print(f"\n{'='*78}")
    print("TEST 2: Mean-based decomposition (mean trailing positions, debias by 1*bias)")
    print(f"{'='*78}")
    print(f"  {'Sentence':40s}  {'metric':>8s}  {'method':>15s}  recovery")

    for sent in sentences:
        tt = [eot] + tokenizer.encode(sent)
        with torch.no_grad():
            hs = model(torch.tensor([tt]).to(device)).hidden_states[L].squeeze(0).cpu().numpy()
        trailing_target = hs[1:].mean(axis=0)
        trailing_tokens = tt[1:]
        unique = set(trailing_tokens)
        u = len(unique)
        n_t = len(trailing_tokens)

        for metric in ["cosine", "ip", "l2"]:
            for label, target in [
                ("no_debias", trailing_target),
                ("1*bias", trailing_target - bias),
            ]:
                rec = decompose_greedy(target, vocab, n_t + 10, metric)
                hits = len(unique & set(rec))
                if hits > 0:
                    toks = [tokenizer.decode([r]) for r in rec[:6]]
                    print(f"  {sent[:38]:40s}  {metric:>8s}  {label:>15s}  {hits}/{u}  {toks}")
                else:
                    print(f"  {sent[:38]:40s}  {metric:>8s}  {label:>15s}  {hits}/{u}")

    # ----------------------------------------------------------------
    # Test 3: Decompose individual positions, see if any are recoverable at all
    # ----------------------------------------------------------------
    print(f"\n{'='*78}")
    print("TEST 3: Decompose each individual position (cosine to vocab, top-1 hit)")
    print(f"{'='*78}")
    print(f"  Tests whether individual L12 hidden states resemble their token's vocab entry.")
    print()

    for sent in sentences:
        tt = [eot] + tokenizer.encode(sent)
        with torch.no_grad():
            hs = model(torch.tensor([tt]).to(device)).hidden_states[L].squeeze(0).cpu().numpy()
        print(f"  {sent!r}")
        for i in range(1, len(tt)):
            target_token = tt[i]
            target_str = tokenizer.decode([target_token])
            h_i = hs[i]
            h_norm = np.linalg.norm(h_i)
            sims = vocab @ h_i / (vocab_norms * h_norm)
            top = np.argsort(-sims)[:5]
            top_strs = [tokenizer.decode([int(t)]) for t in top]
            target_rank = int(np.sum(sims > sims[target_token]))
            print(f"    pos {i} (token {target_str!r}, true rank {target_rank:>5d}): top-5 = {top_strs}")


if __name__ == "__main__":
    main()
