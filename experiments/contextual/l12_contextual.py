#!/usr/bin/env python3
"""Plain contextual decomposition at L12.

Same method as L1-L11 sink-skip + N-debias, but at L12 (post-ln_f).
We have not tested this layer with this method anywhere else. The L12
representation is the most threat-model-relevant since it is the closest
thing to what an embedding endpoint returns.
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
    L = 12
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
    refs = [
        "the quick brown fox jumps over the lazy dog",
        "she went to the store to buy some food",
        "he ran down the long road to the old house",
        "they played in the park with the children",
    ]

    print(f"Building corrected vocab at L{L} (post-ln_f)...")
    vocab = build_vocab_pos1(model, L, device, eot)
    print(f"  vocab shape: {vocab.shape}, avg norm: {np.mean(np.linalg.norm(vocab, axis=1)):.2f}")

    # Per-token bias from refs
    bias_accum = []
    for ref in refs:
        rt = tokenizer.encode(ref)
        with torch.no_grad():
            rh = model(torch.tensor([rt]).to(device)).hidden_states[L].squeeze(0).cpu().numpy()
        rt_trailing = rh[1:]
        n_r = rt_trailing.shape[0]
        bias_accum.append((rt_trailing.sum(axis=0) - vocab[rt[1:]].sum(axis=0)) / n_r)
    bias = np.mean(bias_accum, axis=0)
    print(f"  Per-token bias norm: {np.linalg.norm(bias):.2f}")

    print(f"\n{'Sentence':45s}  {'unique':>6s}  {'no debias':>11s}  {'N-debias':>10s}  tokens (debiased)")
    print("-" * 130)

    for sent in sentences:
        tt = [eot] + tokenizer.encode(sent)
        with torch.no_grad():
            hs = model(torch.tensor([tt]).to(device)).hidden_states[L].squeeze(0).cpu().numpy()
        trailing_target = hs[1:].sum(axis=0)
        trailing_tokens = tt[1:]
        unique = set(trailing_tokens)
        u = len(unique)
        n_t = len(trailing_tokens)

        # Without debias
        rec_nodeb = decompose_greedy(trailing_target, vocab, n_t + 10)
        h_nodeb = len(unique & set(rec_nodeb))

        # With N-debias
        debiased = trailing_target - n_t * bias
        rec_deb = decompose_greedy(debiased, vocab, n_t + 10)
        h_deb = len(unique & set(rec_deb))
        toks = [tokenizer.decode([r]) for r in rec_deb[:8]]

        print(f"  {sent[:43]:45s}  {u:>6d}  {h_nodeb:>5d}/{u:<4d}    {h_deb:>5d}/{u:<4d}    {toks}")

    del vocab
    gc.collect()


if __name__ == "__main__":
    main()
