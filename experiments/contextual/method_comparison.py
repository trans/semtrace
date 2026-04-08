#!/usr/bin/env python3
"""All methods, one test set. Apples-to-apples comparison.

Runs every contextual decomposition variant we have on the same 20-sentence
test set and reports recovery as a single comparison table.

Methods:
  S1.   Static upper bound: decompose actual wte[tokens].sum against wte
  S2.   Static greedy on contextual sum (no debias) against wte
  S3.   Static + bias-subtract: (ctx_sum - N*bias) against wte
  C(L). Contextual decomposition: (ctx_sum - N*bias) against ctx_vocab[L]
  CL(L).Contextual int8 low band: same as C(L) but quantize-then-decompose
  B(L). Bridge: (ctx_sum @ W[L]) decomposed against wte
  BD(L).Bridge with bias subtract: ((ctx_sum - N*bias) @ W[L]) against wte
  BDC(L).Bridge with bias subtract + coord descent on top

For each method and each layer of interest, report total recovery across
all 20 sentences.
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


def coord_descent(target, vocab, n_tokens, max_iters=15):
    current = decompose_greedy(target, vocab, n_tokens)
    while len(current) < n_tokens:
        ideal = target - vocab[current].sum(axis=0)
        n = np.linalg.norm(vocab, axis=1)
        n[n < 1e-10] = 1.0
        sims = vocab @ ideal / n
        current.append(int(np.argmax(sims)))
    current = current[:n_tokens]

    best_dist = np.linalg.norm(target - vocab[current].sum(axis=0))
    for _ in range(max_iters):
        improved = False
        for pos in range(n_tokens):
            others = [current[j] for j in range(n_tokens) if j != pos]
            others_sum = vocab[others].sum(axis=0)
            ideal = target - others_sum
            n = np.linalg.norm(vocab, axis=1)
            n[n < 1e-10] = 1.0
            sims = vocab @ ideal / n
            best_for_pos = int(np.argmax(sims))
            if best_for_pos != current[pos]:
                new_tokens = list(current)
                new_tokens[pos] = best_for_pos
                new_dist = np.linalg.norm(target - vocab[new_tokens].sum(axis=0))
                if new_dist < best_dist:
                    current[pos] = best_for_pos
                    best_dist = new_dist
                    improved = True
        if not improved:
            break
    return current


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


def main():
    device = "cpu"
    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True).to(device)
    model.eval()
    eot = tokenizer.eos_token_id

    wte = model.wte.weight.detach().cpu().numpy()

    layers = [1, 6, 11, 12]
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

    total_unique = sum(len(set(tt[1:])) for _, tt, _ in test_data)
    print(f"\nTest set: {len(SENTENCES)} sentences, {total_unique} unique tokens total")

    results = {}  # method_name -> total hits

    # ----------------------------------------------------------------
    # S1. Static upper bound: decompose actual wte[tokens].sum
    # ----------------------------------------------------------------
    hits = 0
    for sent, tt, _ in test_data:
        trailing = tt[1:]
        unique = set(trailing)
        target = wte[trailing].sum(axis=0)
        rec = decompose_greedy(target, wte, len(trailing) + 10)
        hits += len(unique & set(rec))
    results["S1: static (upper bound, greedy)"] = hits

    # S1+CD
    hits = 0
    for sent, tt, _ in test_data:
        trailing = tt[1:]
        unique = set(trailing)
        target = wte[trailing].sum(axis=0)
        rec = coord_descent(target, wte, len(trailing))
        hits += len(unique & set(rec))
    results["S1+CD: static (upper bound, coord descent)"] = hits

    # Per-layer methods
    for L in layers:
        print(f"\n  Building vocab + bias + W at L{L}...")
        v = build_vocab_pos1(model, L, device, eot)
        v_low = int8_quantize(v)

        bias_accum = []
        for rt, rh in zip(ref_token_lists, ref_hs_per_layer[L]):
            trailing = rh[1:]
            n_r = trailing.shape[0]
            bias_accum.append((trailing.sum(axis=0) - v[rt[1:]].sum(axis=0)) / n_r)
        bias = np.mean(bias_accum, axis=0)

        # Bridge from contextual at L to static
        W, _, _, _ = np.linalg.lstsq(v, wte, rcond=None)

        # Static-target bias: difference between contextual sum and static sum
        # used for the static-with-bias-subtract method
        static_bias_accum = []
        for rt, rh in zip(ref_token_lists, ref_hs_per_layer[L]):
            trailing = rh[1:]
            n_r = trailing.shape[0]
            static_sum = wte[rt[1:]].sum(axis=0)
            static_bias_accum.append((trailing.sum(axis=0) - static_sum) / n_r)
        static_bias = np.mean(static_bias_accum, axis=0)

        # Per-method counters
        m = {
            f"S2 L{L}: ctx_sum vs wte (no debias)":            0,
            f"S3 L{L}: (ctx_sum - N*static_bias) vs wte":      0,
            f"C  L{L}: (ctx_sum - N*bias) vs ctx_vocab":       0,
            f"CL L{L}: int8 low band (debias→split)":          0,
            f"B  L{L}: ctx_sum @ W vs wte":                    0,
            f"BD L{L}: (ctx_sum - N*bias) @ W vs wte":         0,
            f"BDC L{L}: BD + coord descent":                   0,
        }

        for sent, tt, hs_per_layer in test_data:
            trailing_tokens = tt[1:]
            unique = set(trailing_tokens)
            n_t = len(trailing_tokens)

            ctx_sum = hs_per_layer[L][1:].sum(axis=0)

            # S2: ctx_sum vs wte, no debias
            rec = decompose_greedy(ctx_sum, wte, n_t + 10)
            m[f"S2 L{L}: ctx_sum vs wte (no debias)"] += len(unique & set(rec))

            # S3: ctx_sum - N*static_bias vs wte
            tg = ctx_sum - n_t * static_bias
            rec = decompose_greedy(tg, wte, n_t + 10)
            m[f"S3 L{L}: (ctx_sum - N*static_bias) vs wte"] += len(unique & set(rec))

            # C: contextual decomposition (corrected method)
            tg = ctx_sum - n_t * bias
            rec = decompose_greedy(tg, v, n_t + 10)
            m[f"C  L{L}: (ctx_sum - N*bias) vs ctx_vocab"] += len(unique & set(rec))

            # CL: int8 low band
            tg_low = int8_quantize(tg)
            rec = decompose_greedy(tg_low, v_low, n_t + 10)
            m[f"CL L{L}: int8 low band (debias→split)"] += len(unique & set(rec))

            # B: bridge
            tg = ctx_sum @ W
            rec = decompose_greedy(tg, wte, n_t + 10)
            m[f"B  L{L}: ctx_sum @ W vs wte"] += len(unique & set(rec))

            # BD: bridge with bias subtraction
            tg = (ctx_sum - n_t * bias) @ W
            rec = decompose_greedy(tg, wte, n_t + 10)
            m[f"BD L{L}: (ctx_sum - N*bias) @ W vs wte"] += len(unique & set(rec))

            # BDC: BD + coord descent
            tg = (ctx_sum - n_t * bias) @ W
            rec = coord_descent(tg, wte, n_t)
            m[f"BDC L{L}: BD + coord descent"] += len(unique & set(rec))

        results.update(m)

        del v, v_low, W
        gc.collect()

    # ----------------------------------------------------------------
    # Print sorted comparison
    # ----------------------------------------------------------------
    print(f"\n{'='*78}")
    print(f"METHOD COMPARISON ({len(SENTENCES)} sentences, {total_unique} unique tokens)")
    print(f"{'='*78}")

    sorted_results = sorted(results.items(), key=lambda x: -x[1])
    for name, hits in sorted_results:
        pct = 100 * hits / total_unique
        bar = "#" * int(pct / 2)
        print(f"  {name:55s} {hits:>4d}/{total_unique:<4d}  {pct:>5.1f}%  {bar}")


if __name__ == "__main__":
    main()
