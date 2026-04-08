#!/usr/bin/env python3
"""Three things in one script:

  A. Pooled-vector decomposition. Instead of summing trailing positions,
     mean-pool them (and optionally L2-normalize) -- the actual production
     style. Try to recover tokens via every method we've used, on the
     20-sentence test set, at L1, L6, L11, L12.

  B. Union of methods. For each test sentence, take the union of recovered
     tokens across the top-N methods and compute aggregate union recovery.
     Tells us whether the methods are finding different tokens (union helps)
     or the same tokens (union doesn't).

  C. Magnitude scaling. Two parts:
     - Sentence length vs sum magnitude: how does ||sum|| grow as N grows?
     - Layer vs sum magnitude: how does ||sum|| grow with layer depth?
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

    # ============================================================
    # PART C: Magnitude scaling (do this first while we have all the data)
    # ============================================================
    print(f"\n{'='*78}")
    print("PART C: Magnitude scaling")
    print(f"{'='*78}")

    print("\n  C1: Per-layer sum-of-trailing magnitude (averaged across all sentences)")
    print(f"  {'Layer':>6s}  {'avg ||sum||':>12s}  {'std':>10s}  {'avg ||mean_pool||':>18s}  {'avg ||last_token||':>18s}")
    print("  " + "-" * 75)
    for L in range(0, 13):
        sum_norms = []
        mean_norms = []
        last_norms = []
        for _, tt, hs in test_data:
            trailing = hs[L][1:]
            sum_norms.append(np.linalg.norm(trailing.sum(axis=0)))
            mean_norms.append(np.linalg.norm(trailing.mean(axis=0)))
            last_norms.append(np.linalg.norm(trailing[-1]))
        print(f"  L{L:>2d}     {np.mean(sum_norms):>12.2f}  {np.std(sum_norms):>10.2f}  "
              f"{np.mean(mean_norms):>18.2f}  {np.mean(last_norms):>18.2f}")

    print("\n  C2: Sentence length vs sum magnitude at L1, L6, L11, L12")
    print(f"  {'sentence':45s}  {'N':>3s}  {'L1 sum':>9s}  {'L6 sum':>9s}  {'L11 sum':>9s}  {'L12 sum':>9s}")
    print("  " + "-" * 95)
    # Sort sentences by length and report
    sorted_by_len = sorted(test_data, key=lambda x: len(x[1]) - 1)
    for sent, tt, hs in sorted_by_len:
        n = len(tt) - 1
        norms = [np.linalg.norm(hs[L][1:].sum(axis=0)) for L in [1, 6, 11, 12]]
        print(f"  {sent[:43]:45s}  {n:>3d}  {norms[0]:>9.2f}  {norms[1]:>9.2f}  {norms[2]:>9.2f}  {norms[3]:>9.2f}")

    # ============================================================
    # PART A: Pooled-vector decomposition
    # ============================================================
    print(f"\n{'='*78}")
    print("PART A: Pooled-vector decomposition (production style)")
    print(f"{'='*78}")
    print("  Instead of summing trailing positions, MEAN-pool them and optionally L2-normalize.")
    print("  Each method tested at L1, L6, L11, L12 with mean-pool.")
    print()

    pooled_results = {}  # method_name -> hits

    for L in layers:
        print(f"  Building vocab + bias + W at L{L}...")
        v = build_vocab_pos1(model, L, device, eot)
        v_low = int8_quantize(v)

        # Per-token bias for sum-based methods (computed normally)
        bias_accum = []
        for rt, rh in zip(ref_token_lists, ref_hs_per_layer[L]):
            trailing = rh[1:]
            n_r = trailing.shape[0]
            bias_accum.append((trailing.sum(axis=0) - v[rt[1:]].sum(axis=0)) / n_r)
        bias = np.mean(bias_accum, axis=0)

        # Mean-pool variant: bias for the mean. (mean - mean_of_vocab)
        # This is the "expected difference between mean-pooled hidden and mean of vocab entries"
        mean_bias_accum = []
        for rt, rh in zip(ref_token_lists, ref_hs_per_layer[L]):
            trailing = rh[1:]
            mean_bias_accum.append(trailing.mean(axis=0) - v[rt[1:]].mean(axis=0))
        mean_bias = np.mean(mean_bias_accum, axis=0)

        W, _, _, _ = np.linalg.lstsq(v, wte, rcond=None)

        # Per-method counters at this layer
        m_pool = {
            f"mean-pool C  L{L}: (mean - mean_bias) vs ctx_vocab":      0,
            f"mean-pool CN L{L}: L2-normalized mean vs ctx_vocab":      0,
            f"mean-pool BD L{L}: (mean - mean_bias) @ W vs wte":        0,
            f"last-tok L{L}: hs[-1] vs ctx_vocab":                       0,
        }

        for sent, tt, hs_per_layer in test_data:
            trailing_tokens = tt[1:]
            unique = set(trailing_tokens)
            n_t = len(trailing_tokens)
            trailing_hs = hs_per_layer[L][1:]

            mean_pooled = trailing_hs.mean(axis=0)
            last_tok = trailing_hs[-1]

            # mean-pool C: contextual decomp against vocab
            # Note: with mean-pool, the "right" target is approximately mean of vocab[t_i]
            # So debiased target = mean_pooled - mean_bias
            tg = mean_pooled - mean_bias
            # The greedy decomposer subtracts vocab entries from the residual; if the
            # target is the mean of N vocab entries, decomposition naturally finds them
            # if we run for ~N steps (need to be careful about residual norm logic)
            rec = decompose_greedy(tg * n_t, v, n_t + 10)  # scale up to act like a sum
            m_pool[f"mean-pool C  L{L}: (mean - mean_bias) vs ctx_vocab"] += len(unique & set(rec))

            # mean-pool with L2 normalization (the production case)
            mp_norm = np.linalg.norm(mean_pooled)
            if mp_norm > 1e-10:
                normalized = mean_pooled / mp_norm
                # We've lost magnitude but the direction is preserved
                rec = decompose_greedy(normalized * n_t * 100, v, n_t + 10)
                m_pool[f"mean-pool CN L{L}: L2-normalized mean vs ctx_vocab"] += len(unique & set(rec))

            # mean-pool bridge
            tg = (mean_pooled - mean_bias) @ W
            rec = decompose_greedy(tg * n_t, wte, n_t + 10)
            m_pool[f"mean-pool BD L{L}: (mean - mean_bias) @ W vs wte"] += len(unique & set(rec))

            # last-token only
            rec = decompose_greedy(last_tok, v, n_t + 10)
            m_pool[f"last-tok L{L}: hs[-1] vs ctx_vocab"] += len(unique & set(rec))

        pooled_results.update(m_pool)

        del v, v_low, W
        gc.collect()

    # ============================================================
    # PART B: Union of methods (using the standard sum-based methods)
    # ============================================================
    print(f"\n{'='*78}")
    print("PART B: Union of top methods (sum-based, sink-skip + N-debias)")
    print(f"{'='*78}")
    print("  For each sentence, take union of recovered tokens across these methods")
    print("  at L1, L6, L11. Reports whether the methods miss different tokens.")
    print()

    # Re-build vocabs for union analysis
    union_results = {}

    for L in [1, 6, 11]:
        print(f"  Building vocab at L{L} (for union analysis)...")
        v = build_vocab_pos1(model, L, device, eot)
        v_low = int8_quantize(v)

        bias_accum = []
        for rt, rh in zip(ref_token_lists, ref_hs_per_layer[L]):
            trailing = rh[1:]
            n_r = trailing.shape[0]
            bias_accum.append((trailing.sum(axis=0) - v[rt[1:]].sum(axis=0)) / n_r)
        bias = np.mean(bias_accum, axis=0)
        W, _, _, _ = np.linalg.lstsq(v, wte, rcond=None)

        # Track per-method recoveries and union
        union_hits = 0
        all_method_hits = {
            f"C  L{L}":   0,
            f"CL L{L}":   0,
            f"BDC L{L}":  0,
        }

        for sent, tt, hs_per_layer in test_data:
            trailing_tokens = tt[1:]
            unique = set(trailing_tokens)
            n_t = len(trailing_tokens)

            ctx_sum = hs_per_layer[L][1:].sum(axis=0)
            tg = ctx_sum - n_t * bias

            # C: standard contextual decomp
            rec_C = set(decompose_greedy(tg, v, n_t + 10))
            all_method_hits[f"C  L{L}"] += len(unique & rec_C)

            # CL: int8 low band
            tg_low = int8_quantize(tg)
            rec_CL = set(decompose_greedy(tg_low, v_low, n_t + 10))
            all_method_hits[f"CL L{L}"] += len(unique & rec_CL)

            # BDC: bridge + bias + CD
            bridged = (ctx_sum - n_t * bias) @ W
            rec_BDC = set(coord_descent(bridged, wte, n_t))
            all_method_hits[f"BDC L{L}"] += len(unique & rec_BDC)

            # Union
            full_union = rec_C | rec_CL | rec_BDC
            union_hits += len(unique & full_union)

        union_results[f"L{L}"] = {
            'C':   all_method_hits[f"C  L{L}"],
            'CL':  all_method_hits[f"CL L{L}"],
            'BDC': all_method_hits[f"BDC L{L}"],
            'union': union_hits,
        }

        del v, v_low, W
        gc.collect()

    # ============================================================
    # Print all results
    # ============================================================
    print(f"\n{'='*78}")
    print("SUMMARY: POOLED-VECTOR METHODS")
    print(f"{'='*78}")
    sorted_pool = sorted(pooled_results.items(), key=lambda x: -x[1])
    for name, hits in sorted_pool:
        pct = 100 * hits / total_unique
        print(f"  {name:60s} {hits:>4d}/{total_unique:<4d}  {pct:>5.1f}%")

    print(f"\n{'='*78}")
    print("SUMMARY: UNION OF METHODS PER LAYER")
    print(f"{'='*78}")
    print(f"  {'Layer':>6s}  {'C':>10s}  {'CL':>10s}  {'BDC':>10s}  {'union':>10s}  {'union/best':>12s}")
    for L in [1, 6, 11]:
        d = union_results[f"L{L}"]
        best = max(d['C'], d['CL'], d['BDC'])
        gain = d['union'] - best
        u_total = total_unique
        print(f"  L{L:>2d}     {d['C']:>4d}/{u_total:<4d}  {d['CL']:>4d}/{u_total:<4d}  "
              f"{d['BDC']:>4d}/{u_total:<4d}  {d['union']:>4d}/{u_total:<4d}  +{gain}")


if __name__ == "__main__":
    main()
