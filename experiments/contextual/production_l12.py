#!/usr/bin/env python3
"""Production-realistic L12 mean-pool + L2-normalize decomposition.

This is the closest thing we can test to what a real embedding endpoint
returns: a single fixed-dim vector that is the L2-normalized mean of the
final layer's hidden states across positions.

Methods tested at L11 and L12:
  - mean-pool (no normalize), contextual decomp against ctx_vocab
  - mean-pool L2-normalized, contextual decomp at multiple scales
  - mean-pool, bridge through W, decomp against wte
  - mean-pool L2-normalized, bridge through W, decomp against wte
  - bridge + coord descent
  - union of best methods

Also tests whether the L1 overscaling trick that gave 97% generalizes
to deeper layers.
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
        nrm = np.linalg.norm(vocab, axis=1)
        nrm[nrm < 1e-10] = 1.0
        sims = vocab @ ideal / nrm
        current.append(int(np.argmax(sims)))
    current = current[:n_tokens]

    best_dist = np.linalg.norm(target - vocab[current].sum(axis=0))
    for _ in range(max_iters):
        improved = False
        for pos in range(n_tokens):
            others = [current[j] for j in range(n_tokens) if j != pos]
            others_sum = vocab[others].sum(axis=0)
            ideal = target - others_sum
            nrm = np.linalg.norm(vocab, axis=1)
            nrm[nrm < 1e-10] = 1.0
            sims = vocab @ ideal / nrm
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

    layers = [11, 12]  # focus on production-relevant
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
    print(f"Test set: {len(SENTENCES)} sentences, {total_unique} unique tokens")

    for L in layers:
        print(f"\n{'='*78}")
        print(f"LAYER L{L} (mean-pool, with and without L2 normalize)")
        print(f"{'='*78}")

        v = build_vocab_pos1(model, L, device, eot)
        v_norm_avg = float(np.mean(np.linalg.norm(v, axis=1)))
        print(f"  vocab avg norm: {v_norm_avg:.2f}")

        # Per-token bias (for sum-based methods)
        bias_accum = []
        for rt, rh in zip(ref_token_lists, ref_hs_per_layer[L]):
            trailing = rh[1:]
            n_r = trailing.shape[0]
            bias_accum.append((trailing.sum(axis=0) - v[rt[1:]].sum(axis=0)) / n_r)
        sum_bias = np.mean(bias_accum, axis=0)

        # Mean-bias (for mean-pool methods)
        mean_bias_accum = []
        for rt, rh in zip(ref_token_lists, ref_hs_per_layer[L]):
            trailing = rh[1:]
            mean_bias_accum.append(trailing.mean(axis=0) - v[rt[1:]].mean(axis=0))
        mean_bias = np.mean(mean_bias_accum, axis=0)

        W, _, _, _ = np.linalg.lstsq(v, wte, rcond=None)

        methods = {
            f"L{L} mean-pool, ctx_vocab, no debias":             0,
            f"L{L} mean-pool, ctx_vocab, mean_bias debias":      0,
            f"L{L} mean-pool, ctx_vocab, scale=N*100":           0,
            f"L{L} mean-pool, ctx_vocab, scale=vocab_avg*N":     0,
            f"L{L} L2-norm mean, ctx_vocab, scale=N*100":        0,
            f"L{L} L2-norm mean, ctx_vocab, scale=vocab_avg*N":  0,
            f"L{L} L2-norm mean, ctx_vocab, scale=1":            0,
            f"L{L} mean-pool, bridge to wte":                     0,
            f"L{L} mean-pool, bridge to wte, +CD":                0,
            f"L{L} L2-norm mean, bridge to wte, scale=N*100":    0,
            f"L{L} L2-norm mean, bridge to wte, scale=N*100,+CD":0,
        }

        # Track per-sentence recoveries for union analysis
        method_token_sets = {k: [] for k in methods}

        for sent, tt, hs_per_layer in test_data:
            trailing_tokens = tt[1:]
            unique = set(trailing_tokens)
            n_t = len(trailing_tokens)
            trailing = hs_per_layer[L][1:]

            mean_pooled = trailing.mean(axis=0)
            mp_norm = float(np.linalg.norm(mean_pooled))
            normalized = mean_pooled / max(mp_norm, 1e-12)

            def run_and_track(label, target, vocab):
                rec = decompose_greedy(target, vocab, n_t + 10)
                methods[label] += len(unique & set(rec))
                method_token_sets[label].append(set(rec))

            def run_cd_and_track(label, target, vocab):
                rec = coord_descent(target, vocab, n_t)
                methods[label] += len(unique & set(rec))
                method_token_sets[label].append(set(rec))

            # mean-pool against ctx vocab, various scalings/biases
            run_and_track(f"L{L} mean-pool, ctx_vocab, no debias",
                          mean_pooled * n_t, v)
            run_and_track(f"L{L} mean-pool, ctx_vocab, mean_bias debias",
                          (mean_pooled - mean_bias) * n_t, v)
            run_and_track(f"L{L} mean-pool, ctx_vocab, scale=N*100",
                          mean_pooled * n_t * 100 / max(mp_norm, 1e-12), v)
            run_and_track(f"L{L} mean-pool, ctx_vocab, scale=vocab_avg*N",
                          mean_pooled * v_norm_avg * n_t / max(mp_norm, 1e-12), v)

            # L2-normalized mean against ctx vocab
            run_and_track(f"L{L} L2-norm mean, ctx_vocab, scale=N*100",
                          normalized * n_t * 100, v)
            run_and_track(f"L{L} L2-norm mean, ctx_vocab, scale=vocab_avg*N",
                          normalized * v_norm_avg * n_t, v)
            run_and_track(f"L{L} L2-norm mean, ctx_vocab, scale=1",
                          normalized, v)

            # Bridge methods
            tg = ((mean_pooled - mean_bias) * n_t) @ W
            run_and_track(f"L{L} mean-pool, bridge to wte", tg, wte)
            run_cd_and_track(f"L{L} mean-pool, bridge to wte, +CD", tg, wte)

            tg = (normalized * n_t * 100) @ W
            run_and_track(f"L{L} L2-norm mean, bridge to wte, scale=N*100", tg, wte)
            run_cd_and_track(f"L{L} L2-norm mean, bridge to wte, scale=N*100,+CD", tg, wte)

        # Print sorted
        sorted_results = sorted(methods.items(), key=lambda x: -x[1])
        for label, hits in sorted_results:
            pct = 100 * hits / total_unique
            print(f"    {label:60s} {hits:>4d}/{total_unique:<4d}  {pct:>5.1f}%")

        # Union analysis
        print(f"\n  Union analysis at L{L}:")
        # Take union of top-3 methods
        top_3_labels = [label for label, _ in sorted_results[:3]]
        top_3_sets = [method_token_sets[label] for label in top_3_labels]
        union_hits = 0
        for i, (sent, tt, _) in enumerate(test_data):
            unique = set(tt[1:])
            full_union = set()
            for mset in top_3_sets:
                full_union |= mset[i]
            union_hits += len(unique & full_union)
        print(f"    union of top-3 methods: {union_hits}/{total_unique}  ({100*union_hits/total_unique:.1f}%)")
        print(f"    (top 3: {[t[5:] for t in top_3_labels]})")

        del v, W
        gc.collect()


if __name__ == "__main__":
    main()
