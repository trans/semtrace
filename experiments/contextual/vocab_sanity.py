#!/usr/bin/env python3
"""Two foundational tests for the contextual vocabulary frame:

  Q1. Semantic clustering. With the sink-corrected vocab built from
      [EOT, token] pairs at position 1, do similar tokens still cluster
      together by cosine similarity? Compare to the static wte clustering
      for the same probe tokens.

  Q2. Position invariance. How close is `cat at position 1 of [EOT, cat]`
      (which is the vocab entry) to `cat` appearing at later positions in
      longer sentences? Measure cosine to the vocab entry as a function of
      sentence position and preceding-content variation.

Both questions need answers before we trust any contextual decomposition
result that uses a position-1 vocabulary.
"""
import gc
import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer


PROBES = ["cat", " cat", " dog", " happy", " run", " quick", " mat", " car", " king", " water"]
LAYERS = [1, 6, 11]
TOPK = 8


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


def cos_neighbors(query_idx, vocab, k):
    q = vocab[query_idx]
    norms = np.linalg.norm(vocab, axis=1)
    norms[norms < 1e-10] = 1.0
    sims = vocab @ q / (norms * np.linalg.norm(q))
    sims[query_idx] = -np.inf  # exclude self
    top = np.argsort(-sims)[:k]
    return top, sims[top]


def main():
    device = "cpu"
    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True).to(device)
    model.eval()
    eot = tokenizer.eos_token_id

    wte = model.wte.weight.detach().cpu().numpy()

    # Encode probes (each probe should be exactly one token)
    probe_ids = []
    for p in PROBES:
        ids = tokenizer.encode(p)
        if len(ids) != 1:
            print(f"  WARNING: {p!r} encodes to {len(ids)} tokens, skipping")
            continue
        probe_ids.append((p, ids[0]))
    print(f"Probes: {[(p, i) for p, i in probe_ids]}")

    # ==================================================================
    # Q1a: Static wte nearest neighbors (baseline)
    # ==================================================================
    print(f"\n{'='*78}")
    print("Q1a. Static wte nearest neighbors (baseline)")
    print(f"{'='*78}")
    for probe, pid in probe_ids:
        top, sims = cos_neighbors(pid, wte, TOPK)
        nbrs = [(tokenizer.decode([int(t)]), float(s)) for t, s in zip(top, sims)]
        nbr_str = ", ".join(f"{t!r}({s:.2f})" for t, s in nbrs)
        print(f"  {probe!r:>14s}: {nbr_str}")

    # ==================================================================
    # Q1b: Sink-corrected contextual vocab nearest neighbors at each layer
    # ==================================================================
    for L in LAYERS:
        print(f"\n{'='*78}")
        print(f"Q1b. Sink-corrected vocab at L{L} — nearest neighbors")
        print(f"{'='*78}")
        v = build_vocab_pos1(model, L, device, eot)
        for probe, pid in probe_ids:
            top, sims = cos_neighbors(pid, v, TOPK)
            nbrs = [(tokenizer.decode([int(t)]), float(s)) for t, s in zip(top, sims)]
            nbr_str = ", ".join(f"{t!r}({s:.2f})" for t, s in nbrs)
            print(f"  {probe!r:>14s}: {nbr_str}")

        # Q2 setup: keep this vocab around for the position-invariance test below
        if L == 1:
            v_L1 = v
        elif L == 6:
            v_L6 = v
        elif L == 11:
            v_L11 = v
        del v
        gc.collect()

    # ==================================================================
    # Q2: Position invariance
    # ==================================================================
    print(f"\n{'='*78}")
    print("Q2. Position invariance: how close is `cat` at varying positions")
    print(f"    to its vocab entry (which is `cat` at position 1 of [EOT, cat])?")
    print(f"{'='*78}")

    # Test sentences containing " cat" at different positions
    test_sentences = [
        " cat",                                    # → cat at position 1 (this IS the vocab entry)
        "the cat",                                 # cat at position 2
        "the big cat",                             # cat at position 3
        "the small black cat",                     # cat at position 4
        "I saw a small black cat",                 # cat at position 6
        "yesterday I saw a small black cat",       # cat at position 7
        "in the garden yesterday I saw a small black cat",   # cat at position 9
        "while walking in the garden yesterday I saw a small black cat",  # cat at position 11
    ]

    target_id = tokenizer.encode(" cat")[0]
    print(f"\n  Target token: {' cat'!r} (id {target_id})")

    for L, vocab in [(1, v_L1), (6, v_L6), (11, v_L11)]:
        canonical = vocab[target_id]
        print(f"\n  Layer L{L}, canonical {' cat'!r} norm = {np.linalg.norm(canonical):.2f}")
        print(f"  {'Sentence':70s}  {'pos':>4s}  {'norm':>8s}  {'cos(canonical)':>15s}")
        print("  " + "-" * 105)

        for sent in test_sentences:
            tt = [eot] + tokenizer.encode(sent)
            with torch.no_grad():
                hs = model(torch.tensor([tt]).to(device)).hidden_states[L].squeeze(0).cpu().numpy()
            # Find the position of " cat" in the sequence
            cat_positions = [i for i, t in enumerate(tt) if t == target_id]
            if not cat_positions:
                continue
            for pos in cat_positions:
                vec = hs[pos]
                c = float(np.dot(vec, canonical) / (np.linalg.norm(vec) * np.linalg.norm(canonical)))
                print(f"  {sent[:68]!r:70s}  {pos:>4d}  {np.linalg.norm(vec):>8.2f}  {c:>15.4f}")

    # ==================================================================
    # Q1c: Pairwise sentence cosines (semantic discrimination on full
    #      sentences). Tests whether GPT-2 sentence representations
    #      really lack discrimination, or whether this was sink-driven.
    # ==================================================================
    print(f"\n{'='*78}")
    print("Q1c. Sentence-pair cosine similarity (sink-skipped trailing-position sums)")
    print(f"{'='*78}")
    print("  Compares pairs that should be similar/different. If these are all 0.99+")
    print("  the original 'GPT-2 lacks discrimination' finding stands. If discrimination")
    print("  appears with sink-skip, the original finding was sink-driven.")
    print()

    pairs = [
        ("the dog chased the cat", "the cat chased the dog", "same words, swapped order"),
        ("I am happy", "I am not happy", "negation"),
        ("the cat sat on the mat", "the cat sat on the rug", "synonym swap"),
        ("the cat sat on the mat", "quantum physics is fascinating", "unrelated"),
        ("she went to the store", "he went to the store", "pronoun swap"),
        ("the quick brown fox", "the lazy brown fox", "adjective swap"),
    ]

    for L in LAYERS:
        print(f"\n  --- Layer L{L} ---")
        for s1, s2, label in pairs:
            t1 = [eot] + tokenizer.encode(s1)
            t2 = [eot] + tokenizer.encode(s2)
            with torch.no_grad():
                h1 = model(torch.tensor([t1]).to(device)).hidden_states[L].squeeze(0).cpu().numpy()
                h2 = model(torch.tensor([t2]).to(device)).hidden_states[L].squeeze(0).cpu().numpy()
            sum1 = h1[1:].sum(axis=0)  # sink-skipped trailing sum
            sum2 = h2[1:].sum(axis=0)
            c = float(np.dot(sum1, sum2) / (np.linalg.norm(sum1) * np.linalg.norm(sum2)))

            # Also compute raw (sink-included) sum for comparison
            raw1 = h1.sum(axis=0)
            raw2 = h2.sum(axis=0)
            c_raw = float(np.dot(raw1, raw2) / (np.linalg.norm(raw1) * np.linalg.norm(raw2)))

            print(f"    {label:30s}  raw cos: {c_raw:.4f}  skipped cos: {c:.4f}")


if __name__ == "__main__":
    main()
