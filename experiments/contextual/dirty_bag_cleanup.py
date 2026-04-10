#!/usr/bin/env python3
"""Dirty bag cleanup via leave-one-out intruder detection.

The production case: the bridge at L12 gives a noisy bag (~33% correct
tokens, ~67% wrong). Can leave-one-out identify which tokens are intruders?

Method:
  1. Run bridge+CD to get a dirty bag for each sentence
  2. For each token in the bag, compute L2 change when removed
     - Negative delta = intruder (removing it helps)
     - Positive delta = likely correct (removing it hurts)
  3. Check: do the negative-delta tokens correspond to the wrong ones?
  4. Remove intruders and report precision of the cleaned bag
  5. For each removed position, try top-K vocab replacements by L2

Uses the 20-sentence test set at L12 with bridge+CD.
"""
import gc
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


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
]


def get_pooled(model, tokens, device):
    with torch.no_grad():
        out = model(torch.tensor([tokens]).to(device), output_hidden_states=True)
    l12 = out.hidden_states[12].squeeze(0)
    return l12[1:].mean(dim=0).cpu().numpy()


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
    vocab_size = model.transformer.wte.weight.shape[0]
    chunks = []
    for start in range(0, vocab_size, batch):
        end = min(start + batch, vocab_size)
        ids = torch.tensor([[prefix_token, t] for t in range(start, end)]).to(device)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        h = out.hidden_states[layer][:, 1, :].cpu().numpy()
        chunks.append(h)
    return np.concatenate(chunks, axis=0)


def main():
    device = "cpu"
    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True).to(device)
    model.eval()
    eot = tokenizer.eos_token_id
    wte = model.transformer.wte.weight.detach().cpu().numpy()

    L = 12
    print(f"Building L{L} vocab and bridge...")
    v = build_vocab_pos1(model, L, device, eot)

    refs = [
        "the quick brown fox jumps over the lazy dog",
        "she went to the store to buy some food",
        "he ran down the long road to the old house",
        "they played in the park with the children",
    ]
    bias_accum = []
    for ref in refs:
        rt = tokenizer.encode(ref)
        with torch.no_grad():
            rh = model(torch.tensor([rt]).to(device), output_hidden_states=True) \
                .hidden_states[L].squeeze(0).cpu().numpy()
        trailing = rh[1:]
        bias_accum.append((trailing.sum(axis=0) - v[rt[1:]].sum(axis=0)) / len(rt[1:]))
    bias = np.mean(bias_accum, axis=0)

    W, _, _, _ = np.linalg.lstsq(v, wte, rcond=None)

    # Aggregate stats
    total_correct_in_bag = 0
    total_correct_flagged_keep = 0
    total_wrong_flagged_remove = 0
    total_tokens = 0
    total_unique = 0
    total_bag_after_cleanup = 0

    for sent in SENTENCES:
        print(f"\n{'='*70}")
        print(f"Sentence: {sent!r}")
        true_tokens = [eot] + tokenizer.encode(sent)
        true_bag = set(true_tokens[1:])
        n_t = len(true_tokens) - 1
        u = len(true_bag)
        total_unique += u

        target_emb = get_pooled(model, true_tokens, device)

        # Get dirty bag via bridge + CD
        with torch.no_grad():
            ctx_sum = model(torch.tensor([true_tokens]).to(device),
                           output_hidden_states=True).hidden_states[L] \
                .squeeze(0)[1:].sum(dim=0).cpu().numpy()
        bridged = (ctx_sum - n_t * bias) @ W
        dirty_bag = coord_descent(bridged, wte, n_t, max_iters=15)

        # Score the dirty bag
        bag_correct = len(true_bag & set(dirty_bag))
        total_correct_in_bag += bag_correct
        total_tokens += n_t

        dirty_strs = [tokenizer.decode([t]) for t in dirty_bag]
        print(f"  Dirty bag: {bag_correct}/{u} correct tokens")
        print(f"  {dirty_strs}")

        # Leave-one-out: for each token in the dirty bag, compute L2 change
        # We construct a sequence from the dirty bag (in the order CD gave us)
        # and compute L2 to the target embedding
        base_emb = get_pooled(model, [eot] + dirty_bag, device)
        base_l2 = float(np.linalg.norm(target_emb - base_emb))

        loo_results = []
        for k in range(n_t):
            shortened = dirty_bag[:k] + dirty_bag[k+1:]
            short_emb = get_pooled(model, [eot] + shortened, device)
            short_l2 = float(np.linalg.norm(target_emb - short_emb))
            delta = short_l2 - base_l2
            is_correct = dirty_bag[k] in true_bag
            loo_results.append((k, dirty_bag[k], delta, is_correct))

        # Sort by delta (most suspicious first = lowest/most negative delta)
        loo_results.sort(key=lambda x: x[2])

        print(f"\n  Leave-one-out analysis (sorted by suspicion):")
        print(f"  {'pos':>4s}  {'token':>12s}  {'delta':>8s}  {'actual':>8s}")
        n_shown = min(n_t, 12)
        for k, tok, delta, is_correct in loo_results[:n_shown]:
            tok_str = tokenizer.decode([tok])
            label = "RIGHT" if is_correct else "WRONG"
            marker = "<<" if delta < 0 and not is_correct else \
                     "!!" if delta < 0 and is_correct else ""
            print(f"  {k:>4d}  {tok_str!r:>12s}  {delta:>+8.4f}  {label:>8s}  {marker}")

        # Classification accuracy: how well does delta < 0 predict "wrong token"?
        flagged_remove = [(k, tok, d, c) for k, tok, d, c in loo_results if d < 0]
        flagged_keep = [(k, tok, d, c) for k, tok, d, c in loo_results if d >= 0]

        true_positives = sum(1 for _, _, _, c in flagged_remove if not c)  # correctly flagged wrong
        false_positives = sum(1 for _, _, _, c in flagged_remove if c)     # incorrectly flagged right
        true_negatives = sum(1 for _, _, _, c in flagged_keep if c)        # correctly kept right
        false_negatives = sum(1 for _, _, _, c in flagged_keep if not c)   # missed wrong tokens

        total_wrong_flagged_remove += true_positives
        total_correct_flagged_keep += true_negatives

        precision = true_positives / len(flagged_remove) if flagged_remove else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        print(f"\n  Intruder detection (delta < 0 = flag as wrong):")
        print(f"    flagged for removal: {len(flagged_remove)} tokens")
        print(f"    true positives (correctly flagged wrong): {true_positives}")
        print(f"    false positives (incorrectly flagged right): {false_positives}")
        print(f"    precision: {precision:.1%}  recall: {recall:.1%}")

        # The cleaned bag: keep only tokens with delta >= 0
        cleaned = [tok for k, tok, d, c in loo_results if d >= 0]
        cleaned_correct = len(true_bag & set(cleaned))
        total_bag_after_cleanup += cleaned_correct
        print(f"\n  Cleaned bag: {cleaned_correct}/{u} correct out of {len(cleaned)} kept")
        print(f"  (was {bag_correct}/{u} in {n_t} before cleanup)")

    print(f"\n{'='*70}")
    print("AGGREGATE")
    print(f"{'='*70}")
    print(f"  Before cleanup: {total_correct_in_bag}/{total_unique} correct tokens in dirty bags")
    print(f"  After cleanup:  {total_bag_after_cleanup}/{total_unique} correct tokens in cleaned bags")
    print(f"  (Cleanup removes intruders but may also remove some correct tokens)")

    del v, W
    gc.collect()


if __name__ == "__main__":
    main()
