#!/usr/bin/env python3
"""Dirty bag cleanup: overshoot + leave-one-out trim + targeted replacement.

Full pipeline for improving a noisy bag from the bridge:
  1. Let greedy overshoot to N+K tokens (more candidates to work with)
  2. Leave-one-out to flag intruders (delta < 0)
  3. Remove intruders, trimming back toward N
  4. If still short of N, search for replacement tokens at missing positions
     by trying top-K vocab candidates scored by forward-pass L2
  5. Report cleaned bag accuracy vs original dirty bag
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


def greedy_overshoot(target, vocab, n_target, overshoot=5):
    """Greedy decomposition allowing up to n_target + overshoot tokens."""
    residual = target.copy()
    recovered = []
    prev_norm = float("inf")
    max_steps = n_target + overshoot
    for _ in range(max_steps):
        r_norm = np.linalg.norm(residual)
        if r_norm < 0.001:
            break
        # Don't stop on norm increase until we have at least n_target
        if len(recovered) >= n_target and r_norm > prev_norm:
            break
        prev_norm = r_norm
        norms = np.linalg.norm(vocab, axis=1)
        norms[norms < 1e-10] = 1.0
        sims = vocab @ residual / (norms * r_norm)
        best = int(np.argmax(sims))
        recovered.append(best)
        residual = residual - vocab[best]
    return recovered


def leave_one_out_trim(model, target_emb, sequence, device, eot, n_target):
    """Remove tokens with negative L2-delta until sequence length <= n_target.
    Returns the trimmed sequence."""
    current = list(sequence)

    while len(current) > n_target:
        base_l2 = float(np.linalg.norm(
            target_emb - get_pooled(model, [eot] + current, device)))

        # Find the token whose removal most IMPROVES (or least hurts) L2
        best_remove = None
        best_l2_after = base_l2

        for k in range(len(current)):
            shortened = current[:k] + current[k+1:]
            l2 = float(np.linalg.norm(
                target_emb - get_pooled(model, [eot] + shortened, device)))
            if l2 < best_l2_after:
                best_l2_after = l2
                best_remove = k

        if best_remove is not None:
            current.pop(best_remove)
        else:
            # No removal improves L2; force-remove the least-impact token
            deltas = []
            for k in range(len(current)):
                shortened = current[:k] + current[k+1:]
                l2 = float(np.linalg.norm(
                    target_emb - get_pooled(model, [eot] + shortened, device)))
                deltas.append((k, l2 - base_l2))
            deltas.sort(key=lambda x: x[1])
            current.pop(deltas[0][0])

    return current


def targeted_replace(model, target_emb, sequence, n_target, device, eot, wte, top_k=30):
    """If sequence is shorter than n_target, find replacement tokens.
    For each missing slot, try top_k vocab candidates by cosine to the
    residual, score by forward-pass L2, keep the best."""
    current = list(sequence)

    while len(current) < n_target:
        # Compute what's "missing" from the embedding
        current_emb = get_pooled(model, [eot] + current, device)
        residual = target_emb - current_emb

        # Find top-K vocab entries by cosine to residual
        res_norm = np.linalg.norm(residual)
        if res_norm < 1e-10:
            break
        wte_norms = np.linalg.norm(wte, axis=1)
        wte_norms[wte_norms < 1e-10] = 1.0
        sims = wte @ residual / (wte_norms * res_norm)
        candidates = np.argsort(-sims)[:top_k]

        # Try inserting each candidate at each position, pick best by L2
        best_insert = None
        best_l2 = float("inf")

        for cand in candidates:
            # Try appending (simplest — just add to end)
            trial = current + [int(cand)]
            trial_emb = get_pooled(model, [eot] + trial, device)
            l2 = float(np.linalg.norm(target_emb - trial_emb))
            if l2 < best_l2:
                best_l2 = l2
                best_insert = int(cand)

        if best_insert is not None:
            current.append(best_insert)
        else:
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
        bias_accum.append((rh[1:].sum(axis=0) - v[rt[1:]].sum(axis=0)) / len(rt[1:]))
    bias = np.mean(bias_accum, axis=0)
    W, _, _, _ = np.linalg.lstsq(v, wte, rcond=None)

    # Aggregates
    total_unique = 0
    agg_dirty = 0
    agg_overshoot = 0
    agg_trimmed = 0
    agg_replaced = 0

    for sent in SENTENCES:
        print(f"\n{'='*70}")
        print(f"{sent!r}")
        true_tokens = [eot] + tokenizer.encode(sent)
        true_bag = set(true_tokens[1:])
        n_t = len(true_tokens) - 1
        u = len(true_bag)
        total_unique += u

        target_emb = get_pooled(model, true_tokens, device)

        # Bridge target
        with torch.no_grad():
            ctx_sum = model(torch.tensor([true_tokens]).to(device),
                           output_hidden_states=True).hidden_states[L] \
                .squeeze(0)[1:].sum(dim=0).cpu().numpy()
        bridged = (ctx_sum - n_t * bias) @ W

        # Step 1: Greedy with overshoot
        overshot = greedy_overshoot(bridged, wte, n_t, overshoot=5)
        os_correct = len(true_bag & set(overshot))
        agg_overshoot += os_correct
        print(f"  Overshoot ({len(overshot)} tokens): {os_correct}/{u} correct")

        # Also get the N-forced version for comparison
        from contextlib import redirect_stdout
        import io
        dirty_n = greedy_overshoot(bridged, wte, n_t, overshoot=0)[:n_t]
        while len(dirty_n) < n_t:
            dirty_n.append(0)
        dirty_correct = len(true_bag & set(dirty_n))
        agg_dirty += dirty_correct
        print(f"  N-forced  ({len(dirty_n)} tokens): {dirty_correct}/{u} correct")

        # Step 2: Leave-one-out trim
        trimmed = leave_one_out_trim(model, target_emb, overshot, device, eot, n_t)
        trim_correct = len(true_bag & set(trimmed))
        agg_trimmed += trim_correct
        print(f"  Trimmed   ({len(trimmed)} tokens): {trim_correct}/{u} correct")

        # Step 3: Targeted replacement for missing slots
        if len(trimmed) < n_t:
            replaced = targeted_replace(model, target_emb, trimmed, n_t, device, eot, wte)
        else:
            replaced = trimmed
        rep_correct = len(true_bag & set(replaced))
        agg_replaced += rep_correct
        print(f"  Replaced  ({len(replaced)} tokens): {rep_correct}/{u} correct")

        # Show the progression
        dirty_strs = [tokenizer.decode([t]) for t in dirty_n]
        rep_strs = [tokenizer.decode([t]) for t in replaced]
        true_strs = [tokenizer.decode([t]) for t in true_tokens[1:]]
        print(f"    dirty:    {dirty_strs}")
        print(f"    cleaned:  {rep_strs}")
        print(f"    true:     {true_strs}")

    print(f"\n{'='*70}")
    print("AGGREGATE")
    print(f"{'='*70}")
    print(f"  N-forced dirty bag:     {agg_dirty}/{total_unique} ({100*agg_dirty/total_unique:.1f}%)")
    print(f"  With overshoot:         {agg_overshoot}/{total_unique} ({100*agg_overshoot/total_unique:.1f}%)")
    print(f"  After LOO trim:         {agg_trimmed}/{total_unique} ({100*agg_trimmed/total_unique:.1f}%)")
    print(f"  After replacement:      {agg_replaced}/{total_unique} ({100*agg_replaced/total_unique:.1f}%)")

    del v, W
    gc.collect()


if __name__ == "__main__":
    main()
