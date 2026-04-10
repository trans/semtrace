#!/usr/bin/env python3
"""Hybrid beam search with LLM-seeded initialization.

Two improvements over the plain beam search:

  1. LLM-seeded initialization: instead of starting from an empty prefix,
     generate multiple complete orderings via constrained-greedy (one per
     possible first token in the bag). Score each by L2. Use the best as
     a reference for comparison.

  2. Hybrid scoring during beam expansion: at each step, rank candidates
     by a combination of LM log-prob AND partial-sequence L2 distance to
     target. This lets the beam prune based on embedding closeness during
     search, not just at the end.

Tested on "Mary had a little lamb..." (23 tokens).
"""
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer


TEXT = "Mary had a little lamb its fleece was white as snow and everywhere that Mary went the lamb was sure to go"


def get_pooled(model, tokens, device):
    with torch.no_grad():
        out = model(torch.tensor([tokens]).to(device), output_hidden_states=True)
    l12 = out.hidden_states[12].squeeze(0)
    return l12[1:].mean(dim=0).cpu().numpy()


def constrained_greedy(model, bag, device, eot, first_token=None):
    """Generate one ordering by constrained greedy: at each step, pick the
    highest-LM-probability remaining bag token."""
    remaining = list(bag)
    sequence = []
    prefix = [eot]

    if first_token is not None:
        idx = remaining.index(first_token)
        remaining.pop(idx)
        sequence.append(first_token)
        prefix.append(first_token)

    while remaining:
        with torch.no_grad():
            out = model(torch.tensor([prefix]).to(device))
        logits = out.logits.squeeze(0)[-1]
        best_score = -float('inf')
        best_idx = 0
        for i, t in enumerate(remaining):
            score = float(logits[t])
            if score > best_score:
                best_score = score
                best_idx = i
        chosen = remaining.pop(best_idx)
        sequence.append(chosen)
        prefix.append(chosen)
    return sequence


def hybrid_beam_search(model, bag, device, eot, target_emb, beam_width=16,
                       l2_weight=0.1, l2_every=3):
    """Beam search scoring by LM log-prob + L2 distance.

    At every `l2_every` steps, compute partial-sequence L2 and add it to
    the ranking score. This guides the beam toward embeddings that are
    close to the target throughout the search, not just at the end.
    """
    beams = [([], list(bag), 0.0)]
    step = 0
    n_total = len(bag)

    while beams[0][1]:
        step += 1
        new_beams = []
        for seq, remaining, score in beams:
            prefix = [eot] + seq
            with torch.no_grad():
                out = model(torch.tensor([prefix]).to(device))
            logits = out.logits.squeeze(0)[-1]
            log_probs = F.log_softmax(logits, dim=-1)
            seen = set()
            for i, t in enumerate(remaining):
                if t in seen:
                    continue
                seen.add(t)
                new_remaining = list(remaining)
                new_remaining.pop(i)
                new_seq = seq + [t]
                new_score = score + float(log_probs[t])
                new_beams.append((new_seq, new_remaining, new_score))

        # At L2 steps, re-score by LM + L2
        if step % l2_every == 0 or not new_beams[0][1]:  # also at final step
            scored = []
            for seq, remaining, lm_score in new_beams[:beam_width * 3]:  # pre-filter to save compute
                emb = get_pooled(model, [eot] + seq, device)
                l2 = float(np.linalg.norm(target_emb - emb))
                # Combined score: higher is better for LM, lower is better for L2
                combined = lm_score - l2_weight * l2
                scored.append((seq, remaining, combined))
            scored.sort(key=lambda x: -x[2])
            beams = scored[:beam_width]
            if step % 5 == 0 or not beams[0][1]:
                print(f"      step {step}/{n_total} (L2-scored)")
        else:
            new_beams.sort(key=lambda x: -x[2])
            beams = new_beams[:beam_width]

    # Final scoring by L2 only
    final = []
    for seq, _, score in beams:
        emb = get_pooled(model, [eot] + seq, device)
        l2 = float(np.linalg.norm(target_emb - emb))
        final.append((seq, l2))
    final.sort(key=lambda x: x[1])
    return final


def main():
    device = "cpu"
    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True).to(device)
    model.eval()
    eot = tokenizer.eos_token_id

    true_tokens = [eot] + tokenizer.encode(TEXT)
    bag = true_tokens[1:]
    n_t = len(bag)
    true_strs = [tokenizer.decode([t]) for t in bag]
    target_emb = get_pooled(model, true_tokens, device)

    print(f"\nText: {TEXT!r}")
    print(f"Tokens ({n_t}): {true_strs}")

    # ================================================================
    # Phase 1: LLM-seeded initialization
    # ================================================================
    print(f"\n{'='*70}")
    print("Phase 1: Generate orderings seeded from each possible first token")
    print(f"{'='*70}")

    unique_first_tokens = list(set(bag))
    print(f"  {len(unique_first_tokens)} unique first tokens to try")

    seeds = []
    for ft in unique_first_tokens:
        ordering = constrained_greedy(model, bag, device, eot, first_token=ft)
        emb = get_pooled(model, [eot] + ordering, device)
        l2 = float(np.linalg.norm(target_emb - emb))
        pos_match = sum(1 for a, b in zip(ordering, bag) if a == b)
        seeds.append((ordering, l2, pos_match, ft))

    seeds.sort(key=lambda x: x[1])

    print(f"\n  Top 5 seeds by L2:")
    for i, (seq, l2, pm, ft) in enumerate(seeds[:5]):
        strs = ' '.join([tokenizer.decode([t]) for t in seq])
        print(f"    seed {i+1} (first={tokenizer.decode([ft])!r}): L2={l2:.4f}  pm={pm}/{n_t}")
        print(f"      {strs}")

    best_seed = seeds[0][0]
    best_seed_l2 = seeds[0][1]
    best_seed_pm = seeds[0][2]
    print(f"\n  Best seed: L2={best_seed_l2:.4f}, pos-match={best_seed_pm}/{n_t}")

    # Also check the true ordering's rank among all seeds
    true_l2 = float(np.linalg.norm(target_emb - get_pooled(model, true_tokens, device)))
    print(f"  True sequence L2: {true_l2:.4f}")

    # ================================================================
    # Phase 2: Hybrid beam search
    # ================================================================
    print(f"\n{'='*70}")
    print("Phase 2: Hybrid beam search (LM log-prob + L2 scoring)")
    print(f"{'='*70}")

    for B, l2w in [(16, 0.1), (32, 0.1), (32, 0.5)]:
        print(f"\n  --- B={B}, L2_weight={l2w} ---")
        results = hybrid_beam_search(
            model, bag, device, eot, target_emb,
            beam_width=B, l2_weight=l2w, l2_every=3
        )

        best_seq, best_l2 = results[0]
        best_strs = ' '.join([tokenizer.decode([t]) for t in best_seq])
        best_pm = sum(1 for a, b in zip(best_seq, bag) if a == b)
        print(f"  best: L2={best_l2:.4f}  pm={best_pm}/{n_t}")
        print(f"    {best_strs}")
        if best_pm == n_t:
            print(f"    *** PERFECT ***")

        for i, (seq, l2) in enumerate(results[:3]):
            strs = ' '.join([tokenizer.decode([t]) for t in seq])
            pm = sum(1 for a, b in zip(seq, bag) if a == b)
            print(f"    beam {i+1}: L2={l2:.4f}  pm={pm}/{n_t}  {strs}")


if __name__ == "__main__":
    main()
