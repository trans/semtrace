#!/usr/bin/env python3
"""Leave-one-out probing for order triangulation.

Given a candidate ordering of a bag of words:
  1. For each position k, remove the token at k and compute L2 of the
     shortened sequence to target. The L2 INCREASE from removal indicates
     how important that position is (higher = more correctly placed).
  2. Use this as a confidence map: high-confidence positions are likely
     correct, low-confidence ones should be searched.
  3. For low-confidence positions, try each bag token there and pick
     whichever minimizes L2.

Also test: given the TRUE sequence, does leave-one-out correctly identify
which positions are right? (Sanity check.)
"""
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


TEXT = "Mary had a little lamb its fleece was white as snow and everywhere that Mary went the lamb was sure to go"


def get_pooled(model, tokens, device):
    with torch.no_grad():
        out = model(torch.tensor([tokens]).to(device), output_hidden_states=True)
    l12 = out.hidden_states[12].squeeze(0)
    return l12[1:].mean(dim=0).cpu().numpy()


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
    target_emb = get_pooled(model, true_tokens, device)

    # Use the swap-refined result from earlier as our "candidate"
    # (approximate: take the true sequence and make some swaps to simulate
    # an imperfect ordering, similar to what Mistral+refinement gives)
    # For a clean test, let's use:
    # A) the true sequence (sanity check)
    # B) a shuffled version (realistic test)

    import random
    random.seed(123)
    shuffled = list(bag)
    random.shuffle(shuffled)

    for label, candidate in [("TRUE", list(bag)), ("SHUFFLED", shuffled)]:
        print(f"\n{'='*70}")
        print(f"Candidate: {label}")
        strs = [tokenizer.decode([t]) for t in candidate]
        print(f"  sequence: {' '.join(strs)}")

        base_l2 = float(np.linalg.norm(target_emb - get_pooled(model, [eot] + candidate, device)))
        print(f"  base L2: {base_l2:.4f}")

        # Leave-one-out: remove each position and compute L2
        print(f"\n  Leave-one-out (remove position k, compute L2 of remaining):")
        loo_l2s = []
        for k in range(n_t):
            shortened = candidate[:k] + candidate[k+1:]
            l2 = float(np.linalg.norm(target_emb - get_pooled(model, [eot] + shortened, device)))
            delta = l2 - base_l2  # positive = removal hurt (token was useful)
            loo_l2s.append((k, candidate[k], l2, delta))

        # Sort by delta (most useful position first)
        loo_l2s.sort(key=lambda x: -x[3])
        print(f"  {'pos':>4s}  {'token':>12s}  {'L2_without':>11s}  {'delta':>8s}  {'verdict':>10s}")
        for k, tok, l2, delta in loo_l2s:
            tok_str = tokenizer.decode([tok])
            # For the TRUE sequence, check if this position matches
            if label == "TRUE":
                verdict = "correct" if delta > 0 else "???"
            else:
                # For shuffled, check if this position happens to match the true
                is_right = (candidate[k] == bag[k])
                verdict = "RIGHT pos" if is_right else "WRONG pos"
            print(f"  {k:>4d}  {tok_str!r:>12s}  {l2:>11.4f}  {delta:>+8.4f}  {verdict}")

    # ================================================================
    # Now: position-wise search using leave-one-out guidance
    # ================================================================
    print(f"\n{'='*70}")
    print("Position-wise refinement guided by leave-one-out")
    print(f"{'='*70}")

    # Start from the shuffled sequence
    current = list(shuffled)
    current_l2 = float(np.linalg.norm(target_emb - get_pooled(model, [eot] + current, device)))
    print(f"  Starting L2: {current_l2:.4f}")

    for round_num in range(5):
        # Leave-one-out to find the LEAST confident positions
        loo = []
        for k in range(n_t):
            shortened = current[:k] + current[k+1:]
            l2 = float(np.linalg.norm(target_emb - get_pooled(model, [eot] + shortened, device)))
            delta = l2 - current_l2
            loo.append((k, delta))

        # Sort positions by delta (lowest first = least useful = most suspicious)
        loo.sort(key=lambda x: x[1])

        # Try swapping the most suspicious positions with others
        improved = False
        for suspicious_k, suspicious_delta in loo[:5]:  # top-5 most suspicious
            best_swap = None
            best_l2 = current_l2
            for other_k in range(n_t):
                if other_k == suspicious_k:
                    continue
                candidate_seq = list(current)
                candidate_seq[suspicious_k], candidate_seq[other_k] = \
                    candidate_seq[other_k], candidate_seq[suspicious_k]
                c_l2 = float(np.linalg.norm(target_emb - get_pooled(model, [eot] + candidate_seq, device)))
                if c_l2 < best_l2:
                    best_l2 = c_l2
                    best_swap = (suspicious_k, other_k)

            if best_swap is not None:
                i, j = best_swap
                current[i], current[j] = current[j], current[i]
                current_l2 = best_l2
                print(f"  round {round_num+1}: swap suspicious pos {i} with {j} → L2={current_l2:.4f}")
                improved = True
                break  # one swap per round, then re-assess

        if not improved:
            print(f"  round {round_num+1}: no improvement found")
            break

    pm = sum(1 for a, b in zip(current, bag) if a == b)
    strs = [tokenizer.decode([t]) for t in current]
    print(f"\n  Final: L2={current_l2:.4f}  pos-match={pm}/{n_t}")
    print(f"  {' '.join(strs)}")
    true_strs = [tokenizer.decode([t]) for t in bag]
    print(f"  True: {' '.join(true_strs)}")


if __name__ == "__main__":
    main()
