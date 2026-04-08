#!/usr/bin/env python3
"""Forward-pass scoring: given a candidate bag of words, find the best
ordering by forward-passing each candidate sequence and comparing its
pooled embedding to the target.

Pipeline:
  1. Forward-pass the original sentence to get the target embedding
     (mean-pooled trailing L12 hidden states, optionally L2-normalized).
  2. Get a candidate bag of words from per-position L1 decomposition
     (which gives 100% on these sentences).
  3. Generate plausible orderings of that bag using the model's own
     language modeling: at each step, restrict the next token to the
     remaining bag, take the highest-probability remaining token.
  4. Forward-pass the candidate ordering and compare its pooled embedding
     to the target. Report the embedding distance and whether the ordering
     matches the original.
  5. Also try multiple greedy beams and beam search.

This is the "use the model as a verifier" approach.
"""
import gc
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer


SENTENCES = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "Mary had a little lamb",
    "Four score and seven years ago",
    "I love cats and dogs",
    "the big red car drove fast",
]


def get_pooled_l12(model, tokens, device):
    with torch.no_grad():
        out = model(torch.tensor([tokens]).to(device), output_hidden_states=True)
    l12 = out.hidden_states[12].squeeze(0).cpu().numpy()
    pooled = l12[1:].mean(axis=0)  # mean over trailing positions
    return pooled


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def constrained_greedy_order(model, tokenizer, bag, device, eot):
    """Generate the best ordering of `bag` (multiset) by greedy LM:
    at each step, among remaining bag tokens, pick the one with highest
    next-token probability."""
    remaining = list(bag)
    sequence = []
    prefix = [eot]
    while remaining:
        with torch.no_grad():
            out = model(torch.tensor([prefix]).to(device))
        logits = out.logits.squeeze(0)[-1]  # last position's next-token distribution
        # Score each remaining candidate
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


def beam_search_order(model, tokenizer, bag, device, eot, target_embedding, beam_width=4):
    """Beam search over orderings, scoring each by embedding distance to target."""
    # Each beam state: (sequence_so_far, remaining_bag, log_prob_so_far)
    beams = [([], list(bag), 0.0)]

    while beams[0][1]:  # while there are still tokens to place
        new_beams = []
        for seq, remaining, lp in beams:
            prefix = [eot] + seq
            with torch.no_grad():
                out = model(torch.tensor([prefix]).to(device))
            logits = out.logits.squeeze(0)[-1]
            log_probs = F.log_softmax(logits, dim=-1)
            # For each remaining candidate, create a new beam
            seen = set()
            for i, t in enumerate(remaining):
                if t in seen:
                    continue  # skip duplicate token positions in this expansion
                seen.add(t)
                new_remaining = list(remaining)
                new_remaining.pop(i)
                new_seq = seq + [t]
                new_lp = lp + float(log_probs[t])
                new_beams.append((new_seq, new_remaining, new_lp))
        # Keep top beam_width by log prob
        new_beams.sort(key=lambda x: -x[2])
        beams = new_beams[:beam_width]

    # Now score each completed sequence by embedding distance
    scored = []
    for seq, _, lp in beams:
        full = [eot] + seq
        emb = get_pooled_l12(model, full, device)
        d = float(np.linalg.norm(target_embedding - emb))
        c = cos(target_embedding, emb)
        scored.append((seq, lp, d, c))
    scored.sort(key=lambda x: x[2])  # by ascending embedding distance
    return scored


def main():
    device = "cpu"
    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True).to(device)
    model.eval()
    eot = tokenizer.eos_token_id

    for sent in SENTENCES:
        original = [eot] + tokenizer.encode(sent)
        bag = original[1:]  # the trailing tokens (the "bag of words")

        print(f"\n{sent!r}")
        print(f"  original tokens: {[tokenizer.decode([t]) for t in bag]}")

        target_emb = get_pooled_l12(model, original, device)

        # Method 1: greedy LM-ordered
        greedy_seq = constrained_greedy_order(model, tokenizer, bag, device, eot)
        greedy_full = [eot] + greedy_seq
        greedy_emb = get_pooled_l12(model, greedy_full, device)
        greedy_dist = float(np.linalg.norm(target_emb - greedy_emb))
        greedy_cos = cos(target_emb, greedy_emb)
        greedy_match = sum(1 for a, b in zip(greedy_seq, bag) if a == b)
        print(f"  greedy LM order: {[tokenizer.decode([t]) for t in greedy_seq]}")
        print(f"    position-match: {greedy_match}/{len(bag)}, cos={greedy_cos:.4f}, dist={greedy_dist:.2f}")

        # Method 2: beam search by embedding distance
        scored = beam_search_order(model, tokenizer, bag, device, eot, target_emb, beam_width=4)
        best_seq, best_lp, best_dist, best_cos = scored[0]
        best_match = sum(1 for a, b in zip(best_seq, bag) if a == b)
        print(f"  beam-search best: {[tokenizer.decode([t]) for t in best_seq]}")
        print(f"    position-match: {best_match}/{len(bag)}, cos={best_cos:.4f}, dist={best_dist:.2f}")
        # Also show 2nd and 3rd best from beam
        for i, (seq, lp, d, c) in enumerate(scored[1:4]):
            tok_str = [tokenizer.decode([t]) for t in seq]
            mp = sum(1 for a, b in zip(seq, bag) if a == b)
            print(f"    beam #{i+2}:  {tok_str}  match={mp}/{len(bag)} cos={c:.4f}")


if __name__ == "__main__":
    main()
