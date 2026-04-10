#!/usr/bin/env python3
"""Beam width sweep for bag-constrained ordering search.

Re-runs the forward-pass beam search from forward_pass_scoring.py but with
configurable beam width. Reports both position-match AND L2 distance (since
we now know L2 is the discriminative metric, not cosine).

Sweep: B=4 (baseline, what we had), B=8, and optionally B=16.
"""
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


def get_pooled(model, tokens, device):
    with torch.no_grad():
        out = model(torch.tensor([tokens]).to(device), output_hidden_states=True)
    l12 = out.hidden_states[12].squeeze(0)
    return l12[1:].mean(dim=0).cpu().numpy()


def beam_search_order(model, bag, device, eot, target_emb, beam_width=4):
    """Beam search over orderings of bag, scoring by LM log-prob during
    search and ranking final candidates by L2 embedding distance."""
    beams = [([], list(bag), 0.0)]

    while beams[0][1]:  # while remaining tokens exist
        new_beams = []
        for seq, remaining, lp in beams:
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
                new_lp = lp + float(log_probs[t])
                new_beams.append((new_seq, new_remaining, new_lp))
        new_beams.sort(key=lambda x: -x[2])
        beams = new_beams[:beam_width]

    # Score completed sequences by L2 embedding distance
    scored = []
    for seq, _, lp in beams:
        emb = get_pooled(model, [eot] + seq, device)
        l2 = float(np.linalg.norm(target_emb - emb))
        scored.append((seq, lp, l2))
    scored.sort(key=lambda x: x[2])  # rank by L2 (ascending = best)
    return scored


def main():
    device = "cpu"
    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True).to(device)
    model.eval()
    eot = tokenizer.eos_token_id

    beam_widths = [4, 8, 16]

    # Summary table
    summary = {}

    for B in beam_widths:
        print(f"\n{'='*70}")
        print(f"BEAM WIDTH = {B}")
        print(f"{'='*70}")

        total_match = 0
        total_tokens = 0

        for sent in SENTENCES:
            true_tokens = [eot] + tokenizer.encode(sent)
            bag = true_tokens[1:]
            n_t = len(bag)
            target_emb = get_pooled(model, true_tokens, device)

            scored = beam_search_order(model, bag, device, eot, target_emb, beam_width=B)
            best_seq, best_lp, best_l2 = scored[0]
            pos_match = sum(1 for a, b in zip(best_seq, bag) if a == b)
            total_match += pos_match
            total_tokens += n_t
            perfect = "PERFECT" if pos_match == n_t else ""

            best_strs = [tokenizer.decode([t]) for t in best_seq]
            true_strs = [tokenizer.decode([t]) for t in bag]
            print(f"\n  {sent!r}")
            print(f"    best:  {best_strs}  match={pos_match}/{n_t}  L2={best_l2:.4f}  {perfect}")
            if pos_match < n_t:
                print(f"    true:  {true_strs}")
                # Show top 3 for non-perfect cases
                for i, (seq, lp, l2) in enumerate(scored[:3]):
                    tok_str = [tokenizer.decode([t]) for t in seq]
                    pm = sum(1 for a, b in zip(seq, bag) if a == b)
                    print(f"    beam {i+1}: {tok_str}  match={pm}/{n_t}  L2={l2:.4f}")

        summary[B] = (total_match, total_tokens)
        print(f"\n  B={B} total: {total_match}/{total_tokens} position-match "
              f"({100*total_match/total_tokens:.1f}%)")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for B in beam_widths:
        m, t = summary[B]
        n_perfect = sum(
            1 for sent in SENTENCES
            if True  # placeholder — computed below
        )
        print(f"  B={B:>3d}: {m}/{t} position-match ({100*m/t:.1f}%)")


if __name__ == "__main__":
    main()
