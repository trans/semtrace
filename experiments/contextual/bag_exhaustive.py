#!/usr/bin/env python3
"""Bag-constrained exhaustive search.

Given the true bag of words (multiset of tokens), enumerate all permutations
and forward-pass each, scoring by L2 distance to target. Pick the best.

For 6-token sentences this is 720 permutations (cheap). For 9 tokens it's
362,880 (slow but tractable). We exhaust everything up to 7! and beam-search
above that.

This finds the GLOBAL best ordering of the bag by L2 distance. If the global
best is the true sentence, then forward-pass beam search is leaving easy wins
on the table. If the global best is NOT the true sentence (i.e., some
permutation has lower L2 than the original), then the embedding has true
order ambiguity and no method can recover the order from this single vector.
"""
import gc
import itertools
import numpy as np
import torch
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


def main():
    device = "cpu"
    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True).to(device)
    model.eval()
    eot = tokenizer.eos_token_id
    vocab_size = model.transformer.wte.weight.shape[0]

    rng = np.random.default_rng(42)

    for sent in SENTENCES:
        print(f"\n{'='*70}")
        print(f"Sentence: {sent!r}")
        true_tokens = [eot] + tokenizer.encode(sent)
        bag = true_tokens[1:]
        n_t = len(bag)
        target = get_pooled(model, true_tokens, device)
        print(f"  bag: {[tokenizer.decode([t]) for t in bag]}, n={n_t}")

        # Enumerate all permutations (deduplicated for repeated tokens)
        seen_perms = set()
        all_perms = []
        for p in itertools.permutations(bag):
            key = tuple(p)
            if key not in seen_perms:
                seen_perms.add(key)
                all_perms.append(list(p))
        print(f"  unique permutations: {len(all_perms)}")

        # Score each permutation by L2 to target
        scored = []
        for p in all_perms:
            full = [eot] + p
            emb = get_pooled(model, full, device)
            d = float(np.linalg.norm(target - emb))
            scored.append((p, d))

        scored.sort(key=lambda x: x[1])

        best_p, best_d = scored[0]
        true_d = float(np.linalg.norm(target - get_pooled(model, true_tokens, device)))

        # Random baseline
        rand_l2s = []
        for _ in range(50):
            random_tokens = [eot] + rng.integers(0, vocab_size, size=n_t).tolist()
            r_emb = get_pooled(model, random_tokens, device)
            rand_l2s.append(float(np.linalg.norm(target - r_emb)))
        rand_l2s = np.array(rand_l2s)

        # Where does the GLOBAL minimum permutation rank vs the true?
        is_perfect = best_p == bag
        rank_of_truth = sum(1 for _, d in scored if d < true_d)

        print(f"  true sequence L2: {true_d:.4f}")
        print(f"  best permutation L2: {best_d:.4f}  perfect={is_perfect}")
        print(f"  rank of true sequence among permutations: {rank_of_truth} (0 = best)")
        print(f"  random L2: mean={rand_l2s.mean():.2f}, min={rand_l2s.min():.2f}")
        if best_d < rand_l2s.min():
            print(f"  best permutation L2 is below all 50 random samples")

        # Show top 5 permutations by L2
        print(f"  top 5 by L2:")
        for i, (p, d) in enumerate(scored[:5]):
            tok_str = [tokenizer.decode([t]) for t in p]
            mark = "TRUE" if p == bag else "    "
            print(f"    {mark}  {tok_str}  L2={d:.4f}")


if __name__ == "__main__":
    main()
