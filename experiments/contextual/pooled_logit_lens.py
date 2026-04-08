#!/usr/bin/env python3
"""Pooled logit lens: project the pooled L12 embedding through wte.T and
read top-K tokens. This is what you get for free in the production threat
model — no W training needed, just mean-pool the L12 hidden states and
multiply by wte.T.

Compare to:
  - Our bridge method (which approximates this with a learned W)
  - Per-position logit lens (which uses ALL positions' logits, not just the mean)
"""
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


def main():
    device = "cpu"
    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True).to(device)
    model.eval()
    eot = tokenizer.eos_token_id

    wte = model.transformer.wte.weight.detach().cpu().numpy()  # (vocab, 768)

    total_unique = 0
    hits_pooled_top_n = 0
    hits_pooled_top_2n = 0
    hits_pooled_top_3n = 0

    for sent in SENTENCES:
        tt = [eot] + tokenizer.encode(sent)
        with torch.no_grad():
            out = model(torch.tensor([tt]).to(device))
        l12 = out.hidden_states[12].squeeze(0).cpu().numpy()  # (n+1, 768)

        trailing = l12[1:]  # (n, 768)
        n = trailing.shape[0]
        unique = set(tt[1:])
        u = len(unique)
        total_unique += u

        # Pooled vector
        pooled = trailing.mean(axis=0)
        # Project through wte.T
        scores = wte @ pooled  # (vocab,)
        # Take top-N predicted tokens
        top_n = np.argsort(-scores)[:n]
        top_2n = np.argsort(-scores)[:2*n]
        top_3n = np.argsort(-scores)[:3*n]

        h_n = len(unique & set(top_n.tolist()))
        h_2n = len(unique & set(top_2n.tolist()))
        h_3n = len(unique & set(top_3n.tolist()))
        hits_pooled_top_n += h_n
        hits_pooled_top_2n += h_2n
        hits_pooled_top_3n += h_3n

    print(f"\nPooled L12 @ wte.T, taking top-K predicted tokens (no training, no W):")
    print(f"  top-N:  {hits_pooled_top_n}/{total_unique}  ({100*hits_pooled_top_n/total_unique:.1f}%)")
    print(f"  top-2N: {hits_pooled_top_2n}/{total_unique}  ({100*hits_pooled_top_2n/total_unique:.1f}%)")
    print(f"  top-3N: {hits_pooled_top_3n}/{total_unique}  ({100*hits_pooled_top_3n/total_unique:.1f}%)")

    # Now try with L11 too
    print(f"\nFor comparison, L11 @ wte.T (need to ln_f first):")
    ln_f = model.transformer.ln_f
    h11_total = 0
    for sent in SENTENCES:
        tt = [eot] + tokenizer.encode(sent)
        with torch.no_grad():
            out = model(torch.tensor([tt]).to(device))
        l11 = out.hidden_states[11].squeeze(0)
        with torch.no_grad():
            l11_norm = ln_f(l11).cpu().numpy()
        trailing = l11_norm[1:]
        n = trailing.shape[0]
        unique = set(tt[1:])
        pooled = trailing.mean(axis=0)
        scores = wte @ pooled
        top_n = np.argsort(-scores)[:n]
        h11_total += len(unique & set(top_n.tolist()))
    print(f"  L11 (ln_f applied) top-N: {h11_total}/{total_unique}  ({100*h11_total/total_unique:.1f}%)")


if __name__ == "__main__":
    main()
