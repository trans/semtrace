#!/usr/bin/env python3
"""Verify the gradient-descent finding and calibrate the cosine baseline.

Two questions:

1. Did the gradient-descent garbage tokens really have cos 0.997 to the
   target? Take them, forward-pass independently, compute pooled L12,
   compare to target. Verify the number.

2. What's the baseline cosine for unrelated sentences in pooled L12 space?
   If random sequences typically have cos 0.95+, then "we found something
   at 0.997" isn't a strong inversion claim — it's noise floor.

We test:
  - Self-cosine (sanity, should be 1.0)
  - Cosine to N random 6-token sequences
  - Cosine to other meaningful sentences in our test set
  - Cosine to the gradient-descent garbage (if we re-run a few)
  - Cosine to permutations of the same tokens
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


def get_pooled_embedding(model, tokens, device):
    with torch.no_grad():
        out = model(torch.tensor([tokens]).to(device), output_hidden_states=True)
    l12 = out.hidden_states[12].squeeze(0)
    return l12[1:].mean(dim=0).cpu().numpy()


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    device = "cpu"
    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True).to(device)
    model.eval()
    eot = tokenizer.eos_token_id
    vocab_size = model.transformer.wte.weight.shape[0]

    rng = np.random.default_rng(42)

    print("\nComputing target embeddings for the test sentences...")
    sentence_data = []
    for sent in SENTENCES:
        tokens = [eot] + tokenizer.encode(sent)
        emb = get_pooled_embedding(model, tokens, device)
        sentence_data.append((sent, tokens, emb))

    print("\n" + "="*78)
    print("Test 1: Self-cosine (sanity check)")
    print("="*78)
    for sent, _, emb in sentence_data:
        c = cos(emb, emb)
        print(f"  {sent[:40]:42s}  cos to self: {c:.6f}")

    print("\n" + "="*78)
    print("Test 2: Cosine between meaningful sentences in our test set")
    print("="*78)
    for i, (s1, _, e1) in enumerate(sentence_data):
        for j, (s2, _, e2) in enumerate(sentence_data):
            if i >= j:
                continue
            c = cos(e1, e2)
            print(f"  cos({s1[:25]!r}, {s2[:25]!r}) = {c:.4f}")

    print("\n" + "="*78)
    print("Test 3: Cosine to random N-token sequences")
    print("="*78)
    for sent, true_tokens, target_emb in sentence_data:
        n_t = len(true_tokens) - 1  # trailing positions
        cosines = []
        for trial in range(50):
            random_tokens = [eot] + rng.integers(0, vocab_size, size=n_t).tolist()
            r_emb = get_pooled_embedding(model, random_tokens, device)
            cosines.append(cos(target_emb, r_emb))
        cosines = np.array(cosines)
        print(f"  {sent[:38]:40s} (n={n_t}): "
              f"random-cos mean={cosines.mean():.4f} min={cosines.min():.4f} max={cosines.max():.4f}")

    print("\n" + "="*78)
    print("Test 4: Cosine to the same bag in different orderings")
    print("="*78)
    for sent, true_tokens, target_emb in sentence_data:
        bag = true_tokens[1:]
        n_t = len(bag)
        # Try several random permutations
        cosines = []
        for trial in range(20):
            perm = list(bag)
            rng.shuffle(perm)
            perm_full = [eot] + perm
            p_emb = get_pooled_embedding(model, perm_full, device)
            cosines.append(cos(target_emb, p_emb))
        cosines = np.array(cosines)
        print(f"  {sent[:38]:40s}  perm-cos mean={cosines.mean():.4f} "
              f"min={cosines.min():.4f} max={cosines.max():.4f}")

    print("\n" + "="*78)
    print("Test 5: Reproduce a specific 'garbage' result from gradient descent")
    print("="*78)
    # From gradient_invert_v2 output for "the big red car drove fast":
    # Recovered: ['ulty', ' Im', 'Driver', 'rawdownload', ' builds', ' Goodell']
    garbage_tokens_str = ['ulty', ' Im', 'Driver', 'rawdownload', ' builds', ' Goodell']
    garbage_ids = []
    for s in garbage_tokens_str:
        ids = tokenizer.encode(s)
        if len(ids) == 1:
            garbage_ids.append(ids[0])
        else:
            garbage_ids.append(ids[0])  # take first if multi-token
    garbage_full = [eot] + garbage_ids
    garbage_emb = get_pooled_embedding(model, garbage_full, device)

    # Compare to "the big red car drove fast" target
    target_idx = SENTENCES.index("the big red car drove fast")
    real_emb = sentence_data[target_idx][2]

    c = cos(garbage_emb, real_emb)
    d = float(np.sum((garbage_emb - real_emb) ** 2))
    print(f"  garbage tokens: {garbage_tokens_str}")
    print(f"  cos to 'the big red car drove fast': {c:.6f}")
    print(f"  L2 distance: {d:.4f}")

    # Compare against random baselines for same sentence
    n_t = 6
    rand_cosines = []
    for trial in range(100):
        random_tokens = [eot] + rng.integers(0, vocab_size, size=n_t).tolist()
        r_emb = get_pooled_embedding(model, random_tokens, device)
        rand_cosines.append(cos(real_emb, r_emb))
    rand_cosines = np.array(rand_cosines)
    print(f"  baseline (100 random): mean={rand_cosines.mean():.4f} max={rand_cosines.max():.4f}")
    pct = (rand_cosines >= c).sum() / len(rand_cosines)
    print(f"  fraction of random sequences with cos >= garbage: {pct*100:.1f}%")


if __name__ == "__main__":
    main()
