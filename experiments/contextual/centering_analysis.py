#!/usr/bin/env python3
"""Centering analysis: subtract the global mean of pooled L12 embeddings.

Hypothesis: cosine in pooled L12 is saturated at 0.99+ because all
pooled vectors share a strong common direction (the centroid). If we
subtract the centroid from each vector, the residuals should have
much more discriminative cosine.

Tests:
  1. Compute the centroid of pooled L12 across many sentences (use a
     larger 50+ sentence pool to get a stable estimate).
  2. For our test pairs, compute cosine and L2 BEFORE centering and AFTER.
  3. Check whether centered cosine separates real-vs-real, real-vs-random,
     and gradient-descent-garbage from each other better than raw cosine.
"""
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# Use a wider pool for centroid estimation
CENTROID_POOL = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "Mary had a little lamb",
    "Four score and seven years ago",
    "I love cats and dogs",
    "the big red car drove fast",
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
    "the moon rose above the silver lake",
    "yesterday I went to the market",
    "the bread is fresh and warm",
    "she dances gracefully across the stage",
    "the bell tower stood for centuries",
    "thunder rolled across the distant hills",
    "two birds sang from the willow tree",
    "the kettle whistled on the stove",
    "snowflakes drifted past the window",
    "the lighthouse beam swept the sea",
    "we crossed the river at dawn",
    "the carpenter sanded the oak board",
    "violins played a soft melody",
    "tall grass swayed in the breeze",
    "the inn served hot stew and bread",
    "the captain steered toward the harbor",
    "petals fell from the cherry tree",
    "an owl hooted in the dark forest",
    "the librarian shelved the old volumes",
    "frost coated the morning windows",
]


def get_pooled(model, tokens, device):
    with torch.no_grad():
        out = model(torch.tensor([tokens]).to(device), output_hidden_states=True)
    l12 = out.hidden_states[12].squeeze(0)
    return l12[1:].mean(dim=0).cpu().numpy()


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def l2(a, b):
    return float(np.linalg.norm(a - b))


def main():
    device = "cpu"
    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True).to(device)
    model.eval()
    eot = tokenizer.eos_token_id
    vocab_size = model.transformer.wte.weight.shape[0]

    rng = np.random.default_rng(42)

    # Compute centroid from the pool
    print(f"\nComputing centroid from {len(CENTROID_POOL)} sentences...")
    pool_embs = []
    for sent in CENTROID_POOL:
        tokens = [eot] + tokenizer.encode(sent)
        pool_embs.append(get_pooled(model, tokens, device))
    pool_embs = np.array(pool_embs)
    centroid = pool_embs.mean(axis=0)
    print(f"  centroid norm: {np.linalg.norm(centroid):.2f}")
    print(f"  pool emb norms: mean {np.linalg.norm(pool_embs, axis=1).mean():.2f}")

    # Centered version of pool embeddings
    pool_centered = pool_embs - centroid
    print(f"  centered emb norms: mean {np.linalg.norm(pool_centered, axis=1).mean():.2f}")

    # ============================================================
    # Test 1: Pairwise cosines among the pool — raw vs centered
    # ============================================================
    print(f"\n{'='*78}")
    print("Test 1: Pairwise cosines among the centroid pool sentences")
    print(f"{'='*78}")

    # Compute pairwise cosines for both versions
    raw_coss = []
    centered_coss = []
    for i in range(len(pool_embs)):
        for j in range(i+1, len(pool_embs)):
            raw_coss.append(cos(pool_embs[i], pool_embs[j]))
            centered_coss.append(cos(pool_centered[i], pool_centered[j]))
    raw_coss = np.array(raw_coss)
    centered_coss = np.array(centered_coss)

    print(f"  raw cosines: mean={raw_coss.mean():.4f} min={raw_coss.min():.4f} max={raw_coss.max():.4f}")
    print(f"  centered:    mean={centered_coss.mean():.4f} min={centered_coss.min():.4f} max={centered_coss.max():.4f}")

    # ============================================================
    # Test 2: Random sequences — cosine to "the cat sat on the mat"
    # ============================================================
    print(f"\n{'='*78}")
    print("Test 2: Random sequences vs target — raw cosine vs centered cosine")
    print(f"{'='*78}")

    target_sent = "the cat sat on the mat"
    target_tokens = [eot] + tokenizer.encode(target_sent)
    target_emb = get_pooled(model, target_tokens, device)
    target_centered = target_emb - centroid
    n_t = len(target_tokens) - 1

    raw_rand = []
    centered_rand = []
    for trial in range(50):
        random_tokens = [eot] + rng.integers(0, vocab_size, size=n_t).tolist()
        r_emb = get_pooled(model, random_tokens, device)
        r_centered = r_emb - centroid
        raw_rand.append(cos(target_emb, r_emb))
        centered_rand.append(cos(target_centered, r_centered))
    raw_rand = np.array(raw_rand)
    centered_rand = np.array(centered_rand)

    # Compare to permutations of the same bag
    raw_perm = []
    centered_perm = []
    for trial in range(50):
        bag = list(target_tokens[1:])
        rng.shuffle(bag)
        p_emb = get_pooled(model, [eot] + bag, device)
        p_centered = p_emb - centroid
        raw_perm.append(cos(target_emb, p_emb))
        centered_perm.append(cos(target_centered, p_centered))
    raw_perm = np.array(raw_perm)
    centered_perm = np.array(centered_perm)

    print(f"  Random vs target:")
    print(f"    raw cos:      mean={raw_rand.mean():.4f} min={raw_rand.min():.4f} max={raw_rand.max():.4f}")
    print(f"    centered cos: mean={centered_rand.mean():.4f} min={centered_rand.min():.4f} max={centered_rand.max():.4f}")
    print(f"  Permutations vs target:")
    print(f"    raw cos:      mean={raw_perm.mean():.4f} min={raw_perm.min():.4f} max={raw_perm.max():.4f}")
    print(f"    centered cos: mean={centered_perm.mean():.4f} min={centered_perm.min():.4f} max={centered_perm.max():.4f}")

    # Discrimination ratio: how well does centered cosine separate perms from random?
    # Higher centered_perm cosine and lower centered_rand cosine = better separation
    print(f"\n  Discrimination (perm cos - random cos):")
    print(f"    raw:      {raw_perm.mean() - raw_rand.mean():+.4f}")
    print(f"    centered: {centered_perm.mean() - centered_rand.mean():+.4f}")

    # ============================================================
    # Test 3: The gradient-descent garbage — raw vs centered ranking
    # ============================================================
    print(f"\n{'='*78}")
    print("Test 3: Gradient-descent garbage — does centering reveal it as bad?")
    print(f"{'='*78}")

    # Garbage from "the big red car drove fast" gradient descent
    big_red_target = "the big red car drove fast"
    big_red_tokens = [eot] + tokenizer.encode(big_red_target)
    br_emb = get_pooled(model, big_red_tokens, device)
    br_centered = br_emb - centroid

    garbage_strs = ['ulty', ' Im', 'Driver', 'rawdownload', ' builds', ' Goodell']
    garbage_ids = [tokenizer.encode(s)[0] for s in garbage_strs]
    garbage_emb = get_pooled(model, [eot] + garbage_ids, device)
    garbage_centered = garbage_emb - centroid

    print(f"  Garbage: {garbage_strs}")
    print(f"  vs 'the big red car drove fast':")
    print(f"    raw cos:      {cos(garbage_emb, br_emb):.6f}")
    print(f"    centered cos: {cos(garbage_centered, br_centered):.6f}")
    print(f"    raw L2:       {l2(garbage_emb, br_emb):.4f}")
    print(f"    centered L2:  {l2(garbage_centered, br_centered):.4f}")

    # Random baseline for both metrics
    raw_rand_br = []
    centered_rand_br = []
    rand_l2_raw = []
    rand_l2_centered = []
    for trial in range(50):
        random_tokens = [eot] + rng.integers(0, vocab_size, size=6).tolist()
        r_emb = get_pooled(model, random_tokens, device)
        r_centered = r_emb - centroid
        raw_rand_br.append(cos(br_emb, r_emb))
        centered_rand_br.append(cos(br_centered, r_centered))
        rand_l2_raw.append(l2(br_emb, r_emb))
        rand_l2_centered.append(l2(br_centered, r_centered))
    raw_rand_br = np.array(raw_rand_br)
    centered_rand_br = np.array(centered_rand_br)
    rand_l2_raw = np.array(rand_l2_raw)
    rand_l2_centered = np.array(rand_l2_centered)

    pct_raw = (raw_rand_br >= cos(garbage_emb, br_emb)).sum() / 50
    pct_centered = (centered_rand_br >= cos(garbage_centered, br_centered)).sum() / 50
    pct_l2_raw = (rand_l2_raw <= l2(garbage_emb, br_emb)).sum() / 50
    pct_l2_centered = (rand_l2_centered <= l2(garbage_centered, br_centered)).sum() / 50

    print(f"  Percentile of garbage vs 50 random sequences:")
    print(f"    raw cos (% of random with HIGHER cos):       {pct_raw*100:.0f}%")
    print(f"    centered cos (% of random with HIGHER cos):  {pct_centered*100:.0f}%")
    print(f"    raw L2 (% of random with LOWER L2):          {pct_l2_raw*100:.0f}%")
    print(f"    centered L2 (% of random with LOWER L2):     {pct_l2_centered*100:.0f}%")

    # ============================================================
    # Test 4: Test pair distinguishability after centering
    # ============================================================
    print(f"\n{'='*78}")
    print("Test 4: Pair distinguishability — semantically similar vs dissimilar")
    print(f"{'='*78}")

    pairs = [
        ("the cat sat on the mat", "the cat sat on the rug", "synonym swap"),
        ("the dog chased the cat", "the cat chased the dog", "swap order"),
        ("I am happy", "I am not happy", "negation"),
        ("the cat sat on the mat", "quantum physics is fascinating", "unrelated"),
        ("the cat sat on the mat", "the dog ran in the park", "different content same topic"),
    ]

    print(f"  {'pair':50s}  {'raw cos':>9s}  {'cent cos':>9s}  {'raw L2':>8s}  {'cent L2':>8s}")
    print("  " + "-" * 95)
    for s1, s2, label in pairs:
        e1 = get_pooled(model, [eot] + tokenizer.encode(s1), device)
        e2 = get_pooled(model, [eot] + tokenizer.encode(s2), device)
        c1 = e1 - centroid
        c2 = e2 - centroid
        print(f"  {label:50s}  {cos(e1, e2):>9.4f}  {cos(c1, c2):>9.4f}  {l2(e1, e2):>8.2f}  {l2(c1, c2):>8.2f}")


if __name__ == "__main__":
    main()
