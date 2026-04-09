#!/usr/bin/env python3
"""Test whether L2 distance is a more discriminating metric than cosine
in pooled L12 space.

The verify_cosine_baseline experiment showed that cosine in pooled L12 is
saturated at ~0.99 for any pair of sequences (real or random). The
gradient-descent "garbage" had cos 0.997 to target but 26% of random
sequences scored higher.

This experiment asks: does L2 distance separate the right answer from
random sequences cleanly?

Tests:
  1. L2 baseline: distribution of L2(target, random_sequence) for many
     random sequences, and L2(target, target) for sanity.
  2. L2 of meaningful but unrelated sentences vs L2 of random.
  3. L2 of permutations of the right bag vs L2 of random.
  4. Re-run gradient descent with L2 distance loss (we already use L2!),
     but verify by ranking the result against the random baseline.
  5. Re-run forward-pass beam search and rank candidates by L2 distance
     (which it already does internally) — verify the metric is actually
     useful for ranking.
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

    print("\nComputing target embeddings...")
    sentence_data = []
    for sent in SENTENCES:
        tokens = [eot] + tokenizer.encode(sent)
        emb = get_pooled_embedding(model, tokens, device)
        sentence_data.append((sent, tokens, emb))

    # ============================================================
    # Distribution of L2 distances for random and meaningful sequences
    # ============================================================
    print(f"\n{'='*78}")
    print("L2 distance distributions")
    print(f"{'='*78}")
    print(f"  {'sentence':40s}  {'norm':>7s}  {'rand mean':>10s}  {'rand min':>9s}  {'self':>6s}")
    print("  " + "-" * 80)

    for sent, true_tokens, target_emb in sentence_data:
        n_t = len(true_tokens) - 1
        target_norm = float(np.linalg.norm(target_emb))

        # L2 to random sequences
        rand_l2s = []
        for trial in range(100):
            random_tokens = [eot] + rng.integers(0, vocab_size, size=n_t).tolist()
            r_emb = get_pooled_embedding(model, random_tokens, device)
            rand_l2s.append(l2(target_emb, r_emb))
        rand_l2s = np.array(rand_l2s)

        # Self L2 (should be 0)
        self_l2 = l2(target_emb, target_emb)

        print(f"  {sent[:38]:40s}  {target_norm:>7.1f}  {rand_l2s.mean():>10.2f}  {rand_l2s.min():>9.2f}  {self_l2:>6.2f}")

    # ============================================================
    # L2 of meaningful sentence pairs
    # ============================================================
    print(f"\n{'='*78}")
    print("L2 between meaningful sentence pairs (in test set)")
    print(f"{'='*78}")
    for i, (s1, _, e1) in enumerate(sentence_data):
        for j, (s2, _, e2) in enumerate(sentence_data):
            if i >= j:
                continue
            print(f"  L2({s1[:25]!r}, {s2[:25]!r}) = {l2(e1, e2):.2f}")

    # ============================================================
    # L2 of bag permutations vs random
    # ============================================================
    print(f"\n{'='*78}")
    print("L2 of permutations of the SAME bag vs L2 of random sequences")
    print(f"{'='*78}")
    for sent, true_tokens, target_emb in sentence_data:
        bag = true_tokens[1:]
        n_t = len(bag)
        # Permutations
        perm_l2s = []
        for trial in range(20):
            perm = list(bag)
            rng.shuffle(perm)
            p_emb = get_pooled_embedding(model, [eot] + perm, device)
            perm_l2s.append(l2(target_emb, p_emb))
        perm_l2s = np.array(perm_l2s)

        # Random
        rand_l2s = []
        for trial in range(20):
            random_tokens = [eot] + rng.integers(0, vocab_size, size=n_t).tolist()
            r_emb = get_pooled_embedding(model, random_tokens, device)
            rand_l2s.append(l2(target_emb, r_emb))
        rand_l2s = np.array(rand_l2s)

        print(f"  {sent[:38]:40s}  perm L2: mean={perm_l2s.mean():.2f} min={perm_l2s.min():.2f}  "
              f"random L2: mean={rand_l2s.mean():.2f} min={rand_l2s.min():.2f}")

    # ============================================================
    # The garbage tokens from gradient descent — proper ranking
    # ============================================================
    print(f"\n{'='*78}")
    print("Gradient descent garbage: where does it rank by L2?")
    print(f"{'='*78}")
    garbage_strs = ['ulty', ' Im', 'Driver', 'rawdownload', ' builds', ' Goodell']
    garbage_ids = []
    for s in garbage_strs:
        ids = tokenizer.encode(s)
        garbage_ids.append(ids[0])
    garbage_emb = get_pooled_embedding(model, [eot] + garbage_ids, device)

    target_idx = SENTENCES.index("the big red car drove fast")
    target_emb = sentence_data[target_idx][2]
    n_t = 6

    garbage_cos = cos(garbage_emb, target_emb)
    garbage_l2 = l2(garbage_emb, target_emb)

    rand_l2s = []
    rand_coss = []
    for trial in range(100):
        random_tokens = [eot] + rng.integers(0, vocab_size, size=n_t).tolist()
        r_emb = get_pooled_embedding(model, random_tokens, device)
        rand_l2s.append(l2(target_emb, r_emb))
        rand_coss.append(cos(target_emb, r_emb))
    rand_l2s = np.array(rand_l2s)
    rand_coss = np.array(rand_coss)

    print(f"  Garbage tokens: {garbage_strs}")
    print(f"  vs 'the big red car drove fast':")
    print(f"    cos:  garbage={garbage_cos:.6f}  random mean={rand_coss.mean():.6f}  random min={rand_coss.min():.6f}")
    print(f"    L2:   garbage={garbage_l2:.4f}     random mean={rand_l2s.mean():.4f}     random min={rand_l2s.min():.4f}")

    # Percentile
    pct_cos_better = (rand_coss >= garbage_cos).sum() / 100
    pct_l2_better = (rand_l2s <= garbage_l2).sum() / 100
    print(f"  Percentile of garbage among random sequences:")
    print(f"    by cosine (% of random with HIGHER cos): {pct_cos_better*100:.1f}%")
    print(f"    by L2 (% of random with LOWER L2):       {pct_l2_better*100:.1f}%")

    # Try: the right answer's L2 to itself
    right_full = sentence_data[target_idx][1]
    right_emb_recheck = get_pooled_embedding(model, right_full, device)
    print(f"  Right answer L2 to itself: {l2(right_emb_recheck, target_emb):.4f}")

    # ============================================================
    # The forward-pass-beam result that got "wrong order, cos 0.999"
    # — was its L2 also low?
    # ============================================================
    print(f"\n{'='*78}")
    print("Forward-pass beam wrong-order result: L2 vs cosine")
    print(f"{'='*78}")
    # From forward_pass_scoring.py output:
    # 'the cat on the mat sat' (wrong order of right bag), cos 0.9993
    wrong_order_str = "the cat on the mat sat"
    wrong_order_tokens = [eot] + tokenizer.encode(wrong_order_str)
    wrong_order_emb = get_pooled_embedding(model, wrong_order_tokens, device)

    cat_idx = SENTENCES.index("the cat sat on the mat")
    cat_target = sentence_data[cat_idx][2]
    n_t = 6

    wo_cos = cos(wrong_order_emb, cat_target)
    wo_l2 = l2(wrong_order_emb, cat_target)

    rand_l2s = []
    for trial in range(100):
        random_tokens = [eot] + rng.integers(0, vocab_size, size=n_t).tolist()
        r_emb = get_pooled_embedding(model, random_tokens, device)
        rand_l2s.append(l2(cat_target, r_emb))
    rand_l2s = np.array(rand_l2s)

    print(f"  'the cat on the mat sat' vs 'the cat sat on the mat' target:")
    print(f"    cos: {wo_cos:.6f}")
    print(f"    L2:  {wo_l2:.4f}")
    print(f"    random L2 mean: {rand_l2s.mean():.4f}, min: {rand_l2s.min():.4f}")
    pct_l2 = (rand_l2s <= wo_l2).sum() / 100
    print(f"    percent of random with LOWER L2 than wrong-order: {pct_l2*100:.1f}%")


if __name__ == "__main__":
    main()
