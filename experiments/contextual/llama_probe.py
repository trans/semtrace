#!/usr/bin/env python3
"""Probe Llama 3.2 3B via Ollama's embed API.

Five tests to determine whether our GPT-2 Small findings generalize:

  1. Cosine discrimination: are pairwise cosines saturated (0.99+) or
     discriminative? If saturated, cosine is useless (same as GPT-2).
  2. Semantic clustering: do individual word embeddings cluster by meaning?
  3. Additivity: is embed("the cat sat") ≈ embed("the") + embed("cat") + embed("sat")?
  4. Greedy decomposition: can we recover words from a sentence embedding
     using a vocabulary of individual word embeddings?
  5. Leave-one-out: does removing one word from a sentence detectably
     change the embedding? Which words matter most?

All tests use only the public /api/embed endpoint.
"""
import json
import subprocess
import numpy as np
from collections import defaultdict


def embed(text, model="llama3.2:3b"):
    """Get embedding via Ollama /api/embed."""
    resp = subprocess.run(
        ['curl', '-s', 'http://localhost:11434/api/embed', '-d',
         json.dumps({'model': model, 'input': text})],
        capture_output=True, text=True
    )
    data = json.loads(resp.stdout)
    return np.array(data['embeddings'][0])


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def l2(a, b):
    return float(np.linalg.norm(a - b))


def main():
    print("Probing Llama 3.2 3B via Ollama embed API...\n")

    # Quick check: get one embedding and see its properties
    test = embed("hello")
    print(f"Embedding dim: {len(test)}")
    print(f"Embedding norm: {np.linalg.norm(test):.6f}")
    print(f"(norm ≈ 1.0 means L2-normalized)")

    # ================================================================
    # Test 1: Cosine discrimination
    # ================================================================
    print(f"\n{'='*70}")
    print("Test 1: Cosine discrimination between sentence pairs")
    print(f"{'='*70}")

    sentences = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "Mary had a little lamb",
        "Four score and seven years ago",
        "I love cats and dogs",
        "the big red car drove fast",
        "quantum physics is fascinating",
        "she baked a chocolate cake for the party",
        "the stock market crashed yesterday",
        "children played in the garden",
    ]

    embs = {}
    for s in sentences:
        embs[s] = embed(s)

    print(f"\n  Pairwise cosines (sample):")
    pairs = [
        (0, 1, "similar topic (animals)"),
        (0, 6, "unrelated (cat/quantum)"),
        (2, 3, "different content"),
        (4, 0, "related (cats)"),
        (7, 8, "unrelated (cake/stocks)"),
        (1, 9, "similar (park/garden)"),
    ]
    for i, j, label in pairs:
        c = cos(embs[sentences[i]], embs[sentences[j]])
        d = l2(embs[sentences[i]], embs[sentences[j]])
        print(f"    cos={c:.4f}  L2={d:.4f}  {label}")

    # All pairwise stats
    all_cos = []
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            all_cos.append(cos(embs[sentences[i]], embs[sentences[j]]))
    all_cos = np.array(all_cos)
    print(f"\n  All pairwise cosines: mean={all_cos.mean():.4f} min={all_cos.min():.4f} max={all_cos.max():.4f}")

    # Random "sentences" for baseline
    import random
    random.seed(42)
    words = "the a is was on in to and of for it that with at by from up out if do my so we an or no go".split()
    rand_cos = []
    for _ in range(10):
        rs = " ".join(random.choices(words, k=6))
        re = embed(rs)
        rand_cos.append(cos(embs[sentences[0]], re))
    rand_cos = np.array(rand_cos)
    print(f"  Random 6-word vs target: mean={rand_cos.mean():.4f} min={rand_cos.min():.4f} max={rand_cos.max():.4f}")

    # ================================================================
    # Test 2: Semantic clustering
    # ================================================================
    print(f"\n{'='*70}")
    print("Test 2: Semantic clustering of individual word embeddings")
    print(f"{'='*70}")

    probe_words = ["cat", "dog", "king", "queen", "happy", "sad",
                   "run", "walk", "car", "truck", "red", "blue",
                   "water", "fire", "big", "small", "love", "hate",
                   "fish", "bird", "tree", "flower", "sun", "moon"]

    word_embs = {}
    for w in probe_words:
        word_embs[w] = embed(w)

    # For each probe word, find its nearest neighbors
    print(f"\n  Nearest neighbors (among {len(probe_words)} words):")
    for probe in ["cat", "king", "happy", "car", "water", "big"]:
        dists = [(w, cos(word_embs[probe], word_embs[w]))
                 for w in probe_words if w != probe]
        dists.sort(key=lambda x: -x[1])
        top3 = [(w, f"{c:.3f}") for w, c in dists[:3]]
        print(f"    {probe:>6s} → {top3}")

    # ================================================================
    # Test 3: Additivity
    # ================================================================
    print(f"\n{'='*70}")
    print("Test 3: Is embed(sentence) ≈ sum of embed(each word)?")
    print(f"{'='*70}")

    test_sents = [
        "the cat sat on the mat",
        "Mary had a little lamb",
        "I love cats and dogs",
    ]

    for sent in test_sents:
        sent_emb = embed(sent)
        words_list = sent.split()
        word_embs_list = [embed(w) for w in words_list]
        word_sum = np.sum(word_embs_list, axis=0)
        word_sum_norm = word_sum / (np.linalg.norm(word_sum) + 1e-12)

        c = cos(sent_emb, word_sum)
        c_norm = cos(sent_emb, word_sum_norm)
        print(f"  {sent[:40]:42s}  cos(sent, raw_sum)={c:.4f}  cos(sent, norm_sum)={c_norm:.4f}")

    # ================================================================
    # Test 4: Greedy decomposition
    # ================================================================
    print(f"\n{'='*70}")
    print("Test 4: Greedy decomposition against a word vocabulary")
    print(f"{'='*70}")

    # Build a small vocabulary of common words
    vocab_words = list(set(
        "the a an is was were are am be been being have has had do does did "
        "will would shall should can could may might must need "
        "I you he she it we they me him her us them my your his its our their "
        "this that these those who what which where when how why "
        "and but or nor for yet so if then else "
        "in on at to from by with of about between through during "
        "up down out off over under "
        "not no yes all any some every each much many more most "
        "cat dog sat mat ran park little lamb had Mary go went sure snow "
        "white fleece everywhere love cats dogs big red car drove fast "
        "old man fished boat sun rose slowly morning village sleepy warm "
        "children played colorful kite bird tree water fire king queen "
        "happy sad run walk truck blue flower moon score seven years ago "
        "four new good great small".split()
    ))
    print(f"  Vocabulary: {len(vocab_words)} words")

    print("  Embedding vocabulary...")
    vocab_embs = {}
    for w in vocab_words:
        vocab_embs[w] = embed(w)
    vocab_matrix = np.array([vocab_embs[w] for w in vocab_words])
    vocab_norms = np.linalg.norm(vocab_matrix, axis=1)
    vocab_norms[vocab_norms < 1e-10] = 1.0

    def decompose_greedy(target, max_steps=15):
        residual = target.copy()
        recovered = []
        prev_norm = float("inf")
        for _ in range(max_steps):
            r_norm = np.linalg.norm(residual)
            if r_norm < 0.001 or r_norm > prev_norm:
                break
            prev_norm = r_norm
            sims = vocab_matrix @ residual / (vocab_norms * r_norm)
            best = int(np.argmax(sims))
            recovered.append(vocab_words[best])
            residual = residual - vocab_matrix[best]
        return recovered

    for sent in ["the cat sat on the mat", "Mary had a little lamb",
                 "I love cats and dogs", "the big red car drove fast"]:
        sent_emb = embed(sent)
        recovered = decompose_greedy(sent_emb, max_steps=12)
        true_words = set(sent.lower().split())
        rec_set = set(w.lower() for w in recovered)
        overlap = len(true_words & rec_set)
        print(f"  {sent[:35]:37s}  found: {recovered[:8]}  overlap: {overlap}/{len(true_words)}")

    # ================================================================
    # Test 5: Leave-one-out
    # ================================================================
    print(f"\n{'='*70}")
    print("Test 5: Leave-one-out — which word contributes most?")
    print(f"{'='*70}")

    for sent in ["the cat sat on the mat", "Mary had a little lamb"]:
        sent_emb = embed(sent)
        words_list = sent.split()
        print(f"\n  {sent!r}")
        print(f"  {'removed':>15s}  {'cos_without':>12s}  {'delta_from_1':>13s}")

        for i, w in enumerate(words_list):
            remaining = " ".join(words_list[:i] + words_list[i+1:])
            rem_emb = embed(remaining)
            c = cos(sent_emb, rem_emb)
            delta = 1.0 - c
            print(f"  {w:>15s}  {c:>12.6f}  {delta:>13.6f}")


if __name__ == "__main__":
    main()
