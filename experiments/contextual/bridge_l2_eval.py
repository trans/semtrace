#!/usr/bin/env python3
"""Re-evaluate bridge + CD using L2 distance to target.

Previous bridge evaluations reported bag-match (token identity) which gave
~33% at L12. But cosine in pooled L12 is saturated, and L2 may be more
discriminating. This re-runs the bridge methods and reports BOTH:
  - Token-level bag match (the original metric)
  - L2 distance from the recovered sequence's pooled L12 to the target
  - Comparison to a random baseline L2

If the bridge gets bag match 33% but L2 is much lower than random, it
means the bridge IS finding meaningful answers — just not perfect token
identity. That would change how we interpret it.
"""
import gc
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


def get_pooled(model, tokens, device):
    with torch.no_grad():
        out = model(torch.tensor([tokens]).to(device), output_hidden_states=True)
    l12 = out.hidden_states[12].squeeze(0)
    return l12[1:].mean(dim=0).cpu().numpy()


def decompose_greedy(target, embeddings, max_steps):
    residual = target.copy()
    recovered = []
    prev_norm = float("inf")
    for _ in range(max_steps):
        r_norm = np.linalg.norm(residual)
        if r_norm < 0.001 or r_norm > prev_norm:
            break
        prev_norm = r_norm
        norms = np.linalg.norm(embeddings, axis=1)
        norms[norms < 1e-10] = 1.0
        sims = embeddings @ residual / (norms * r_norm)
        best = int(np.argmax(sims))
        recovered.append(best)
        residual = residual - embeddings[best]
    return recovered


def coord_descent(target, vocab, n_tokens, max_iters=15):
    current = decompose_greedy(target, vocab, n_tokens)
    while len(current) < n_tokens:
        ideal = target - vocab[current].sum(axis=0)
        nrm = np.linalg.norm(vocab, axis=1)
        nrm[nrm < 1e-10] = 1.0
        sims = vocab @ ideal / nrm
        current.append(int(np.argmax(sims)))
    current = current[:n_tokens]
    best_dist = np.linalg.norm(target - vocab[current].sum(axis=0))
    for _ in range(max_iters):
        improved = False
        for pos in range(n_tokens):
            others = [current[j] for j in range(n_tokens) if j != pos]
            others_sum = vocab[others].sum(axis=0)
            ideal = target - others_sum
            nrm = np.linalg.norm(vocab, axis=1)
            nrm[nrm < 1e-10] = 1.0
            sims = vocab @ ideal / nrm
            best_for_pos = int(np.argmax(sims))
            if best_for_pos != current[pos]:
                new_tokens = list(current)
                new_tokens[pos] = best_for_pos
                new_dist = np.linalg.norm(target - vocab[new_tokens].sum(axis=0))
                if new_dist < best_dist:
                    current[pos] = best_for_pos
                    best_dist = new_dist
                    improved = True
        if not improved:
            break
    return current


def build_vocab_pos1(model, layer, device, prefix_token, batch=256):
    vocab_size = model.transformer.wte.weight.shape[0]
    chunks = []
    for start in range(0, vocab_size, batch):
        end = min(start + batch, vocab_size)
        ids = torch.tensor([[prefix_token, t] for t in range(start, end)]).to(device)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        h = out.hidden_states[layer][:, 1, :].cpu().numpy()
        chunks.append(h)
    return np.concatenate(chunks, axis=0)


def main():
    device = "cpu"
    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True).to(device)
    model.eval()
    eot = tokenizer.eos_token_id
    vocab_size = model.transformer.wte.weight.shape[0]
    wte = model.transformer.wte.weight.detach().cpu().numpy()

    rng = np.random.default_rng(42)

    L = 12
    print(f"Building L{L} corrected vocab...")
    v = build_vocab_pos1(model, L, device, eot)

    refs = [
        "the quick brown fox jumps over the lazy dog",
        "she went to the store to buy some food",
        "he ran down the long road to the old house",
        "they played in the park with the children",
    ]
    bias_accum = []
    for ref in refs:
        rt = tokenizer.encode(ref)
        with torch.no_grad():
            rh = model(torch.tensor([rt]).to(device), output_hidden_states=True).hidden_states[L].squeeze(0).cpu().numpy()
        trailing = rh[1:]
        n_r = trailing.shape[0]
        bias_accum.append((trailing.sum(axis=0) - v[rt[1:]].sum(axis=0)) / n_r)
    bias = np.mean(bias_accum, axis=0)

    print("Fitting bridge W (L12 -> wte)...")
    W, _, _, _ = np.linalg.lstsq(v, wte, rcond=None)

    # Pre-compute target embeddings
    print("\nPre-computing targets and random baselines...")
    test_data = []
    for sent in SENTENCES:
        tt = [eot] + tokenizer.encode(sent)
        target = get_pooled(model, tt, device)
        test_data.append((sent, tt, target))

    print(f"\n{'='*78}")
    print("BRIDGE + CD AT L12 — token match AND L2 distance to target")
    print(f"{'='*78}")
    print(f"  {'sentence':40s}  {'u':>3s}  {'bag_match':>9s}  {'L2_recov':>9s}  {'L2_rand':>9s}  L2_pct")
    print("  " + "-" * 100)

    total_bag_match = 0
    total_unique = 0
    total_l2_recov = 0
    total_l2_rand = 0
    n_better_than_random = 0

    for sent, tt, target in test_data:
        trailing = tt[1:]
        unique = set(trailing)
        u = len(unique)
        n_t = len(trailing)
        total_unique += u

        # Forward-pass to get the actual L12 hidden states for the bridge
        with torch.no_grad():
            out = model(torch.tensor([tt]).to(device), output_hidden_states=True)
        ctx_sum = out.hidden_states[L].squeeze(0)[1:].sum(dim=0).cpu().numpy()

        # Bridge + bias + CD
        bridged = (ctx_sum - n_t * bias) @ W
        rec = coord_descent(bridged, wte, n_t, max_iters=15)
        bag_match = len(unique & set(rec))
        total_bag_match += bag_match

        # Forward-pass the recovered tokens to get L2 distance
        recov_emb = get_pooled(model, [eot] + rec, device)
        l2_recov = float(np.linalg.norm(target - recov_emb))
        total_l2_recov += l2_recov

        # Random baseline L2: 30 random sequences of same length
        rand_l2s = []
        for _ in range(30):
            random_tokens = [eot] + rng.integers(0, vocab_size, size=n_t).tolist()
            r_emb = get_pooled(model, random_tokens, device)
            rand_l2s.append(float(np.linalg.norm(target - r_emb)))
        rand_l2_mean = np.mean(rand_l2s)
        rand_l2_min = np.min(rand_l2s)
        total_l2_rand += rand_l2_mean

        # Percentile of recov vs random (lower is better — fraction of random with lower L2)
        pct = (np.array(rand_l2s) < l2_recov).sum() / len(rand_l2s)
        if l2_recov < rand_l2_min:
            n_better_than_random += 1

        marker = "*" if l2_recov < rand_l2_min else " "
        print(f" {marker}{sent[:38]:40s}  {u:>3d}  {bag_match:>3d}/{u:<2d}    {l2_recov:>8.2f}  {rand_l2_mean:>8.2f}  {pct*100:.0f}%")

    print()
    print(f"  AGGREGATE:")
    print(f"    bag-match total: {total_bag_match}/{total_unique}  ({100*total_bag_match/total_unique:.1f}%)")
    print(f"    avg L2 recovered: {total_l2_recov/len(test_data):.2f}")
    print(f"    avg L2 random:    {total_l2_rand/len(test_data):.2f}")
    print(f"    sentences where bridge L2 < random_min: {n_better_than_random}/{len(test_data)}")

    del v, W
    gc.collect()


if __name__ == "__main__":
    main()
