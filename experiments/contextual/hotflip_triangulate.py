#!/usr/bin/env python3
"""HotFlip-style discrete refinement starting from a decomposition anchor.

The triangulation idea: decomposition gives us a noisy starting point that's
already on the "manifold of valid token sequences" (every candidate is a real
token). Gradient inversion would drift off this manifold into spurious minima.
Discrete swap-search stays on the manifold by construction and uses the model's
forward pass as a verifier.

Pipeline:
  1. Get an initial candidate sequence from decomposition (use the contextual
     decomp at L1 since it works well, or the bridge if we want to test the
     hard case).
  2. For each position k in the candidate, try replacing the token with each
     of its top-K nearest-neighbors in static embedding space.
  3. For each candidate swap, forward-pass the resulting sequence and compute
     the L2 distance to the target embedding.
  4. Apply the best swap. Repeat until no swap improves the distance.

This is essentially HotFlip applied to embedding inversion. It avoids the
continuous-discrete gap by operating only on real tokens, and it uses the
decomposition anchor to bootstrap the search to a meaningful starting point.
"""
import argparse
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
    """Get the L12 mean-pool of the trailing positions."""
    with torch.no_grad():
        out = model(torch.tensor([tokens]).to(device), output_hidden_states=True)
    l12 = out.hidden_states[12].squeeze(0)
    return l12[1:].mean(dim=0).cpu().numpy()


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


def per_position_decompose(model, tokens, layer, vocab, device):
    """Per-position L1 decomposition: returns the recovered token at each position."""
    with torch.no_grad():
        out = model(torch.tensor([tokens]).to(device), output_hidden_states=True)
    hs = out.hidden_states[layer].squeeze(0).cpu().numpy()
    trailing = hs[1:]  # skip sink
    vocab_norms = np.linalg.norm(vocab, axis=1)
    vocab_norms[vocab_norms < 1e-10] = 1.0
    recovered = []
    for i in range(trailing.shape[0]):
        target = trailing[i]
        t_norm = np.linalg.norm(target)
        if t_norm < 1e-10:
            recovered.append(0)
            continue
        sims = vocab @ target / (vocab_norms * t_norm)
        recovered.append(int(np.argmax(sims)))
    return recovered


def hotflip_refine(
    model, target_emb, initial_tokens, device, eot, wte, top_k=20, max_iters=10
):
    """Discrete swap search to minimize embedding distance to target.

    At each iteration, for each position, try replacing the current token
    with each of its top-K nearest neighbors in static embedding space.
    Pick the single best swap across all positions and apply it. Iterate.
    """
    target_t = torch.tensor(target_emb, device=device)
    current = list(initial_tokens)
    n = len(current)

    # Pre-compute static neighbor lists for efficiency
    wte_norms = torch.norm(wte, dim=-1, keepdim=True)
    wte_normed = wte / (wte_norms + 1e-12)

    def get_topk_neighbors(token_id, k):
        """For a token, return its top-K static-cosine neighbors."""
        v = wte[token_id]
        v_n = v / (v.norm() + 1e-12)
        sims = wte_normed @ v_n
        return torch.topk(sims, k).indices.cpu().numpy().tolist()

    def forward_dist(tokens):
        full = [eot] + tokens
        emb = get_pooled_embedding(model, full, device)
        return float(np.sum((target_emb - emb) ** 2))

    current_dist = forward_dist(current)

    for iteration in range(max_iters):
        best_swap = None  # (position, new_token, new_dist)
        best_dist_after = current_dist

        for pos in range(n):
            old_token = current[pos]
            # Get top-K neighbors of the current token at this position
            neighbors = get_topk_neighbors(old_token, top_k)
            for cand in neighbors:
                if cand == old_token:
                    continue
                new_seq = list(current)
                new_seq[pos] = cand
                new_dist = forward_dist(new_seq)
                if new_dist < best_dist_after:
                    best_dist_after = new_dist
                    best_swap = (pos, cand)

        if best_swap is None:
            break

        pos, new_token = best_swap
        current[pos] = new_token
        current_dist = best_dist_after

    return current, current_dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-iters", type=int, default=8)
    args = parser.parse_args()

    device = "cpu"
    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True).to(device)
    model.eval()
    eot = tokenizer.eos_token_id

    wte_t = model.transformer.wte.weight.detach()
    wte_np = wte_t.cpu().numpy()

    # Build the L1 vocab once for per-position decomposition
    print("Building L1 corrected vocab (for initial anchor)...")
    v_l1 = build_vocab_pos1(model, 1, device, eot)
    print(f"  vocab shape: {v_l1.shape}")

    for sent in SENTENCES:
        print(f"\n{'='*70}")
        print(f"Sentence: {sent!r}")
        true_tokens = [eot] + tokenizer.encode(sent)
        true_trailing = true_tokens[1:]
        n_t = len(true_trailing)
        true_strs = [tokenizer.decode([t]) for t in true_trailing]
        print(f"  True: {true_strs}")

        target_emb = get_pooled_embedding(model, true_tokens, device)

        # ---- Get initial anchor from per-position L1 decomposition ----
        anchor = per_position_decompose(model, true_tokens, 1, v_l1, device)
        anchor_strs = [tokenizer.decode([t]) for t in anchor]
        anchor_pos_match = sum(1 for a, b in zip(anchor, true_trailing) if a == b)
        print(f"  Anchor (L1 per-pos): {anchor_strs}")
        print(f"    pos-match: {anchor_pos_match}/{n_t}")

        anchor_full = [eot] + anchor
        anchor_emb = get_pooled_embedding(model, anchor_full, device)
        anchor_dist = float(np.sum((target_emb - anchor_emb) ** 2))
        print(f"    initial dist to target: {anchor_dist:.4f}")

        # ---- HotFlip refinement ----
        refined, refined_dist = hotflip_refine(
            model, target_emb, anchor, device, eot, wte_t,
            top_k=args.top_k, max_iters=args.max_iters,
        )
        refined_strs = [tokenizer.decode([t]) for t in refined]
        refined_pos_match = sum(1 for a, b in zip(refined, true_trailing) if a == b)
        print(f"  After HotFlip ({args.max_iters} iters, top-K={args.top_k}):")
        print(f"    {refined_strs}")
        print(f"    pos-match: {refined_pos_match}/{n_t}, dist: {refined_dist:.4f}")
        if refined_pos_match == n_t:
            print(f"    *** PERFECT RECOVERY ***")

        # ---- Also try starting from a wrong anchor (random tokens) ----
        # to see if HotFlip can recover from a bad start
        rng = np.random.default_rng(42)
        wrong_anchor = rng.integers(0, wte_np.shape[0], size=n_t).tolist()
        wrong_strs = [tokenizer.decode([int(t)]) for t in wrong_anchor]
        print(f"  Wrong anchor (random): {wrong_strs}")

        wrong_full = [eot] + [int(t) for t in wrong_anchor]
        wrong_init_dist = float(np.sum((target_emb - get_pooled_embedding(model, wrong_full, device)) ** 2))
        print(f"    initial dist: {wrong_init_dist:.4f}")

        refined_w, refined_w_dist = hotflip_refine(
            model, target_emb, [int(t) for t in wrong_anchor], device, eot, wte_t,
            top_k=args.top_k, max_iters=args.max_iters,
        )
        refined_w_strs = [tokenizer.decode([t]) for t in refined_w]
        refined_w_pos_match = sum(1 for a, b in zip(refined_w, true_trailing) if a == b)
        print(f"  After HotFlip from wrong: {refined_w_strs}")
        print(f"    pos-match: {refined_w_pos_match}/{n_t}, dist: {refined_w_dist:.4f}")

    del v_l1
    gc.collect()


if __name__ == "__main__":
    main()
