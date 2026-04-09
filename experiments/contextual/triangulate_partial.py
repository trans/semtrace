#!/usr/bin/env python3
"""Test triangulation: HotFlip refinement starting from a partially-correct
anchor (the case the user's insight was about).

Two scenarios:
  A. Synthetic corruption: take the perfect L1 per-position anchor, corrupt
     half the positions to random tokens. Run HotFlip. Can it recover?
  B. First-N anchor: keep only the first 3 positions of the perfect L1 anchor,
     randomize the rest. Run HotFlip. Can it complete the sentence?
  C. Bridge L12 anchor: use the actual bridge result (real partial anchor).
     Run HotFlip. Can it refine?

The user's intuition: even a partial anchor should be enough to break the
gradient-descent ocean problem, because it constrains the search to the
neighborhood of a meaningful sequence.
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
    "the big red car drove fast",
]


def get_pooled_embedding(model, tokens, device):
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
    with torch.no_grad():
        out = model(torch.tensor([tokens]).to(device), output_hidden_states=True)
    hs = out.hidden_states[layer].squeeze(0).cpu().numpy()
    trailing = hs[1:]
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


def hotflip_refine(model, target_emb, initial_tokens, device, eot, wte_t,
                   top_k=20, max_iters=10, locked_positions=None):
    """HotFlip with optional locked positions (won't be swapped).

    locked_positions: set of indices that should be kept fixed during search.
    """
    if locked_positions is None:
        locked_positions = set()

    wte_normed = wte_t / (wte_t.norm(dim=-1, keepdim=True) + 1e-12)

    def topk_neighbors(token_id, k):
        v = wte_t[token_id]
        v_n = v / (v.norm() + 1e-12)
        sims = wte_normed @ v_n
        return torch.topk(sims, k).indices.cpu().numpy().tolist()

    def forward_dist(tokens):
        emb = get_pooled_embedding(model, [eot] + tokens, device)
        return float(np.sum((target_emb - emb) ** 2))

    current = list(initial_tokens)
    n = len(current)
    current_dist = forward_dist(current)

    for it in range(max_iters):
        best_swap = None
        best_dist_after = current_dist
        for pos in range(n):
            if pos in locked_positions:
                continue
            old = current[pos]
            for cand in topk_neighbors(old, top_k):
                if cand == old:
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
    device = "cpu"
    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True).to(device)
    model.eval()
    eot = tokenizer.eos_token_id

    wte_t = model.transformer.wte.weight.detach()
    wte_np = wte_t.cpu().numpy()

    print("Building L1 vocab for per-position anchor...")
    v_l1 = build_vocab_pos1(model, 1, device, eot)

    rng = np.random.default_rng(42)

    summary = []

    for sent in SENTENCES:
        print(f"\n{'='*70}")
        print(f"Sentence: {sent!r}")
        true_tokens = [eot] + tokenizer.encode(sent)
        true_trailing = true_tokens[1:]
        n_t = len(true_trailing)
        true_strs = [tokenizer.decode([t]) for t in true_trailing]
        target_emb = get_pooled_embedding(model, true_tokens, device)

        # Get the perfect L1 anchor
        l1_anchor = per_position_decompose(model, true_tokens, 1, v_l1, device)

        # ---- Scenario A: 50% random corruption ----
        n_corrupt = n_t // 2
        corrupt_positions = set(rng.choice(n_t, size=n_corrupt, replace=False).tolist())
        corrupted = list(l1_anchor)
        for p in corrupt_positions:
            corrupted[p] = int(rng.integers(0, wte_np.shape[0]))
        corrupted_match = sum(1 for a, b in zip(corrupted, true_trailing) if a == b)
        print(f"\n  A. 50% random corruption ({n_corrupt} positions corrupted)")
        print(f"     anchor: {[tokenizer.decode([t]) for t in corrupted]}")
        print(f"     anchor pos-match: {corrupted_match}/{n_t}")

        refined_a, dist_a = hotflip_refine(
            model, target_emb, corrupted, device, eot, wte_t,
            top_k=20, max_iters=10
        )
        refined_a_match = sum(1 for a, b in zip(refined_a, true_trailing) if a == b)
        print(f"     after HotFlip: {[tokenizer.decode([t]) for t in refined_a]}")
        print(f"     pos-match: {refined_a_match}/{n_t}, dist: {dist_a:.4f}")

        # ---- Scenario B: First-3 anchor only ----
        if n_t > 3:
            first3 = list(l1_anchor)
            for p in range(3, n_t):
                first3[p] = int(rng.integers(0, wte_np.shape[0]))
            first3_match = sum(1 for a, b in zip(first3, true_trailing) if a == b)
            print(f"\n  B. Keep first 3 positions correct, randomize rest")
            print(f"     anchor: {[tokenizer.decode([t]) for t in first3]}")
            print(f"     anchor pos-match: {first3_match}/{n_t}")

            refined_b, dist_b = hotflip_refine(
                model, target_emb, first3, device, eot, wte_t,
                top_k=20, max_iters=10, locked_positions={0, 1, 2}
            )
            refined_b_match = sum(1 for a, b in zip(refined_b, true_trailing) if a == b)
            print(f"     after HotFlip (first 3 locked): {[tokenizer.decode([t]) for t in refined_b]}")
            print(f"     pos-match: {refined_b_match}/{n_t}, dist: {dist_b:.4f}")
        else:
            refined_b_match = None
            first3_match = None

        summary.append((sent, n_t, corrupted_match, refined_a_match, first3_match, refined_b_match))

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  {'sentence':40s}  {'n':>3s}  {'A_in':>5s}  {'A_out':>6s}  {'B_in':>5s}  {'B_out':>6s}")
    for sent, n, a_in, a_out, b_in, b_out in summary:
        b_in_s = f"{b_in}/{n}" if b_in is not None else "n/a"
        b_out_s = f"{b_out}/{n}" if b_out is not None else "n/a"
        print(f"  {sent[:38]:40s}  {n:>3d}  {a_in:>3d}/{n:<2d}  {a_out:>3d}/{n:<2d}  {b_in_s:>5s}  {b_out_s:>6s}")

    del v_l1
    gc.collect()


if __name__ == "__main__":
    main()
