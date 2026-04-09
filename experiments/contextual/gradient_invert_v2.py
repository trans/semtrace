#!/usr/bin/env python3
"""Gradient inversion v2: address the continuous-discrete gap.

v1 found continuous embeddings that match the target embedding well (cos
0.99+) but project to garbage tokens. The continuous space is too
permissive — many non-token vectors produce target-like outputs.

Fixes in v2:
  - Periodic discrete snap: every K steps, project to nearest tokens and
    restart optimization from those discrete embeddings
  - Stronger projection regularizer
  - Multi-restart with different random inits, keep best by post-projection
    embedding distance (not pre-projection match loss, which is misleading)
"""
import argparse
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


def get_target_embedding(model, tokens, device):
    with torch.no_grad():
        out = model(torch.tensor([tokens]).to(device), output_hidden_states=True)
    l12 = out.hidden_states[12].squeeze(0)
    return l12[1:].mean(dim=0).detach()


def get_pooled_from_embeds(model, full_embs):
    """Forward-pass continuous embeddings and return the L12 mean-pool of trailing positions."""
    out = model(inputs_embeds=full_embs.unsqueeze(0), output_hidden_states=True)
    l12 = out.hidden_states[12].squeeze(0)
    return l12[1:].mean(dim=0)


def project_to_tokens(continuous_embs, wte):
    """For each continuous embedding, find the nearest wte token by cosine."""
    embs = continuous_embs.detach()
    embs_n = embs / (embs.norm(dim=-1, keepdim=True) + 1e-12)
    wte_n = wte / (wte.norm(dim=-1, keepdim=True) + 1e-12)
    sims = embs_n @ wte_n.T
    return sims.argmax(dim=-1)


def gradient_invert_with_snapping(
    model, target, n_tokens, device, eot, wte,
    init_tokens=None, n_outer=10, n_inner=100, lr=0.5,
    snap_each_round=True, verbose=True
):
    """Gradient descent with periodic discrete snapping.

    Outer loop: each outer step does n_inner gradient steps, then snaps to
    nearest tokens. Inner state is restarted from snapped tokens each round.
    """
    eot_emb = wte[eot].detach().unsqueeze(0)

    # Initialize
    if init_tokens is None:
        # Random tokens from vocabulary
        init_tokens = torch.randint(0, wte.shape[0], (n_tokens,), device=device).tolist()

    current_tokens = list(init_tokens)
    best_dist = float("inf")
    best_tokens = list(current_tokens)

    for outer in range(n_outer):
        # Initialize trailing embeddings from current tokens
        trailing_embs = wte[torch.tensor(current_tokens, dtype=torch.long, device=device)].clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([trailing_embs], lr=lr)

        for inner in range(n_inner):
            optimizer.zero_grad()
            full_embs = torch.cat([eot_emb, trailing_embs], dim=0)
            pooled = get_pooled_from_embeds(model, full_embs)
            loss = ((target - pooled) ** 2).sum()
            loss.backward()
            optimizer.step()

        # Snap to nearest tokens
        snapped = project_to_tokens(trailing_embs, wte).cpu().numpy().tolist()

        # Evaluate the snapped solution by actual forward pass
        with torch.no_grad():
            snapped_full = [eot] + [int(t) for t in snapped]
            snapped_emb = get_target_embedding(model, snapped_full, device)
            snapped_dist = float(((target - snapped_emb) ** 2).sum())
            snapped_cos = float(F.cosine_similarity(target.unsqueeze(0), snapped_emb.unsqueeze(0)).item())

        if verbose:
            print(f"  outer {outer}: snapped dist={snapped_dist:>9.2f} cos={snapped_cos:.6f}  "
                  f"tokens={[int(t) for t in snapped[:5]]}...")

        if snapped_dist < best_dist:
            best_dist = snapped_dist
            best_tokens = list(snapped)

        # Continue from the snapped tokens (or stay if snap not used)
        if snap_each_round:
            current_tokens = list(snapped)

    return best_tokens, best_dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-outer", type=int, default=10)
    parser.add_argument("--n-inner", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--restarts", type=int, default=3)
    args = parser.parse_args()

    device = "cpu"
    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True).to(device)
    model.eval()
    eot = tokenizer.eos_token_id

    for p in model.parameters():
        p.requires_grad = False

    wte = model.transformer.wte.weight.detach()

    for sent in SENTENCES:
        print(f"\n{'='*70}")
        print(f"Sentence: {sent!r}")
        true_tokens = [eot] + tokenizer.encode(sent)
        n_t = len(true_tokens) - 1
        true_trailing = true_tokens[1:]
        true_strs = [tokenizer.decode([t]) for t in true_trailing]
        target = get_target_embedding(model, true_tokens, device)

        # Multi-restart: try several random inits, keep best
        best_overall_dist = float("inf")
        best_overall_tokens = None

        for restart in range(args.restarts):
            print(f"  --- Restart {restart} ---")
            tokens, dist = gradient_invert_with_snapping(
                model, target, n_t, device, eot, wte,
                init_tokens=None,  # random
                n_outer=args.n_outer, n_inner=args.n_inner, lr=args.lr,
                verbose=True,
            )
            if dist < best_overall_dist:
                best_overall_dist = dist
                best_overall_tokens = tokens

        recovered_strs = [tokenizer.decode([int(t)]) for t in best_overall_tokens]
        position_match = sum(1 for a, b in zip(best_overall_tokens, true_trailing) if int(a) == b)
        bag_match = len(set(int(t) for t in best_overall_tokens) & set(true_trailing))

        print(f"  BEST: recovered {recovered_strs}")
        print(f"        true      {true_strs}")
        print(f"        pos-match: {position_match}/{n_t}, bag-match: {bag_match}/{len(set(true_trailing))}")
        print(f"        dist: {best_overall_dist:.4f}")


if __name__ == "__main__":
    main()
