#!/usr/bin/env python3
"""Gradient-descent embedding inversion: run the model "backward" via autograd.

Instead of trying to invert the model algebraically, treat the input as
continuous embeddings (not discrete tokens), forward-pass through GPT-2,
compute the L2 distance to a target embedding, and gradient-descend on the
continuous embeddings until they produce the target. Then project to the
nearest discrete tokens.

This is the most direct white-box inversion: the gradient flows through
every layer of the model in reverse, telling us how to nudge the input to
make the output closer to the target.

The optimization runs in continuous embedding space, which is technically
off-manifold (the model was trained on discrete wte lookups), so we add a
regularizer pulling the continuous embeddings toward their nearest discrete
token. This keeps the optimization well-behaved.
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
    """Get the L12 mean-pool of the trailing positions for a known token sequence."""
    with torch.no_grad():
        out = model(torch.tensor([tokens]).to(device), output_hidden_states=True)
    l12 = out.hidden_states[12].squeeze(0)
    return l12[1:].mean(dim=0).detach()


def project_to_tokens(continuous_embs, wte):
    """For each continuous embedding, find the nearest wte token by cosine."""
    embs = continuous_embs.detach()
    embs_n = embs / (embs.norm(dim=-1, keepdim=True) + 1e-12)
    wte_n = wte / (wte.norm(dim=-1, keepdim=True) + 1e-12)
    sims = embs_n @ wte_n.T
    return sims.argmax(dim=-1).cpu().numpy()


def nearest_token_distance(continuous_embs, wte):
    """For each continuous embedding, the L2 distance to its nearest wte token."""
    sims = continuous_embs @ wte.T
    closest = sims.argmax(dim=-1)
    nearest = wte[closest]
    return ((continuous_embs - nearest) ** 2).sum()


def gradient_invert(
    model, target, n_tokens, device, eot, wte,
    init_tokens=None, n_steps=500, lr=0.5,
    proj_weight=0.01, verbose=True
):
    """Run gradient descent to find continuous embeddings whose forward pass
    produces `target` (a pooled L12 embedding).

    Args:
      target: torch.Tensor (768,) - the target pooled L12 embedding
      n_tokens: number of trailing positions to recover
      init_tokens: optional list of token IDs to initialize from
      n_steps: number of gradient descent steps
      lr: learning rate
      proj_weight: regularizer weight pulling embeddings toward nearest tokens
    """
    # Initialize trailing embeddings
    if init_tokens is not None:
        init = wte[torch.tensor(init_tokens, dtype=torch.long).to(device)].clone().detach()
    else:
        init = torch.randn(n_tokens, 768, device=device) * wte.std().item() + wte.mean().item()

    trailing_embs = init.clone().detach().requires_grad_(True)
    eot_emb = wte[eot].detach().unsqueeze(0)  # (1, 768)

    optimizer = torch.optim.Adam([trailing_embs], lr=lr)

    best_loss = float("inf")
    best_embs = trailing_embs.detach().clone()

    for step in range(n_steps):
        optimizer.zero_grad()

        full_embs = torch.cat([eot_emb, trailing_embs], dim=0).unsqueeze(0)
        out = model(inputs_embeds=full_embs, output_hidden_states=True)
        l12 = out.hidden_states[12].squeeze(0)
        pooled = l12[1:].mean(dim=0)

        # Main loss: distance to target embedding
        match_loss = ((target - pooled) ** 2).sum()
        # Regularizer: pull continuous embeddings toward nearest discrete token
        proj_loss = nearest_token_distance(trailing_embs, wte)

        loss = match_loss + proj_weight * proj_loss
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_embs = trailing_embs.detach().clone()

        if verbose and step % 50 == 0:
            print(f"    step {step:>4d}: match_loss={match_loss.item():>9.3f}  "
                  f"proj_loss={proj_loss.item():>9.3f}  total={loss.item():>9.3f}")

    return best_embs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--proj-weight", type=float, default=0.01)
    parser.add_argument("--init", default="random", choices=["random", "true", "wrong"])
    args = parser.parse_args()

    device = "cpu"
    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True).to(device)
    model.eval()
    eot = tokenizer.eos_token_id

    # Freeze model parameters (we only optimize the input embeddings)
    for p in model.parameters():
        p.requires_grad = False

    wte = model.transformer.wte.weight.detach()  # (V, 768)

    for sent in SENTENCES:
        print(f"\n{'='*70}")
        print(f"Sentence: {sent!r}")
        true_tokens = [eot] + tokenizer.encode(sent)
        n_t = len(true_tokens) - 1
        true_trailing = true_tokens[1:]
        true_strs = [tokenizer.decode([t]) for t in true_trailing]
        print(f"True trailing tokens: {true_strs}")

        target = get_target_embedding(model, true_tokens, device)

        # Choose initialization
        if args.init == "true":
            init_tokens = true_trailing  # cheating: start from the right answer
        elif args.init == "wrong":
            # Initialize with wrong tokens (e.g., shifted by 100 in vocab)
            init_tokens = [(t + 100) % wte.shape[0] for t in true_trailing]
        else:
            init_tokens = None  # random

        result = gradient_invert(
            model, target, n_t, device, eot, wte,
            init_tokens=init_tokens,
            n_steps=args.steps,
            lr=args.lr,
            proj_weight=args.proj_weight,
            verbose=True,
        )

        # Project to nearest tokens
        recovered = project_to_tokens(result, wte)
        recovered_strs = [tokenizer.decode([int(t)]) for t in recovered]

        # Score: position-match
        position_match = sum(1 for a, b in zip(recovered, true_trailing) if int(a) == b)
        bag_match = len(set(int(t) for t in recovered) & set(true_trailing))

        # Final embedding distance after projection
        with torch.no_grad():
            final_full = [eot] + [int(t) for t in recovered]
            final_emb = get_target_embedding(model, final_full, device)
            final_dist = float(((target - final_emb) ** 2).sum())
            final_cos = float(F.cosine_similarity(target.unsqueeze(0), final_emb.unsqueeze(0)).item())

        print(f"  Recovered: {recovered_strs}")
        print(f"  True:      {true_strs}")
        print(f"  Position-match: {position_match}/{n_t}")
        print(f"  Bag-match:      {bag_match}/{len(set(true_trailing))}")
        print(f"  Final embedding dist: {final_dist:.4f}, cos: {final_cos:.6f}")


if __name__ == "__main__":
    main()
