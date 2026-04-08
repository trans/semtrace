#!/usr/bin/env python3
"""Attention trace: explicitly walk one sentence through the first few
transformer blocks of GPT-2 Small and dump what each component adds.

For each block we capture:
  - the residual stream entering the block (per position)
  - the attention output (what attention adds to the residual, per position)
  - the MLP output (what the MLP adds to the residual, per position)
  - the attention weights (who attends to whom)

Then for each block we report:
  - ||residual_in[i]||, ||attn_add[i]||, ||mlp_add[i]||, ||residual_out[i]||
  - cosine(residual_in[i], residual_out[i])     (how much position i is preserved)
  - cosine(attn_add[i], residual_in[i])         (does attention add something orthogonal?)
  - the attention pattern (which positions does i attend to most)

The key question: at L1, does attention act like "add a near-constant bias to
each token", or does it already mix tokens substantially? And how does that
change by L2, L3?
"""
import argparse
import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="the cat sat on the mat")
    parser.add_argument("--blocks", default="0,1,2,3", help="Which transformer blocks to trace")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    blocks_to_trace = [int(x) for x in args.blocks.split(",")]

    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained(
        "gpt2", output_hidden_states=True, output_attentions=True
    ).to(args.device)
    model.eval()

    tokens = tokenizer.encode(args.text)
    token_strs = [tokenizer.decode([t]) for t in tokens]
    n = len(tokens)
    print(f"Text: {args.text!r}")
    print(f"Tokens ({n}): {token_strs}")

    # Hooks: capture the input/output of each block component
    captured = {}  # (block_idx, name) -> tensor (numpy)

    def make_hook(block_idx, name):
        def hook(_module, inputs, output):
            # output may be a tuple for attention
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            captured[(block_idx, name)] = out.detach().squeeze(0).cpu().numpy()
        return hook

    handles = []
    for bi in blocks_to_trace:
        block = model.h[bi]
        handles.append(block.attn.register_forward_hook(make_hook(bi, "attn_out")))
        handles.append(block.mlp.register_forward_hook(make_hook(bi, "mlp_out")))
        handles.append(block.ln_1.register_forward_hook(make_hook(bi, "ln1_out")))
        handles.append(block.ln_2.register_forward_hook(make_hook(bi, "ln2_out")))

    # Forward pass
    with torch.no_grad():
        out = model(torch.tensor([tokens]).to(args.device))

    for h in handles:
        h.remove()

    # hidden_states[i] is the residual stream BEFORE block i (and hidden_states[12] is post ln_f)
    # So hidden_states[bi] = input to block bi, hidden_states[bi+1] = output of block bi
    hs = [h.squeeze(0).cpu().numpy() for h in out.hidden_states]
    attns = [a.squeeze(0).cpu().numpy() for a in out.attentions]  # [n_heads, n_tok, n_tok]

    print(f"\nL0 (token+pos embed) norms per position:")
    for i in range(n):
        print(f"  pos {i} ({token_strs[i]!r:>10s}): ||h0||={np.linalg.norm(hs[0][i]):.2f}")

    for bi in blocks_to_trace:
        print(f"\n{'='*70}")
        print(f"BLOCK {bi}")
        print(f"{'='*70}")

        residual_in = hs[bi]                    # input to block bi
        residual_out = hs[bi + 1]               # output of block bi
        attn_add = captured[(bi, "attn_out")]   # what attention added
        mlp_add = captured[(bi, "mlp_out")]     # what MLP added

        # Sanity check: residual_in + attn_add + mlp_add should ≈ residual_out
        # (with the LN-normalized branches being what attn/mlp see, but the
        # residual additions themselves are unnormalized)
        reconstructed = residual_in + attn_add + mlp_add
        recon_err = np.linalg.norm(residual_out - reconstructed)
        print(f"Reconstruction check: ||(in + attn + mlp) - out|| = {recon_err:.6f}")

        # Per-position breakdown
        print(f"\n{'pos':>4s}  {'token':>10s}  {'||in||':>9s}  {'||attn||':>9s}  {'||mlp||':>9s}  {'||out||':>9s}  {'cos(in,out)':>11s}  {'cos(in,attn)':>12s}")
        for i in range(n):
            ri = residual_in[i]
            ro = residual_out[i]
            ai = attn_add[i]
            mi = mlp_add[i]
            print(f"  {i:>2d}  {token_strs[i]!r:>10s}  {np.linalg.norm(ri):>9.2f}  {np.linalg.norm(ai):>9.2f}  {np.linalg.norm(mi):>9.2f}  {np.linalg.norm(ro):>9.2f}  {cos(ri, ro):>11.4f}  {cos(ri, ai):>12.4f}")

        # How constant is the attention contribution across positions?
        # If attention is "add a bias", then attn_add[i] should be similar for all i
        print(f"\n  Attention contribution similarity across positions:")
        cos_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                cos_pairs.append(cos(attn_add[i], attn_add[j]))
        print(f"    cos(attn_add[i], attn_add[j]) avg over {len(cos_pairs)} pairs: {np.mean(cos_pairs):.4f}  (min {min(cos_pairs):.4f}, max {max(cos_pairs):.4f})")

        # MLP contribution similarity
        cos_pairs_mlp = []
        for i in range(n):
            for j in range(i + 1, n):
                cos_pairs_mlp.append(cos(mlp_add[i], mlp_add[j]))
        print(f"    cos(mlp_add[i], mlp_add[j])  avg over {len(cos_pairs_mlp)} pairs: {np.mean(cos_pairs_mlp):.4f}  (min {min(cos_pairs_mlp):.4f}, max {max(cos_pairs_mlp):.4f})")

        # Attention pattern (averaged over heads)
        attn_pattern = attns[bi].mean(axis=0)  # [n_tok, n_tok]
        print(f"\n  Attention pattern (head-averaged), row=query position:")
        header = "       " + "  ".join(f"{j:>5d}" for j in range(n))
        print(header)
        for i in range(n):
            row = "  ".join(f"{attn_pattern[i, j]:>5.2f}" for j in range(n))
            print(f"  q{i}:  {row}")

        # How "self-attentive" is each position? Diagonal entries.
        print(f"\n  Self-attention weight per position (attention[i,i]):")
        for i in range(n):
            print(f"    pos {i} ({token_strs[i]!r}): {attn_pattern[i,i]:.4f}")


if __name__ == "__main__":
    main()
