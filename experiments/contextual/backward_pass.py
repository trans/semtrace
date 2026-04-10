#!/usr/bin/env python3
"""Per-layer backward pass: walk a target L12 embedding backward through
GPT-2's blocks using a candidate's intermediate states as guides.

The idea:
  Each block does: output = input + attn_contribution + mlp_contribution
  So: input = output - attn_contribution - mlp_contribution

  We don't know the true contributions (they depend on the true input).
  But we have a GOOD CANDIDATE from the pipeline (L2≈3.5). We forward-pass
  it, capture each block's attn and mlp contributions, and use those to
  estimate the backward subtraction.

  Walk from L12 backward to L0. At L0, subtract wpe and nearest-neighbor
  lookup on wte → tokens.

  If the recovered tokens differ from the candidate, forward-pass them
  (new candidate) and repeat. The iteration should converge because each
  round uses a better candidate's contributions.

Tested on the Mary had a little lamb sentence (23 tokens).
"""
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


TEXT = "Mary had a little lamb its fleece was white as snow and everywhere that Mary went the lamb was sure to go"


def get_pooled(model, tokens, device):
    with torch.no_grad():
        out = model(torch.tensor([tokens]).to(device), output_hidden_states=True)
    l12 = out.hidden_states[12].squeeze(0)
    return l12[1:].mean(dim=0).cpu().numpy()


def forward_with_contributions(model, tokens, device):
    """Forward pass capturing per-block attn and mlp contributions.
    Returns hidden_states at each layer AND the contributions."""
    hooks = {}
    contributions = {}

    def make_hook(block_idx, name):
        def hook(_module, inputs, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            contributions[(block_idx, name)] = out.detach().squeeze(0).cpu().numpy()
        return hook

    handles = []
    for bi in range(model.config.n_layer):
        block = model.transformer.h[bi]
        handles.append(block.attn.register_forward_hook(make_hook(bi, "attn")))
        handles.append(block.mlp.register_forward_hook(make_hook(bi, "mlp")))

    with torch.no_grad():
        out = model(torch.tensor([tokens]).to(device), output_hidden_states=True)

    for h in handles:
        h.remove()

    hidden_states = [h.squeeze(0).cpu().numpy() for h in out.hidden_states]
    return hidden_states, contributions


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


def per_position_decompose_from_hidden(hidden_states_l0, vocab, wpe):
    """Given L0 hidden states (one per position), subtract wpe and find
    nearest wte token."""
    n = hidden_states_l0.shape[0]
    vocab_norms = np.linalg.norm(vocab, axis=1)
    vocab_norms[vocab_norms < 1e-10] = 1.0
    recovered = []
    for i in range(n):
        # L0 = wte[token] + wpe[position]
        # So wte[token] ≈ L0 - wpe[position]
        # But position in the full sequence is i+1 (because position 0 is EOT)
        target = hidden_states_l0[i] - wpe[i + 1]
        t_norm = np.linalg.norm(target)
        if t_norm < 1e-10:
            recovered.append(0)
            continue
        sims = vocab @ target / (vocab_norms * t_norm)
        recovered.append(int(np.argmax(sims)))
    return recovered


def backward_pass(target_l12_per_pos, contributions, n_layers=12):
    """Walk backward from L12 per-position hidden states to L0 estimate.

    At each block l (from 11 down to 0):
      L_l_estimate = L_{l+1}_estimate - attn_contribution[l] - mlp_contribution[l]

    target_l12_per_pos: the TARGET hidden states at L12 (per position)
    contributions: dict of (block_idx, 'attn'/'mlp') -> numpy array (n_pos, 768)
    """
    current = target_l12_per_pos.copy()

    for block_idx in range(n_layers - 1, -1, -1):
        attn = contributions[(block_idx, "attn")]
        mlp = contributions[(block_idx, "mlp")]
        current = current - attn - mlp

    return current


def main():
    device = "cpu"
    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True).to(device)
    model.eval()
    eot = tokenizer.eos_token_id

    wte = model.transformer.wte.weight.detach().cpu().numpy()
    wpe = model.transformer.wpe.weight.detach().cpu().numpy()

    true_tokens = [eot] + tokenizer.encode(TEXT)
    bag = true_tokens[1:]
    n_t = len(bag)
    true_strs = [tokenizer.decode([t]) for t in bag]

    print(f"\nText: {TEXT!r}")
    print(f"Tokens ({n_t}): {' '.join(true_strs)}")

    target_emb = get_pooled(model, true_tokens, device)

    # Get the TRUE per-position L12 hidden states (this is what we'd have
    # in the white-box setting with per-position access)
    true_hs, true_contribs = forward_with_contributions(model, true_tokens, device)
    target_l12_per_pos = true_hs[12][1:]  # trailing positions at L12

    # ================================================================
    # Sanity check: backward pass using the TRUE contributions
    # ================================================================
    print(f"\n{'='*70}")
    print("Sanity check: backward pass with TRUE contributions")
    print(f"{'='*70}")

    # Strip contributions to trailing positions only (indices 1..N)
    true_contribs_trailing = {}
    for (bi, name), arr in true_contribs.items():
        true_contribs_trailing[(bi, name)] = arr[1:]  # skip position 0 (EOT)

    l0_estimate = backward_pass(target_l12_per_pos, true_contribs_trailing)
    recovered = per_position_decompose_from_hidden(l0_estimate, wte, wpe)
    pm = sum(1 for a, b in zip(recovered, bag) if a == b)
    rec_strs = [tokenizer.decode([t]) for t in recovered]
    print(f"  recovered: {' '.join(rec_strs)}")
    print(f"  pos-match: {pm}/{n_t}")
    if pm == n_t:
        print(f"  *** PERFECT (with true contributions) ***")

    # Verify the L0 estimate is close to true L0
    true_l0 = true_hs[0][1:]
    l0_error = np.linalg.norm(l0_estimate - true_l0) / np.linalg.norm(true_l0)
    print(f"  L0 estimate relative error: {l0_error:.6f}")

    # ================================================================
    # Now: backward pass using a CANDIDATE's contributions
    # ================================================================
    # Use the pipeline result as the candidate. For this test, simulate
    # a near-correct candidate by shuffling a few positions.
    import random
    random.seed(42)
    candidate = list(bag)
    # Swap 5 random pairs to simulate an imperfect candidate
    for _ in range(5):
        i, j = random.sample(range(n_t), 2)
        candidate[i], candidate[j] = candidate[j], candidate[i]

    cand_pm = sum(1 for a, b in zip(candidate, bag) if a == b)
    cand_strs = [tokenizer.decode([t]) for t in candidate]
    cand_l2 = float(np.linalg.norm(target_emb - get_pooled(model, [eot] + candidate, device)))
    print(f"\n{'='*70}")
    print(f"Candidate (5 swaps from true): pos-match={cand_pm}/{n_t}, L2={cand_l2:.4f}")
    print(f"  {' '.join(cand_strs)}")

    # Iterative backward refinement
    print(f"\n{'='*70}")
    print("Iterative backward refinement")
    print(f"{'='*70}")

    current_tokens = list(candidate)

    for iteration in range(5):
        # Forward-pass current candidate, capture contributions
        cand_hs, cand_contribs = forward_with_contributions(
            model, [eot] + current_tokens, device)

        # Strip to trailing positions
        cand_contribs_trailing = {}
        for (bi, name), arr in cand_contribs.items():
            cand_contribs_trailing[(bi, name)] = arr[1:]

        # Backward pass: use candidate's contributions on the TARGET L12
        l0_estimate = backward_pass(target_l12_per_pos, cand_contribs_trailing)

        # Decompose at L0
        recovered = per_position_decompose_from_hidden(l0_estimate, wte, wpe)

        pm = sum(1 for a, b in zip(recovered, bag) if a == b)
        rec_strs = [tokenizer.decode([t]) for t in recovered]
        rec_l2 = float(np.linalg.norm(target_emb - get_pooled(model, [eot] + recovered, device)))

        print(f"\n  Iteration {iteration + 1}:")
        print(f"    recovered: {' '.join(rec_strs)}")
        print(f"    pos-match: {pm}/{n_t}, L2={rec_l2:.4f}")

        if pm == n_t:
            print(f"    *** PERFECT RECOVERY ***")
            break

        # Check if we improved
        old_pm = sum(1 for a, b in zip(current_tokens, bag) if a == b)
        if pm > old_pm:
            print(f"    improved from {old_pm} → {pm} pos-match")
            current_tokens = recovered
        elif pm == old_pm and rec_l2 < cand_l2:
            print(f"    same pos-match but better L2: {cand_l2:.4f} → {rec_l2:.4f}")
            current_tokens = recovered
            cand_l2 = rec_l2
        else:
            print(f"    no improvement (was {old_pm} pos-match)")
            # Try anyway to see if next iteration helps
            current_tokens = recovered

    # Final result
    final_pm = sum(1 for a, b in zip(current_tokens, bag) if a == b)
    final_l2 = float(np.linalg.norm(target_emb - get_pooled(model, [eot] + current_tokens, device)))
    print(f"\n{'='*70}")
    print(f"FINAL: pos-match={final_pm}/{n_t}, L2={final_l2:.4f}")
    print(f"  {' '.join([tokenizer.decode([t]) for t in current_tokens])}")
    print(f"TRUE:")
    print(f"  {' '.join(true_strs)}")


if __name__ == "__main__":
    main()
