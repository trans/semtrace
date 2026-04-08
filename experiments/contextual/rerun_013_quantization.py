#!/usr/bin/env python3
"""Re-run experiment 013 (contextual half) with sink-skip + N-debias.

Tests whether quantization destroys the contextual signal once the sink artifact
is removed and the corrected method is used. Uses the same Mary-had-a-little-lamb
text as the original, at L1 (best layer for sink-skip method).

For each precision (f32, f16, int8, int4):
  1. Quantize the target hidden states
  2. Quantize the corrected vocabulary (built with [<|endoftext|>, token] pairs)
  3. Quantize the per-token bias
  4. Run sink-skip + N-debias decomposition
  5. Report recovery
"""
import gc
import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer


TEXT = "Mary had a little lamb its fleece was white as snow and everywhere that Mary went the lamb was sure to go"
LAYER = 1


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


def simulate_quantize(arr, bits):
    if bits == 32:
        return arr.copy()
    if bits == 16:
        return arr.astype(np.float16).astype(np.float32)
    if bits == 8:
        scale = np.abs(arr).max(axis=-1, keepdims=True)
        scale[scale < 1e-10] = 1.0
        q = np.round(arr / scale * 127).clip(-128, 127).astype(np.int8)
        return q.astype(np.float32) * scale / 127
    if bits == 4:
        scale = np.abs(arr).max(axis=-1, keepdims=True)
        scale[scale < 1e-10] = 1.0
        q = np.round(arr / scale * 7).clip(-8, 7).astype(np.int8)
        return q.astype(np.float32) * scale / 7
    raise ValueError(f"Unsupported bits: {bits}")


def build_vocab_pos1(model, layer, device, prefix_token, batch=256):
    vocab_size = model.wte.weight.shape[0]
    chunks = []
    for start in range(0, vocab_size, batch):
        end = min(start + batch, vocab_size)
        ids = torch.tensor([[prefix_token, t] for t in range(start, end)]).to(device)
        with torch.no_grad():
            out = model(ids)
        h = out.hidden_states[layer][:, 1, :].cpu().numpy()
        chunks.append(h)
    return np.concatenate(chunks, axis=0)


def main():
    device = "cpu"
    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True).to(device)
    model.eval()
    eot = tokenizer.eos_token_id

    # Encode test text with EOT prepended
    raw = tokenizer.encode(TEXT)
    tokens = [eot] + raw
    trailing = tokens[1:]
    unique_trailing = set(trailing)
    n_t = len(trailing)
    u = len(unique_trailing)
    print(f"Text: {TEXT[:50]}...")
    print(f"Trailing tokens: {n_t}, unique: {u}")
    print(f"Layer: L{LAYER}")

    # Forward pass at full precision
    with torch.no_grad():
        out = model(torch.tensor([tokens]).to(device))
    h_full = out.hidden_states[LAYER].squeeze(0).cpu().numpy()

    # Build f32 corrected vocab
    print(f"\nBuilding corrected vocab at L{LAYER}...")
    vocab_f32 = build_vocab_pos1(model, LAYER, device, eot)
    print(f"  vocab shape: {vocab_f32.shape}, avg norm: {np.mean(np.linalg.norm(vocab_f32, axis=1)):.2f}")

    # Compute clean per-token bias from refs (sink-skipped)
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
            rh = model(torch.tensor([rt]).to(device)).hidden_states[LAYER].squeeze(0).cpu().numpy()
        rt_trailing = rh[1:]
        n_r = rt_trailing.shape[0]
        bias_accum.append((rt_trailing.sum(axis=0) - vocab_f32[rt[1:]].sum(axis=0)) / n_r)
    bias_f32 = np.mean(bias_accum, axis=0)
    print(f"  per-token bias norm (f32): {np.linalg.norm(bias_f32):.2f}")

    # Static decomposition baseline (unaffected — confirm)
    wte = model.wte.weight.detach().cpu().numpy()
    static_target_f32 = wte[raw].sum(axis=0)
    static_unique = set(raw)

    print(f"\n{'='*70}")
    print("PART 1: Static decomposition (unchanged baseline)")
    print(f"{'='*70}")
    print(f"  {'Bits':>6s}  {'Recovery':>12s}")
    print("  " + "-" * 25)
    for bits in [32, 16, 8, 4]:
        wte_q = simulate_quantize(wte, bits)
        target_q = wte_q[raw].sum(axis=0)
        rec = decompose_greedy(target_q, wte_q, len(raw) + 10)
        hits = len(static_unique & set(rec))
        print(f"  {bits:>4d}b   {hits:>4d}/{len(static_unique):<4d}")

    print(f"\n{'='*70}")
    print(f"PART 2: Contextual decomposition (sink-skip + N-debias) at L{LAYER}")
    print(f"{'='*70}")
    print(f"  {'Bits':>6s}  {'Recovery':>12s}  {'BiasNorm':>10s}  {'TargetNorm':>12s}  tokens")
    print("  " + "-" * 90)

    for bits in [32, 16, 8, 4]:
        # Quantize target hidden states
        h_q = simulate_quantize(h_full, bits)
        # Quantize vocab and bias the same way
        vocab_q = simulate_quantize(vocab_f32, bits)
        bias_q = simulate_quantize(bias_f32, bits)

        trailing_target = h_q[1:].sum(axis=0)
        debiased = trailing_target - n_t * bias_q

        rec = decompose_greedy(debiased, vocab_q, n_t + 10)
        hits = len(unique_trailing & set(rec))
        toks = [tokenizer.decode([r]) for r in rec[:6]]
        bn = np.linalg.norm(bias_q)
        tn = np.linalg.norm(debiased)
        print(f"  {bits:>4d}b   {hits:>4d}/{u:<4d}    {bn:>10.2f}    {tn:>12.2f}    {toks}")

    del vocab_f32
    gc.collect()


if __name__ == "__main__":
    main()
