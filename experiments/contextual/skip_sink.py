#!/usr/bin/env python3
"""Skip the attention sink at position 0 and decompose only positions 1..N-1.

Hypothesis: position 0 is an attention-sink dumping ground whose hidden state
is pumped to ~40x normal magnitude by MLPs. It dominates any sentence sum and
swamps decomposition. If we exclude it, the remaining positions should
decompose much more cleanly.
"""
import argparse
import gc
import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer


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


def build_layer_vocab_with_prefix(model, layer, device, prefix_token, batch=256):
    """Build a contextual vocab by passing [prefix_token, t] for each t.
    Returns the hidden state at position 1 (the actual token), avoiding the
    position-0 attention sink.
    """
    vocab_size = model.wte.weight.shape[0]
    chunks = []
    for start in range(0, vocab_size, batch):
        end = min(start + batch, vocab_size)
        ids = torch.tensor([[prefix_token, t] for t in range(start, end)]).to(device)
        with torch.no_grad():
            out = model(ids)
        h = out.hidden_states[layer][:, 1, :].cpu().numpy()  # position 1, not 0
        chunks.append(h)
    return np.concatenate(chunks, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="the cat sat on the mat")
    parser.add_argument("--layers", default="1,3,5,7,9,11")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prefix", default="eot", choices=["eot", "the"], help="Prefix token for vocab building")
    parser.add_argument("--prepend-eot", action="store_true", help="Prepend <|endoftext|> to the target sentence")
    args = parser.parse_args()

    layers_to_test = [int(x) for x in args.layers.split(",")]

    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True).to(args.device)
    model.eval()

    eot_id = tokenizer.eos_token_id  # <|endoftext|> = 50256

    raw_tokens = tokenizer.encode(args.text)
    if args.prepend_eot:
        tokens = [eot_id] + raw_tokens
    else:
        tokens = raw_tokens
    token_strs = [tokenizer.decode([t]) for t in tokens]
    print(f"Text: {args.text!r}")
    print(f"Tokens: {token_strs}")

    # The trailing positions (1..N-1) are the ones we want to recover
    trailing_tokens = tokens[1:]
    trailing_unique = set(trailing_tokens)
    n_trail, u_trail = len(trailing_tokens), len(trailing_unique)
    print(f"Trailing (skipping pos 0): {n_trail} tokens, {u_trail} unique")
    print(f"  → {[tokenizer.decode([t]) for t in trailing_tokens]}")

    sink_token = eot_id if args.prefix == "eot" else tokenizer.encode("the")[0]
    print(f"Using prefix token id {sink_token} ({tokenizer.decode([sink_token])!r}) for vocab building")

    # Forward pass the target sentence
    with torch.no_grad():
        out = model(torch.tensor([tokens]).to(args.device))
    all_hs = [h.squeeze(0).cpu().numpy() for h in out.hidden_states]

    refs = [
        "the quick brown fox jumps over the lazy dog",
        "she went to the store to buy some food",
        "he ran down the long road to the old house",
        "they played in the park with the children",
    ]

    print(f"\n{'L':>3s}  {'BiasN(p1+)':>12s}  {'TrailingNoDeb':>14s}  {'TrailingDeb':>13s}  tokens")
    print("-" * 100)

    for L in layers_to_test:
        # Build a position-1 vocab using "the" as the prefix
        vocab = build_layer_vocab_with_prefix(model, L, args.device, sink_token)

        # Compute bias from refs, using positions 1..end (skip the sink)
        biases = []
        for ref in refs:
            rt = tokenizer.encode(ref)
            with torch.no_grad():
                rh = model(torch.tensor([rt]).to(args.device)).hidden_states[L].squeeze(0).cpu().numpy()
            trailing = rh[1:]
            n_t = trailing.shape[0]
            # token sum from vocab
            tok_sum = vocab[rt[1:]].sum(axis=0)
            biases.append((trailing.sum(axis=0) - tok_sum) / n_t)
        bias = np.mean(biases, axis=0)
        bn = np.linalg.norm(bias)

        # Target: trailing-position sum
        trailing_target = all_hs[L][1:].sum(axis=0)

        # Decompose without bias
        rec_nodeb = decompose_greedy(trailing_target, vocab, n_trail + 10)
        h_nodeb = len(trailing_unique & set(rec_nodeb))

        # Debias with N (= n_trail)
        debiased = trailing_target - n_trail * bias
        rec_deb = decompose_greedy(debiased, vocab, n_trail + 10)
        h_deb = len(trailing_unique & set(rec_deb))
        toks_deb = [tokenizer.decode([r]) for r in rec_deb[:8]]

        print(f"  L{L:>2d}  {bn:>12.2f}  {h_nodeb:>5d}/{u_trail}         {h_deb:>5d}/{u_trail}        {toks_deb}")

        del vocab
        gc.collect()


if __name__ == "__main__":
    main()
