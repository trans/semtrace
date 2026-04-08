#!/usr/bin/env python3
"""Re-run experiment 011 with sink-skip + N-corrected debias.

Tests the same six sentences from the original experiment 011 against the
corrected method: prepend <|endoftext|>, exclude position 0, subtract N×bias
where N is the trailing token count and bias is computed from sink-corrected
reference sentences. Reports recovery at L1 (best layer) and L6 (the layer
the original experiment used).
"""
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
    sentences = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "Mary had a little lamb",
        "I love cats and dogs",
        "Four score and seven years ago",
        "the big red car drove fast down the road",
    ]
    layers = [1, 6]  # L1 = best layer; L6 = the layer 011 originally used
    refs = [
        "the quick brown fox jumps over the lazy dog",
        "she went to the store to buy some food",
        "he ran down the long road to the old house",
        "they played in the park with the children",
    ]

    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True).to(device)
    model.eval()
    eot = tokenizer.eos_token_id

    # Pre-encode test sentences (with EOT prepended) and ref sentences
    test_token_lists = [[eot] + tokenizer.encode(s) for s in sentences]
    ref_token_lists = [tokenizer.encode(r) for r in refs]

    # Forward-pass each test sentence and grab all hidden states
    test_hiddens = []  # [sentence_idx][layer] -> hidden states (n_tok, 768)
    for tt in test_token_lists:
        with torch.no_grad():
            out = model(torch.tensor([tt]).to(device))
        test_hiddens.append([h.squeeze(0).cpu().numpy() for h in out.hidden_states])

    # Forward-pass each ref sentence
    ref_hiddens = []
    for rt in ref_token_lists:
        with torch.no_grad():
            out = model(torch.tensor([rt]).to(device))
        ref_hiddens.append([h.squeeze(0).cpu().numpy() for h in out.hidden_states])

    # Run for each layer
    for L in layers:
        print(f"\n{'='*70}")
        print(f"LAYER {L}")
        print(f"{'='*70}")

        # Build the corrected vocab at this layer using EOT as the prefix sink
        print(f"Building corrected contextual vocab at L{L} using <|endoftext|> prefix...")
        vocab = build_layer_vocab_with_prefix(model, L, device, eot)

        # Compute per-token bias from refs (sink-skipped, position 1+)
        bias_accum = []
        for rt, rh_list in zip(ref_token_lists, ref_hiddens):
            rh = rh_list[L]
            trailing = rh[1:]  # skip position 0 (the natural sink in unprefixed input)
            n_t = trailing.shape[0]
            tok_sum = vocab[rt[1:]].sum(axis=0)
            bias_accum.append((trailing.sum(axis=0) - tok_sum) / n_t)
        bias = np.mean(bias_accum, axis=0)
        bn = np.linalg.norm(bias)
        print(f"Per-token bias norm: {bn:.2f}")

        print(f"\n{'Sentence':50s}  {'N':>3s}  {'unique':>6s}  {'recovery':>10s}  tokens")
        print("-" * 130)

        for sent, tt, hs_list in zip(sentences, test_token_lists, test_hiddens):
            hs = hs_list[L]
            # Trailing positions = positions 1..end (skipping the prepended EOT)
            trailing_target = hs[1:].sum(axis=0)
            trailing_tokens = tt[1:]
            unique = set(trailing_tokens)
            n_t = len(trailing_tokens)
            u = len(unique)

            debiased = trailing_target - n_t * bias
            rec = decompose_greedy(debiased, vocab, n_t + 10)
            hits = len(unique & set(rec))
            toks = [tokenizer.decode([r]) for r in rec[:8]]
            label = sent[:48]
            print(f"  {label:50s}  {n_t:>3d}  {u:>6d}  {hits:>4d}/{u:<4d}    {toks}")

        del vocab
        gc.collect()


if __name__ == "__main__":
    main()
