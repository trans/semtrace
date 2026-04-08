#!/usr/bin/env python3
"""Measure the actual structure of the per-token bias and per-block delta.

Three distinct quantities, measured across multiple reference sentences at every layer:

  1. Bias[L]      = (sentence hidden sum at L) - (vocab hidden sum at L), averaged per token
                    "the context-vs-isolation offset at layer L"

  2. Delta[L]     = hidden[L] - hidden[L-1], per position
                    "what block L actually adds to each position's residual stream"

  3. After-bias signal alignment: at each layer, what's the cosine of the
     bias-subtracted sentence sum to the corresponding static embedding sum?
     This is the original question — does subtracting the bias reveal a
     static-alignable signal?

For (1) and (2) we report cross-sentence cosine consistency, so we can tell whether
each is a single shared direction or sentence-specific noise.
"""
import gc
import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer


SENTENCES = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "Mary had a little lamb",
    "the quick brown fox jumps over the lazy dog",
    "she went to the store to buy some food",
    "he ran down the long road to the old house",
    "they played in the park with the children",
    "we had dinner at a small restaurant",
]


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


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pairwise_cos(vectors):
    """Average pairwise cosine across a list of vectors."""
    n = len(vectors)
    if n < 2:
        return 1.0
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(cos(vectors[i], vectors[j]))
    return float(np.mean(sims)), float(np.min(sims)), float(np.max(sims))


def main():
    device = "cpu"
    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True).to(device)
    model.eval()
    eot = tokenizer.eos_token_id

    wte = model.wte.weight.detach().cpu().numpy()
    layers = list(range(0, 12))  # L0 (embed) through L11 (last block, pre-ln_f)

    # Encode every sentence with EOT prepended
    sent_token_lists = [[eot] + tokenizer.encode(s) for s in SENTENCES]

    # Forward-pass each sentence once, capture all hidden states
    print("Forward-passing sentences...")
    sent_hiddens = []  # list of [layer -> (n_tok, 768)]
    for tt in sent_token_lists:
        with torch.no_grad():
            out = model(torch.tensor([tt]).to(device))
        sent_hiddens.append([h.squeeze(0).cpu().numpy() for h in out.hidden_states])

    # ==================================================================
    # Quantity 1: Per-token bias at each layer (sink-skipped)
    # ==================================================================
    print("\nBuilding corrected vocabs at each layer (this is the slow part)...")
    print(f"  {'Layer':>6s}  {'BiasNorm(avg)':>14s}  {'BiasCos(min)':>13s}  {'BiasCos(avg)':>13s}  {'BiasCos(max)':>13s}")
    print("  " + "-" * 75)

    bias_per_layer = {}  # L -> [bias for each sentence]
    avg_bias_per_layer = {}

    for L in layers:
        if L == 0:
            # L0 is just the embedding (wte+wpe). No vocab needed; bias is wpe-shaped.
            # We'll compute it relative to wte directly.
            biases_this_layer = []
            for tt, hs_list in zip(sent_token_lists, sent_hiddens):
                trailing = hs_list[L][1:]
                trailing_tokens = tt[1:]
                n_t = trailing.shape[0]
                # At L0, vocab = wte
                vocab_sum = wte[trailing_tokens].sum(axis=0)
                biases_this_layer.append((trailing.sum(axis=0) - vocab_sum) / n_t)
            bias_per_layer[L] = biases_this_layer
            avg_bias_per_layer[L] = np.mean(biases_this_layer, axis=0)
            avg_norm = np.mean([np.linalg.norm(b) for b in biases_this_layer])
            cmin, cavg, cmax = pairwise_cos(biases_this_layer)
            print(f"  L{L:>2d}     {avg_norm:>14.2f}  {cmin:>13.4f}  {cavg:>13.4f}  {cmax:>13.4f}  (vs wte, no vocab built)")
            continue

        # Build sink-corrected vocab at this layer
        v = build_vocab_pos1(model, L, device, eot)

        biases_this_layer = []
        for tt, hs_list in zip(sent_token_lists, sent_hiddens):
            trailing = hs_list[L][1:]
            trailing_tokens = tt[1:]
            n_t = trailing.shape[0]
            vocab_sum = v[trailing_tokens].sum(axis=0)
            biases_this_layer.append((trailing.sum(axis=0) - vocab_sum) / n_t)

        bias_per_layer[L] = biases_this_layer
        avg_bias_per_layer[L] = np.mean(biases_this_layer, axis=0)
        avg_norm = np.mean([np.linalg.norm(b) for b in biases_this_layer])
        cmin, cavg, cmax = pairwise_cos(biases_this_layer)
        print(f"  L{L:>2d}     {avg_norm:>14.2f}  {cmin:>13.4f}  {cavg:>13.4f}  {cmax:>13.4f}")

        # Free vocab memory immediately — we don't need it for delta or alignment tests
        del v
        gc.collect()

    # ==================================================================
    # Quantity 2: Per-block delta = hidden[L] - hidden[L-1]
    # ==================================================================
    print(f"\n{'='*78}")
    print("Per-block delta = hidden[L] - hidden[L-1]")
    print("Averaged across all trailing positions, then averaged across sentences.")
    print("Reports cross-sentence consistency of the per-block contribution.")
    print(f"{'='*78}")
    print(f"  {'Block':>6s}  {'DeltaNorm(avg)':>15s}  {'DeltaCos(min)':>14s}  {'DeltaCos(avg)':>14s}  {'DeltaCos(max)':>14s}")
    print("  " + "-" * 80)

    delta_per_block = {}  # L -> [delta_per_sentence], where each delta is the mean per-position delta for that sentence
    avg_delta_per_block = {}

    for L in range(1, 12):  # blocks 1..11 each have a delta from L-1
        sentence_deltas = []
        for hs_list in sent_hiddens:
            trailing_below = hs_list[L - 1][1:]
            trailing_above = hs_list[L][1:]
            # Per-position delta, then mean across positions
            position_deltas = trailing_above - trailing_below
            sentence_deltas.append(position_deltas.mean(axis=0))

        delta_per_block[L] = sentence_deltas
        avg_delta_per_block[L] = np.mean(sentence_deltas, axis=0)
        avg_norm = np.mean([np.linalg.norm(d) for d in sentence_deltas])
        cmin, cavg, cmax = pairwise_cos(sentence_deltas)
        print(f"  L{L:>2d}     {avg_norm:>15.2f}  {cmin:>14.4f}  {cavg:>14.4f}  {cmax:>14.4f}")

    # ==================================================================
    # Quantity 3: After-bias static alignment
    #   At each layer, after subtracting the per-token bias × N from the
    #   sentence sum, what's the cosine to the corresponding static-sum?
    # ==================================================================
    print(f"\n{'='*78}")
    print("After-bias static alignment")
    print("After subtracting N*bias[L] from the sentence sum at L, cosine to the")
    print("corresponding static embedding sum (wte[tokens].sum). High cosine means")
    print("the residual lives in the static embedding direction — i.e., the bias")
    print("subtraction has uncovered something that looks like a sum of static tokens.")
    print(f"{'='*78}")

    # Pick one test sentence (the cat sat on the mat) for clarity
    test_sent = "the cat sat on the mat"
    test_tokens = [eot] + tokenizer.encode(test_sent)
    with torch.no_grad():
        test_hs = model(torch.tensor([test_tokens]).to(device)).hidden_states
    test_hiddens_per_layer = [h.squeeze(0).cpu().numpy() for h in test_hs]
    trailing_static_sum = wte[test_tokens[1:]].sum(axis=0)

    print(f"\n  Test sentence: {test_sent!r}")
    print(f"  Static sum norm: {np.linalg.norm(trailing_static_sum):.2f}")
    print(f"  {'Layer':>6s}  {'cos(raw, static)':>17s}  {'cos(debiased, static)':>22s}")
    print("  " + "-" * 55)

    for L in layers:
        n_t = len(test_tokens) - 1
        sentence_sum = test_hiddens_per_layer[L][1:].sum(axis=0)
        bias = avg_bias_per_layer[L]
        debiased = sentence_sum - n_t * bias

        c_raw = cos(sentence_sum, trailing_static_sum)
        c_deb = cos(debiased, trailing_static_sum)
        print(f"  L{L:>2d}     {c_raw:>17.4f}  {c_deb:>22.4f}")


if __name__ == "__main__":
    main()
