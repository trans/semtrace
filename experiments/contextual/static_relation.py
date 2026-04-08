#!/usr/bin/env python3
"""Two related questions about how the corrected contextual space relates to
static space:

  A. Direct decomposition of sink-corrected contextual sums against the static
     wte vocabulary. Tests whether sink-skip alone is enough to make contextual
     sums decompose against static. Predicted to fail at L1+ based on the
     near-orthogonal cos(debiased, static) we measured in bias_anatomy.py.

  B. Linear contextual→static map, this time with sink-corrected vocabulary
     pairs. The original 011 result (89.9% identity preservation) used
     sink-contaminated pairs (every contextual side was a single-token forward
     pass = position 0 = sink). With corrected pairs we expect the result to
     be either substantially better (cleaner mapping target) or substantially
     worse (the sink artifacts that the original was implicitly mapping away
     are gone, leaving only the geometric divergence between the two spaces
     to overcome).
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

    wte = model.wte.weight.detach().cpu().numpy()  # (50257, 768)
    print(f"wte: {wte.shape}, avg norm {np.mean(np.linalg.norm(wte, axis=1)):.2f}")

    # Test sentences
    test_sentences = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "Mary had a little lamb",
        "Four score and seven years ago",
    ]
    layers = [1, 6, 11, 12]

    # Reference sentences for bias computation
    refs = [
        "the quick brown fox jumps over the lazy dog",
        "she went to the store to buy some food",
        "he ran down the long road to the old house",
        "they played in the park with the children",
    ]

    # ==================================================================
    # PART A: Direct decomposition of sink-corrected contextual sums
    #         against the static wte vocabulary
    # ==================================================================
    print(f"\n{'='*78}")
    print("PART A: Sink-corrected contextual sum → STATIC vocab decomposition")
    print("(after subtracting per-token contextual bias × N)")
    print(f"{'='*78}")

    # First, we need a "bias" that maps a contextual sum to a static sum.
    # The natural choice: bias[L] = (contextual sum at L) - (static sum)
    # averaged across reference sentences and per token. This is the
    # "static-target" bias, different from the "contextual-vocab-target"
    # bias we use elsewhere.

    for L in layers:
        # Compute static-target bias: how much does a contextual sum at L
        # differ from the static sum on a per-token basis?
        biases = []
        for ref in refs:
            rt = [eot] + tokenizer.encode(ref)
            with torch.no_grad():
                rh = model(torch.tensor([rt]).to(device)).hidden_states[L].squeeze(0).cpu().numpy()
            trailing = rh[1:]
            n_t = trailing.shape[0]
            static_sum = wte[rt[1:]].sum(axis=0)
            biases.append((trailing.sum(axis=0) - static_sum) / n_t)
        bias_static = np.mean(biases, axis=0)
        print(f"\n  L{L}: static-target bias norm = {np.linalg.norm(bias_static):.2f}")

        for sent in test_sentences:
            tt = [eot] + tokenizer.encode(sent)
            with torch.no_grad():
                hs = model(torch.tensor([tt]).to(device)).hidden_states[L].squeeze(0).cpu().numpy()
            trailing_target = hs[1:].sum(axis=0)
            trailing_tokens = tt[1:]
            unique = set(trailing_tokens)
            n_t = len(trailing_tokens)
            u = len(unique)

            # Subtract the static-target bias
            debiased = trailing_target - n_t * bias_static
            rec = decompose_greedy(debiased, wte, n_t + 10)
            hits = len(unique & set(rec))
            toks = [tokenizer.decode([r]) for r in rec[:8]]
            print(f"    {sent[:40]:42s}  {hits}/{u}  {toks}")

    # ==================================================================
    # PART B: Linear contextual→static map with sink-corrected pairs
    # ==================================================================
    print(f"\n{'='*78}")
    print("PART B: Linear contextual→static map (sink-corrected pairs)")
    print(f"{'='*78}")
    print("  Train W (768x768) on (corrected_vocab[L], wte) pairs at each layer.")
    print("  Evaluate by checking whether each token's mapped contextual embedding")
    print("  has the original token as its nearest static neighbor.")
    print()

    for L in layers:
        print(f"  Building corrected vocab at L{L}...")
        v = build_vocab_pos1(model, L, device, eot)

        # Solve W = argmin ||v @ W - wte||
        # via least-squares: W = pinv(v) @ wte, but use lstsq for stability
        print(f"  Fitting linear map L{L}_contextual @ W = wte...")
        W, residuals, rank, sv = np.linalg.lstsq(v, wte, rcond=None)
        print(f"    W shape: {W.shape}, rank: {rank}")

        # Map every contextual embedding through W
        mapped = v @ W

        # For each token, check whether its mapped contextual is closest to wte[token]
        # (do this on a sample of 5000 tokens for speed)
        n_sample = 5000
        rng = np.random.default_rng(42)
        sample_ids = rng.choice(v.shape[0], n_sample, replace=False)

        # Pre-normalize wte for cosine
        wte_norms = np.linalg.norm(wte, axis=1)
        wte_norms[wte_norms < 1e-10] = 1.0
        wte_normed = wte / wte_norms[:, None]

        correct = 0
        rank_sum = 0
        for tid in sample_ids:
            m = mapped[tid]
            m_norm = np.linalg.norm(m)
            if m_norm < 1e-10:
                continue
            sims = wte_normed @ (m / m_norm)
            top = int(np.argmax(sims))
            if top == tid:
                correct += 1
            # Rank of correct token
            rank = int(np.sum(sims > sims[tid]))
            rank_sum += rank

        accuracy = correct / n_sample
        avg_rank = rank_sum / n_sample
        print(f"  L{L}: identity-preservation accuracy = {accuracy*100:.1f}% ({correct}/{n_sample})")
        print(f"  L{L}: average rank of correct token = {avg_rank:.1f} (out of 50,257)")

        # Also: how does the residual of the fit look?
        recon = v @ W
        recon_err = np.linalg.norm(wte - recon, axis=1)
        wte_norm_avg = np.linalg.norm(wte, axis=1)
        rel_err = recon_err / np.maximum(wte_norm_avg, 1e-10)
        print(f"  L{L}: reconstruction relative error (mean): {np.mean(rel_err):.4f}")

        del v, W, mapped
        gc.collect()


if __name__ == "__main__":
    main()
