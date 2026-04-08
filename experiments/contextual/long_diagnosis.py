#!/usr/bin/env python3
"""Diagnose why contextual decomposition fails on long sentences.

L1 gets 89% on short sentences but 24% on Gettysburg. Find which of the
following is responsible:

  H1. Bias scaling: maybe N*bias is wrong because per-position bias doesn't
      scale linearly. Test by sweeping different bias multipliers.

  H2. Position drift: tokens at deep positions look less like their vocab
      entries. Test by measuring per-position cosine to the right vocab entry.

  H3. Greedy stopping: maybe the decomposer stops too early. Test by forcing
      it to continue for more steps.

  H4. Token diversity: maybe more unique tokens cause more confusion. Test
      by tracking when wrong tokens are picked.

  H5. Reference sentence length: maybe a bias computed from longer reference
      sentences would be more accurate. Test with long-text references.
"""
import gc
import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer


GETTYSBURG = (
    "Four score and seven years ago our fathers brought forth on this continent, "
    "a new nation, conceived in Liberty, and dedicated to the proposition that "
    "all men are created equal. Now we are engaged in a great civil war, testing "
    "whether that nation, or any nation so conceived and so dedicated, can long endure."
)
MEDIUM = (
    "the quick brown fox jumps over the lazy dog while the cat watches from the "
    "windowsill and the children play in the garden"
)
SHORT = "the cat sat on the mat"

LONG_REFS = [
    "the quick brown fox jumps over the lazy dog and the small cat watches from a high window",
    "she walked slowly to the river and dipped her hand into the cold dark water",
    "he opened the old book and began to read aloud to the children gathered around him",
    "they played in the park with the children and the dogs ran around in circles all afternoon",
    "the old man fished from the boat as the sun rose over the calm water and birds flew",
    "morning sun warmed the sleepy village and the smell of bread drifted from open windows",
]

SHORT_REFS = [
    "the quick brown fox jumps over the lazy dog",
    "she went to the store to buy some food",
    "he ran down the long road to the old house",
    "they played in the park with the children",
]


def decompose_greedy(target, embeddings, max_steps, force=False):
    """If force=True, ignore the residual-decreasing stop condition."""
    residual = target.copy()
    recovered = []
    prev_norm = float("inf")
    for _ in range(max_steps):
        r_norm = np.linalg.norm(residual)
        if r_norm < 0.001:
            break
        if not force and r_norm > prev_norm:
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


def compute_bias(model, tokenizer, refs, layer, device, vocab):
    eot = tokenizer.eos_token_id
    bias_accum = []
    for ref in refs:
        rt = tokenizer.encode(ref)
        with torch.no_grad():
            rh = model(torch.tensor([rt]).to(device)).hidden_states[layer].squeeze(0).cpu().numpy()
        trailing = rh[1:]
        n_r = trailing.shape[0]
        bias_accum.append((trailing.sum(axis=0) - vocab[rt[1:]].sum(axis=0)) / n_r)
    return np.mean(bias_accum, axis=0)


def main():
    device = "cpu"
    L = 1
    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True).to(device)
    model.eval()
    eot = tokenizer.eos_token_id

    print(f"Building corrected vocab at L{L}...")
    v = build_vocab_pos1(model, L, device, eot)

    bias_short = compute_bias(model, tokenizer, SHORT_REFS, L, device, v)
    bias_long = compute_bias(model, tokenizer, LONG_REFS, L, device, v)
    print(f"Short-ref bias norm: {np.linalg.norm(bias_short):.2f}")
    print(f"Long-ref bias norm:  {np.linalg.norm(bias_long):.2f}")
    cos_b = float(np.dot(bias_short, bias_long) / (np.linalg.norm(bias_short) * np.linalg.norm(bias_long)))
    print(f"Cosine between them: {cos_b:.4f}")

    # Forward-pass each test text
    test_texts = [("short", SHORT), ("medium", MEDIUM), ("gettysburg", GETTYSBURG)]
    test_data = []
    for name, text in test_texts:
        tt = [eot] + tokenizer.encode(text)
        with torch.no_grad():
            out = model(torch.tensor([tt]).to(device))
        hs = out.hidden_states[L].squeeze(0).cpu().numpy()
        test_data.append((name, tt, hs))

    # ================================================================
    # H2: Per-position cosine to the right vocab entry
    # ================================================================
    print(f"\n{'='*78}")
    print("H2: Per-position cosine to the correct vocab entry")
    print(f"{'='*78}")
    print("  How well does each position's hidden state resemble its token's vocab entry?")
    print("  If position drift is the problem, deep positions should have low cosine.")
    print()

    for name, tt, hs in test_data:
        trailing = hs[1:]
        trailing_tokens = tt[1:]
        n = len(trailing_tokens)

        cosines = []
        for i in range(n):
            target_vec = v[trailing_tokens[i]]
            actual = trailing[i]
            c = float(np.dot(actual, target_vec) / (np.linalg.norm(actual) * np.linalg.norm(target_vec)))
            cosines.append(c)

        # Bin by position bucket
        avg_first_5 = np.mean(cosines[:5]) if len(cosines) >= 5 else None
        avg_first_10 = np.mean(cosines[:10]) if len(cosines) >= 10 else None
        avg_last_10 = np.mean(cosines[-10:]) if len(cosines) >= 10 else None
        avg_all = np.mean(cosines)

        print(f"  {name:12s} (n={n}): avg cos to vocab")
        print(f"    first 5  positions: {avg_first_5:.4f}" if avg_first_5 else f"    first 5: n/a")
        print(f"    first 10 positions: {avg_first_10:.4f}" if avg_first_10 else f"    first 10: n/a")
        print(f"    last 10  positions: {avg_last_10:.4f}" if avg_last_10 else f"    last 10: n/a")
        print(f"    all positions:      {avg_all:.4f}")
        print(f"    cos range: [{min(cosines):.4f}, {max(cosines):.4f}]")

    # ================================================================
    # H1: Sweep bias multipliers
    # ================================================================
    print(f"\n{'='*78}")
    print("H1: Sweep bias multipliers (default is N)")
    print(f"{'='*78}")
    print("  If N*bias is wrong, a different multiplier should give better results.")
    print()

    for name, tt, hs in test_data:
        trailing_tokens = tt[1:]
        unique = set(trailing_tokens)
        u = len(unique)
        n_t = len(trailing_tokens)
        target_full = hs[1:].sum(axis=0)

        print(f"  {name:12s} (n={n_t}, u={u}):")
        for mult_factor in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]:
            mult = n_t * mult_factor
            tg = target_full - mult * bias_short
            rec = decompose_greedy(tg, v, n_t + 10)
            hits = len(unique & set(rec))
            print(f"    mult = {mult_factor:.2f}*N = {mult:>6.1f}: {hits}/{u}")

    # ================================================================
    # H3: Force greedy to continue past stopping condition
    # ================================================================
    print(f"\n{'='*78}")
    print("H3: Force greedy to ignore the residual-decreasing stop condition")
    print(f"{'='*78}")
    print("  If greedy is stopping too early, forcing more steps should help.")
    print()

    for name, tt, hs in test_data:
        trailing_tokens = tt[1:]
        unique = set(trailing_tokens)
        u = len(unique)
        n_t = len(trailing_tokens)
        target_full = hs[1:].sum(axis=0)
        tg = target_full - n_t * bias_short

        rec_normal = decompose_greedy(tg, v, n_t + 10, force=False)
        rec_force = decompose_greedy(tg, v, n_t + 10, force=True)
        rec_force_ext = decompose_greedy(tg, v, n_t * 3, force=True)

        hits_normal = len(unique & set(rec_normal))
        hits_force = len(unique & set(rec_force))
        hits_force_ext = len(unique & set(rec_force_ext))

        print(f"  {name:12s}: normal {hits_normal}/{u}  forced(N+10) {hits_force}/{u}  forced(N*3) {hits_force_ext}/{u}")
        print(f"    normal stopped after {len(rec_normal)} steps")

    # ================================================================
    # H5: Long-reference bias
    # ================================================================
    print(f"\n{'='*78}")
    print("H5: Long-reference bias vs short-reference bias")
    print(f"{'='*78}")
    print(f"  Short-ref bias norm: {np.linalg.norm(bias_short):.2f}")
    print(f"  Long-ref bias norm:  {np.linalg.norm(bias_long):.2f}")
    print()

    for name, tt, hs in test_data:
        trailing_tokens = tt[1:]
        unique = set(trailing_tokens)
        u = len(unique)
        n_t = len(trailing_tokens)
        target_full = hs[1:].sum(axis=0)

        rec_s = decompose_greedy(target_full - n_t * bias_short, v, n_t + 10)
        rec_l = decompose_greedy(target_full - n_t * bias_long, v, n_t + 10)
        h_s = len(unique & set(rec_s))
        h_l = len(unique & set(rec_l))
        print(f"  {name:12s}: short-bias {h_s}/{u}  long-bias {h_l}/{u}")


if __name__ == "__main__":
    main()
