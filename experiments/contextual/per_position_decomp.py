#!/usr/bin/env python3
"""Per-position decomposition: for each position in the sentence, find the
closest vocab entry to that position's hidden state directly. No summing,
no bias, just per-vector nearest-neighbor search.

If position drift is the long-sentence problem, then early positions should
still recover their tokens reliably while late positions degrade. This
gives us a "first N tokens" anchor for any sentence length.
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

    test_texts = [("short", SHORT), ("medium", MEDIUM), ("gettysburg", GETTYSBURG)]

    layers = [1, 6, 11, 12]
    print(f"\nLAYER SWEEP: per-position top-1 recovery")
    print(f"  {'text':12s}  {'n':>3s}  {'L1':>8s}  {'L6':>8s}  {'L11':>8s}  {'L12':>8s}")
    print("  " + "-" * 60)

    for name, text in test_texts:
        tt = [eot] + tokenizer.encode(text)
        with torch.no_grad():
            out = model(torch.tensor([tt]).to(device))
        all_hs = [h.squeeze(0).cpu().numpy() for h in out.hidden_states]
        n = len(tt) - 1

        results = []
        for L in layers:
            v = build_vocab_pos1(model, L, device, eot)
            vocab_norms = np.linalg.norm(v, axis=1)
            vocab_norms[vocab_norms < 1e-10] = 1.0
            trailing = all_hs[L][1:]
            correct = 0
            for i in range(n):
                target = trailing[i]
                t_norm = np.linalg.norm(target)
                if t_norm < 1e-10:
                    continue
                sims = v @ target / (vocab_norms * t_norm)
                top1 = int(np.argmax(sims))
                if top1 == tt[1+i]:
                    correct += 1
            results.append(f"{correct}/{n}")
            del v
            gc.collect()
        print(f"  {name:12s}  {n:>3d}  {results[0]:>8s}  {results[1]:>8s}  {results[2]:>8s}  {results[3]:>8s}")

    return  # skip the rest of the original code

    L = 1
    print(f"Building vocab at L{L}...")
    v = build_vocab_pos1(model, L, device, eot)
    vocab_norms = np.linalg.norm(v, axis=1)
    vocab_norms[vocab_norms < 1e-10] = 1.0

    for name, text in test_texts:
        tt = [eot] + tokenizer.encode(text)
        with torch.no_grad():
            hs = model(torch.tensor([tt]).to(device)).hidden_states[L].squeeze(0).cpu().numpy()
        trailing = hs[1:]
        trailing_tokens = tt[1:]
        n = len(trailing_tokens)

        # For each position, find the closest vocab entry to the hidden state
        per_pos_correct = 0
        per_pos_top5_correct = 0
        per_pos_top20_correct = 0
        wrong_picks = []

        for i in range(n):
            target = trailing[i]
            t_norm = np.linalg.norm(target)
            if t_norm < 1e-10:
                continue
            sims = v @ target / (vocab_norms * t_norm)
            top = np.argsort(-sims)[:20]
            true_token = trailing_tokens[i]
            if int(top[0]) == true_token:
                per_pos_correct += 1
            if true_token in top[:5]:
                per_pos_top5_correct += 1
            if true_token in top[:20]:
                per_pos_top20_correct += 1
            if int(top[0]) != true_token and i < 15:
                wrong_picks.append((i, tokenizer.decode([true_token]),
                                    tokenizer.decode([int(top[0])])))

        print(f"\n  {name:12s} (n={n}):")
        print(f"    per-position top-1: {per_pos_correct}/{n}  ({100*per_pos_correct/n:.1f}%)")
        print(f"    per-position top-5: {per_pos_top5_correct}/{n}  ({100*per_pos_top5_correct/n:.1f}%)")
        print(f"    per-position top-20: {per_pos_top20_correct}/{n}  ({100*per_pos_top20_correct/n:.1f}%)")
        if wrong_picks:
            print(f"    sample wrong picks (first 15 positions): {wrong_picks[:5]}")

        # Now break down by position bucket: first 5, first 10, etc.
        print(f"\n    Per-position top-1 accuracy by position bucket:")
        for start in [0, 5, 10, 15, 20, 25, 30, 40, 50]:
            end = min(start + 5, n)
            if start >= n:
                break
            correct = 0
            for i in range(start, end):
                target = trailing[i]
                t_norm = np.linalg.norm(target)
                if t_norm < 1e-10:
                    continue
                sims = v @ target / (vocab_norms * t_norm)
                top1 = int(np.argmax(sims))
                if top1 == trailing_tokens[i]:
                    correct += 1
            print(f"      positions {start:>2d}-{end-1:>2d}: {correct}/{end-start}")


if __name__ == "__main__":
    main()
