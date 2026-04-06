#!/usr/bin/env python3
"""Build contextual vocabulary: embed every token individually through GPT-2.
Saves to disk as numpy .npy files — one per layer.

Usage:
  python3 build_ctx_vocab.py [--model gpt2] [--layers 6,11] [--outdir .]
"""
import argparse
import numpy as np
import torch
from transformers import GPT2Model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--layers", default="6,11", help="Comma-separated layer indices. Note: hidden_states[12] is post-ln_f, not comparable to 0-11.")
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]

    print(f"Loading {args.model}...")
    model = GPT2Model.from_pretrained(args.model, output_hidden_states=True).to(args.device)
    model.eval()
    vocab_size = model.wte.weight.shape[0]
    print(f"Vocab: {vocab_size}, Layers requested: {layers}")

    batch_size = 256
    for layer in layers:
        print(f"\nBuilding contextual vocab at layer {layer}...")
        all_embs = []
        for start in range(0, vocab_size, batch_size):
            end = min(start + batch_size, vocab_size)
            ids = torch.arange(start, end).unsqueeze(1).to(args.device)
            with torch.no_grad():
                out = model(ids)
            h = out.hidden_states[layer].squeeze(1).cpu().numpy()
            all_embs.append(h)
            if start % 10000 == 0:
                print(f"  {start}/{vocab_size}...", flush=True)

        vocab = np.concatenate(all_embs, axis=0)
        path = f"{args.outdir}/ctx_vocab_L{layer}.npy"
        np.save(path, vocab)
        print(f"  Saved {path}: {vocab.shape}, avg norm: {np.mean(np.linalg.norm(vocab, axis=1)):.1f}")

        # Free memory
        del all_embs, vocab

    # Also save static embeddings
    wte = model.wte.weight.detach().cpu().numpy()
    np.save(f"{args.outdir}/static_vocab.npy", wte)
    print(f"\nSaved static_vocab.npy: {wte.shape}")

if __name__ == "__main__":
    main()
