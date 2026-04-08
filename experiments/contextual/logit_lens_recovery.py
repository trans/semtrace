#!/usr/bin/env python3
"""Logit-lens recovery: project per-position hidden states through wte.T to
get next-token logits, then read off the argmax-predicted sequence and check
how well it matches the actual input.

For an autoregressive model, hidden[L][k] @ wte.T gives the logit distribution
over what comes at position k+1. For a well-trained model on fluent English,
the argmax should often be the actual next token. This converts embedding
inversion into "ask the model to predict the input it saw."

Tested at L1, L6, L11, L12 across short / medium / long texts.
"""
import gc
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


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


def main():
    device = "cpu"
    print("Loading GPT-2 Small (with LM head)...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True).to(device)
    model.eval()
    eot = tokenizer.eos_token_id

    # ln_f and wte for the lens
    ln_f = model.transformer.ln_f
    wte = model.transformer.wte.weight  # (vocab, 768) - same as the LM head by weight tying

    test_texts = [("short", SHORT), ("medium", MEDIUM), ("gettysburg", GETTYSBURG)]
    layers = [1, 6, 11, 12]

    for name, text in test_texts:
        tt = [eot] + tokenizer.encode(text)
        with torch.no_grad():
            out = model(torch.tensor([tt]).to(device))
        all_hs = out.hidden_states  # tuple of tensors, one per layer + embed

        # The actual sequence (excluding the prepended EOT)
        actual_tokens = tt[1:]  # positions 1..N
        n = len(actual_tokens)

        print(f"\n{name} (n={n}):")
        for L in layers:
            hidden = all_hs[L].squeeze(0)  # (n+1, 768)

            # Apply ln_f manually if pre-ln_f layer
            if L < 12:
                hidden_normalized = ln_f(hidden)
            else:
                hidden_normalized = hidden  # L12 is already post-ln_f

            # Compute logits at every position
            with torch.no_grad():
                logits = hidden_normalized @ wte.T  # (n+1, vocab)

            # For each position k in 0..n-1, the logits at k predict position k+1
            # So we want argmax(logits[k]) to equal actual_tokens[k] (which is the
            # token at position k+1 in the original sequence)
            top1_predictions = torch.argmax(logits[:-1], dim=-1).cpu().numpy()  # length n
            correct = sum(1 for i in range(n) if int(top1_predictions[i]) == actual_tokens[i])

            # Also: per-position rank of the correct token
            ranks = []
            for i in range(n):
                row = logits[i]
                target = actual_tokens[i]
                rank = int((row > row[target]).sum().item())
                ranks.append(rank)

            avg_rank = float(np.mean(ranks))
            top5 = sum(1 for r in ranks if r < 5)
            top20 = sum(1 for r in ranks if r < 20)

            print(f"  L{L:>2d}: top1={correct}/{n} ({100*correct/n:.0f}%)  "
                  f"top5={top5}/{n}  top20={top20}/{n}  avg_rank={avg_rank:.1f}")

            # On the medium/long texts, show a few example positions
            if name in ("medium", "gettysburg") and L == 12:
                print(f"      sample (L{L}): " + " | ".join(
                    f"{tokenizer.decode([actual_tokens[i]])}->"
                    f"{tokenizer.decode([int(top1_predictions[i])])}"
                    for i in [0, 5, 10, 20, 30, min(50, n-1)] if i < n
                ))


if __name__ == "__main__":
    main()
