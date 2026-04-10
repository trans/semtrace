#!/usr/bin/env python3
"""End-to-end pipeline: L1 bag → LLM reordering → L2 verification → swap refinement.

1. Per-position L1 decomposition → perfect bag of token IDs
2. Ask Mistral 7B (via Ollama) to arrange the words into a sentence
3. Map Mistral's word ordering back to the original token IDs
4. Score by L2 distance to target
5. Pairwise swap refinement: try all (N choose 2) swaps, keep best, repeat
6. Report final L2 and position-match
"""
import json
import subprocess
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


SENTENCES = [
    "the old man fished from the boat while the sun rose slowly",
    "snow fell quietly on the mountain pass as the travelers slept",
    "Mary had a little lamb its fleece was white as snow and everywhere that Mary went the lamb was sure to go",
]


def get_pooled(model, tokens, device):
    with torch.no_grad():
        out = model(torch.tensor([tokens]).to(device), output_hidden_states=True)
    l12 = out.hidden_states[12].squeeze(0)
    return l12[1:].mean(dim=0).cpu().numpy()


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


def per_position_decompose(model, tokens, layer, vocab, device):
    with torch.no_grad():
        out = model(torch.tensor([tokens]).to(device), output_hidden_states=True)
    hs = out.hidden_states[layer].squeeze(0).cpu().numpy()
    trailing = hs[1:]
    vocab_norms = np.linalg.norm(vocab, axis=1)
    vocab_norms[vocab_norms < 1e-10] = 1.0
    recovered = []
    for i in range(trailing.shape[0]):
        target = trailing[i]
        t_norm = np.linalg.norm(target)
        sims = vocab @ target / (vocab_norms * t_norm)
        recovered.append(int(np.argmax(sims)))
    return recovered


def ask_mistral(bag_words):
    """Ask Mistral 7B to arrange words into a sentence."""
    word_list = ', '.join(bag_words)
    prompt = (
        f"Task: Put these shuffled words back in the correct order to form a "
        f"grammatical English sentence.\n"
        f"Rules: Use EVERY word exactly once. Do NOT add any new words. "
        f"Do NOT remove any words.\n\n"
        f"Shuffled words: {word_list}\n\n"
        f"Correct sentence:"
    )
    resp = subprocess.run(
        ['curl', '-s', 'http://localhost:11434/api/generate', '-d',
         json.dumps({'model': 'mistral:7b', 'prompt': prompt, 'stream': False,
                     'options': {'temperature': 0.0, 'num_predict': 100}})],
        capture_output=True, text=True
    )
    return json.loads(resp.stdout)['response'].strip().split('\n')[0]


def map_llm_output_to_bag(llm_text, bag_token_ids, tokenizer):
    """Map LLM's word-level output back to an ordering of the original bag tokens.

    Strategy: tokenize LLM output, then greedily match each output token to
    the closest available bag token. Append any unmatched bag tokens at the end.
    """
    llm_tokens = tokenizer.encode(llm_text)
    remaining_bag = list(bag_token_ids)
    ordering = []

    for lt in llm_tokens:
        if lt in remaining_bag:
            remaining_bag.remove(lt)
            ordering.append(lt)

    # Append any remaining bag tokens that weren't matched
    ordering.extend(remaining_bag)
    return ordering


def swap_refine(model, target_emb, sequence, device, eot, max_rounds=15):
    """Pairwise swap refinement: try all (N choose 2) swaps, keep the one
    that most reduces L2. Repeat until no swap improves."""
    current = list(sequence)
    n = len(current)

    def l2_dist(seq):
        emb = get_pooled(model, [eot] + seq, device)
        return float(np.linalg.norm(target_emb - emb))

    current_l2 = l2_dist(current)

    for round_num in range(max_rounds):
        best_swap = None
        best_l2 = current_l2

        for i in range(n):
            for j in range(i + 1, n):
                candidate = list(current)
                candidate[i], candidate[j] = candidate[j], candidate[i]
                c_l2 = l2_dist(candidate)
                if c_l2 < best_l2:
                    best_l2 = c_l2
                    best_swap = (i, j)

        if best_swap is None:
            break

        i, j = best_swap
        current[i], current[j] = current[j], current[i]
        current_l2 = best_l2
        print(f"      round {round_num + 1}: swap pos {i},{j} → L2={current_l2:.4f}")

        if current_l2 < 0.01:
            break

    return current, current_l2


def main():
    device = "cpu"
    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True).to(device)
    model.eval()
    eot = tokenizer.eos_token_id

    print("Building L1 vocab...")
    v_l1 = build_vocab_pos1(model, 1, device, eot)

    import random

    for sent in SENTENCES:
        print(f"\n{'='*70}")
        print(f"Original: {sent!r}")
        true_tokens = [eot] + tokenizer.encode(sent)
        bag = true_tokens[1:]
        n_t = len(bag)
        target_emb = get_pooled(model, true_tokens, device)

        # Stage 1: L1 per-position bag recovery
        recovered_bag = per_position_decompose(model, true_tokens, 1, v_l1, device)
        bag_correct = sum(1 for a, b in zip(recovered_bag, bag) if a == b)
        print(f"  L1 bag recovery: {bag_correct}/{n_t}")

        # Stage 2: Ask Mistral
        bag_words = sent.split()
        random.seed(42)
        shuffled_words = list(bag_words)
        random.shuffle(shuffled_words)
        print(f"  Shuffled words: {', '.join(shuffled_words)}")

        llm_result = ask_mistral(shuffled_words)
        print(f"  Mistral output: {llm_result}")

        # Map LLM output to token ordering
        llm_ordering = map_llm_output_to_bag(llm_result, list(bag), tokenizer)
        llm_l2 = float(np.linalg.norm(target_emb - get_pooled(model, [eot] + llm_ordering, device)))
        llm_pm = sum(1 for a, b in zip(llm_ordering, bag) if a == b)
        llm_strs = [tokenizer.decode([t]) for t in llm_ordering]
        print(f"  Mapped to tokens: {' '.join(llm_strs)}")
        print(f"  LLM ordering: pos-match={llm_pm}/{n_t}, L2={llm_l2:.4f}")

        # Stage 3: Swap refinement
        print(f"  Swap refinement:")
        refined, refined_l2 = swap_refine(model, target_emb, llm_ordering, device, eot)
        refined_pm = sum(1 for a, b in zip(refined, bag) if a == b)
        refined_strs = [tokenizer.decode([t]) for t in refined]
        print(f"  After refinement: pos-match={refined_pm}/{n_t}, L2={refined_l2:.4f}")
        print(f"  Refined: {' '.join(refined_strs)}")

        true_strs = [tokenizer.decode([t]) for t in bag]
        print(f"  True:    {' '.join(true_strs)}")
        if refined_pm == n_t:
            print(f"  *** PERFECT RECOVERY ***")


if __name__ == "__main__":
    main()
