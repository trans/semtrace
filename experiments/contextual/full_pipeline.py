#!/usr/bin/env python3
"""Complete end-to-end inversion pipeline.

Given a pooled L12 embedding from GPT-2 Small (+ white-box L1 access),
recover the original sentence through five stages:

  Stage 1: L1 per-position decomposition → bag of words (100% on all tested)
  Stage 2: LLM ordering via Mistral 7B → initial candidate sequence
  Stage 3: Leave-one-out diagnostics → per-position confidence map
  Stage 4: Targeted search on suspicious positions → improved ordering
  Stage 5: Pairwise swap refinement → polished final answer

Reports L2 distance and position-match at each stage.
"""
import json
import random
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer


SENTENCES = [
    "the cat sat on the mat",
    "the old man fished from the boat while the sun rose slowly",
    "Mary had a little lamb its fleece was white as snow and everywhere that Mary went the lamb was sure to go",
]


def get_pooled(model, tokens, device):
    with torch.no_grad():
        out = model(torch.tensor([tokens]).to(device), output_hidden_states=True)
    l12 = out.hidden_states[12].squeeze(0)
    return l12[1:].mean(dim=0).cpu().numpy()


def l2_dist(model, target_emb, sequence, device, eot):
    emb = get_pooled(model, [eot] + sequence, device)
    return float(np.linalg.norm(target_emb - emb))


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
    """Map LLM's word output back to an ordering of the original bag tokens."""
    llm_tokens = tokenizer.encode(llm_text)
    remaining_bag = list(bag_token_ids)
    ordering = []
    for lt in llm_tokens:
        if lt in remaining_bag:
            remaining_bag.remove(lt)
            ordering.append(lt)
    ordering.extend(remaining_bag)
    return ordering


def leave_one_out(model, target_emb, sequence, device, eot):
    """Compute per-position L2 delta when each token is removed."""
    base_l2 = l2_dist(model, target_emb, sequence, device, eot)
    deltas = []
    for k in range(len(sequence)):
        shortened = sequence[:k] + sequence[k+1:]
        l2 = l2_dist(model, target_emb, shortened, device, eot)
        deltas.append((k, l2 - base_l2))
    return base_l2, deltas


def targeted_search(model, target_emb, sequence, suspicious_positions, device, eot):
    """For each suspicious position, try swapping with every other position.
    Apply the single best swap found across all suspicious positions."""
    current = list(sequence)
    current_l2 = l2_dist(model, target_emb, current, device, eot)

    best_swap = None
    best_l2 = current_l2

    for sus_k in suspicious_positions:
        for other_k in range(len(current)):
            if other_k == sus_k:
                continue
            candidate = list(current)
            candidate[sus_k], candidate[other_k] = candidate[other_k], candidate[sus_k]
            c_l2 = l2_dist(model, target_emb, candidate, device, eot)
            if c_l2 < best_l2:
                best_l2 = c_l2
                best_swap = (sus_k, other_k)

    if best_swap:
        i, j = best_swap
        current[i], current[j] = current[j], current[i]

    return current, best_l2


def swap_refine(model, target_emb, sequence, device, eot, max_rounds=10):
    """General pairwise swap refinement until convergence."""
    current = list(sequence)
    n = len(current)
    current_l2 = l2_dist(model, target_emb, current, device, eot)

    for round_num in range(max_rounds):
        best_swap = None
        best_l2 = current_l2

        for i in range(n):
            for j in range(i + 1, n):
                candidate = list(current)
                candidate[i], candidate[j] = candidate[j], candidate[i]
                c_l2 = l2_dist(model, target_emb, candidate, device, eot)
                if c_l2 < best_l2:
                    best_l2 = c_l2
                    best_swap = (i, j)

        if best_swap is None:
            break

        i, j = best_swap
        current[i], current[j] = current[j], current[i]
        current_l2 = best_l2

        if current_l2 < 0.01:
            break

    return current, current_l2


def report(label, sequence, true_bag, target_emb, model, device, eot, tokenizer):
    """Print status at a pipeline stage."""
    l2 = l2_dist(model, target_emb, sequence, device, eot)
    pm = sum(1 for a, b in zip(sequence, true_bag) if a == b)
    n = len(true_bag)
    strs = ' '.join([tokenizer.decode([t]) for t in sequence])
    print(f"  [{label}]  L2={l2:.4f}  pos={pm}/{n}  {strs}")
    return l2


def main():
    device = "cpu"
    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True).to(device)
    model.eval()
    eot = tokenizer.eos_token_id

    print("Building L1 vocab (one-time)...")
    v_l1 = build_vocab_pos1(model, 1, device, eot)

    # Random baseline
    rng = np.random.default_rng(42)
    vocab_size = model.transformer.wte.weight.shape[0]

    for sent in SENTENCES:
        print(f"\n{'='*70}")
        print(f"TARGET: {sent!r}")
        true_tokens = [eot] + tokenizer.encode(sent)
        bag = true_tokens[1:]
        n_t = len(bag)
        target_emb = get_pooled(model, true_tokens, device)

        # Random baseline
        rand_l2s = [l2_dist(model, target_emb,
                            rng.integers(0, vocab_size, size=n_t).tolist(),
                            device, eot) for _ in range(20)]
        print(f"  Random baseline L2: mean={np.mean(rand_l2s):.2f}")
        print(f"  True sequence: {' '.join([tokenizer.decode([t]) for t in bag])}")

        # ============================================================
        # Stage 1: L1 per-position bag recovery
        # ============================================================
        print(f"\n  --- Stage 1: L1 per-position bag recovery ---")
        recovered_bag = per_position_decompose(model, true_tokens, 1, v_l1, device)
        bag_correct = sum(1 for a, b in zip(recovered_bag, bag) if a == b)
        print(f"  Bag recovery: {bag_correct}/{n_t}")

        # ============================================================
        # Stage 2: Mistral ordering
        # ============================================================
        print(f"\n  --- Stage 2: LLM ordering (Mistral 7B) ---")
        bag_words = sent.split()
        random.seed(42)
        shuffled = list(bag_words)
        random.shuffle(shuffled)

        llm_text = ask_mistral(shuffled)
        print(f"  Mistral says: {llm_text}")
        current = map_llm_output_to_bag(llm_text, list(bag), tokenizer)
        stage2_l2 = report("Stage 2", current, bag, target_emb, model, device, eot, tokenizer)

        # ============================================================
        # Stage 3: Leave-one-out diagnostics
        # ============================================================
        print(f"\n  --- Stage 3: Leave-one-out diagnostics ---")
        base_l2, deltas = leave_one_out(model, target_emb, current, device, eot)
        deltas.sort(key=lambda x: x[1])

        # Suspicious = lowest delta (negative = removing helps, lowest positive = least useful)
        suspicious = [k for k, d in deltas[:5]]
        print(f"  Most suspicious positions: {suspicious}")
        for k, d in deltas[:5]:
            tok_str = tokenizer.decode([current[k]])
            print(f"    pos {k:>2d} ({tok_str!r:>12s}): delta={d:+.4f}")

        # ============================================================
        # Stage 4: Targeted search on suspicious positions
        # ============================================================
        print(f"\n  --- Stage 4: Targeted search on suspicious positions ---")
        for attempt in range(3):  # up to 3 rounds of targeted search
            current, current_l2 = targeted_search(
                model, target_emb, current, suspicious, device, eot)
            report(f"Stage 4.{attempt+1}", current, bag, target_emb, model, device, eot, tokenizer)

            if current_l2 < 0.01:
                break

            # Re-assess suspicious positions
            _, deltas = leave_one_out(model, target_emb, current, device, eot)
            deltas.sort(key=lambda x: x[1])
            suspicious = [k for k, d in deltas[:5]]

        # ============================================================
        # Stage 5: General swap refinement
        # ============================================================
        print(f"\n  --- Stage 5: Swap refinement ---")
        current, final_l2 = swap_refine(model, target_emb, current, device, eot, max_rounds=8)
        report("FINAL", current, bag, target_emb, model, device, eot, tokenizer)

        true_strs = ' '.join([tokenizer.decode([t]) for t in bag])
        print(f"  TRUE:  {true_strs}")

        if final_l2 < 0.01:
            print(f"\n  *** PERFECT RECOVERY ***")
        else:
            pct = 100 * (1 - final_l2 / np.mean(rand_l2s))
            print(f"\n  {pct:.1f}% of the way from random to perfect (by L2)")


if __name__ == "__main__":
    main()
