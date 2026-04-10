#!/usr/bin/env python3
"""Full inversion pipeline on the Mary Had a Little Lamb nursery rhyme.

23 tokens, 21 unique — a real test of scaling beyond 6-token sentences.
Exhaustive search is impossible (21! ≈ 5×10^19), so we rely on beam search.

Pipeline:
  1. Per-position L1 decomposition → bag of words
  2. Bag-constrained beam search at B=8, B=16, B=32 → order recovery
  3. Report position-match and L2 distance
"""
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer


TEXT = "Mary had a little lamb its fleece was white as snow and everywhere that Mary went the lamb was sure to go"


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
        if t_norm < 1e-10:
            recovered.append(0)
            continue
        sims = vocab @ target / (vocab_norms * t_norm)
        recovered.append(int(np.argmax(sims)))
    return recovered


def beam_search_order(model, bag, device, eot, target_emb, beam_width=8):
    """Bag-constrained beam search. At each step, expand each beam with
    each remaining bag token, rank by LM log-prob, keep top B. At the end,
    re-rank by L2 distance to target."""
    beams = [([], list(bag), 0.0)]
    step = 0
    n_total = len(bag)

    while beams[0][1]:  # while remaining tokens exist
        step += 1
        new_beams = []
        for seq, remaining, lp in beams:
            prefix = [eot] + seq
            with torch.no_grad():
                out = model(torch.tensor([prefix]).to(device))
            logits = out.logits.squeeze(0)[-1]
            log_probs = F.log_softmax(logits, dim=-1)
            seen = set()
            for i, t in enumerate(remaining):
                if t in seen:
                    continue
                seen.add(t)
                new_remaining = list(remaining)
                new_remaining.pop(i)
                new_seq = seq + [t]
                new_lp = lp + float(log_probs[t])
                new_beams.append((new_seq, new_remaining, new_lp))
        new_beams.sort(key=lambda x: -x[2])
        beams = new_beams[:beam_width]
        if step % 5 == 0:
            print(f"      step {step}/{n_total}, {len(new_beams)} candidates → kept {len(beams)}")

    # Score completed sequences by L2 embedding distance
    print(f"      scoring {len(beams)} completed beams by L2...")
    scored = []
    for seq, _, lp in beams:
        emb = get_pooled(model, [eot] + seq, device)
        l2 = float(np.linalg.norm(target_emb - emb))
        scored.append((seq, lp, l2))
    scored.sort(key=lambda x: x[2])  # rank by L2
    return scored


def main():
    device = "cpu"
    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True).to(device)
    model.eval()
    eot = tokenizer.eos_token_id

    true_tokens = [eot] + tokenizer.encode(TEXT)
    bag = true_tokens[1:]
    n_t = len(bag)
    unique = set(bag)
    u = len(unique)
    true_strs = [tokenizer.decode([t]) for t in bag]

    print(f"\nText: {TEXT!r}")
    print(f"Tokens ({n_t}): {true_strs}")
    print(f"Unique: {u}")

    target_emb = get_pooled(model, true_tokens, device)
    target_norm = float(np.linalg.norm(target_emb))
    print(f"Target pooled L12 norm: {target_norm:.2f}")

    # Random baseline L2
    rng = np.random.default_rng(42)
    vocab_size = model.transformer.wte.weight.shape[0]
    rand_l2s = []
    for _ in range(30):
        random_tokens = [eot] + rng.integers(0, vocab_size, size=n_t).tolist()
        r_emb = get_pooled(model, random_tokens, device)
        rand_l2s.append(float(np.linalg.norm(target_emb - r_emb)))
    print(f"Random baseline L2: mean={np.mean(rand_l2s):.2f}, min={np.min(rand_l2s):.2f}")

    # ================================================================
    # Stage 1: Per-position L1 decomposition
    # ================================================================
    print(f"\n{'='*70}")
    print("Stage 1: Per-position L1 decomposition")
    print(f"{'='*70}")

    print("Building L1 vocab...")
    v_l1 = build_vocab_pos1(model, 1, device, eot)

    recovered_bag = per_position_decompose(model, true_tokens, 1, v_l1, device)
    recovered_strs = [tokenizer.decode([t]) for t in recovered_bag]
    pos_match = sum(1 for a, b in zip(recovered_bag, bag) if a == b)
    bag_match = len(set(recovered_bag) & unique)

    print(f"  recovered: {recovered_strs}")
    print(f"  pos-match: {pos_match}/{n_t}")
    print(f"  bag-match: {bag_match}/{u}")
    if pos_match == n_t:
        print(f"  *** PERFECT BAG RECOVERY ***")

    del v_l1

    # ================================================================
    # Stage 2: Bag-constrained beam search
    # ================================================================
    print(f"\n{'='*70}")
    print("Stage 2: Bag-constrained beam search for order recovery")
    print(f"{'='*70}")

    for B in [8, 16, 32]:
        print(f"\n  --- Beam width B={B} ---")
        scored = beam_search_order(model, bag, device, eot, target_emb, beam_width=B)

        best_seq, best_lp, best_l2 = scored[0]
        best_strs = [tokenizer.decode([t]) for t in best_seq]
        best_pos_match = sum(1 for a, b in zip(best_seq, bag) if a == b)

        print(f"  best beam:")
        print(f"    text:  {' '.join(best_strs)}")
        print(f"    pos-match: {best_pos_match}/{n_t}")
        print(f"    L2: {best_l2:.4f}")
        if best_pos_match == n_t:
            print(f"    *** PERFECT ORDER RECOVERY ***")

        # Show top 3
        for i, (seq, lp, l2) in enumerate(scored[:3]):
            strs = [tokenizer.decode([t]) for t in seq]
            pm = sum(1 for a, b in zip(seq, bag) if a == b)
            print(f"    beam {i+1}: pm={pm}/{n_t}  L2={l2:.4f}  {' '.join(strs)}")


if __name__ == "__main__":
    main()
