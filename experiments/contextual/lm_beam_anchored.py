#!/usr/bin/env python3
"""LM-guided beam search starting from a partial anchor.

The triangulation idea, executed:
  1. We have a partial anchor: the first K positions are known to be correct
     (from the L1 per-position decomposition or any other anchor source).
  2. For the remaining N-K positions, generate token-by-token using GPT-2's
     own LM probabilities to propose candidates (constraining the search to
     plausible continuations of the anchor).
  3. At each generation step, score each beam by forward-passing the
     partial sequence and computing embedding distance to the target.
  4. Keep the top beam_width beams.
  5. Pick the best by final embedding distance.

The LM's proposals constrain the search to the "manifold of plausible English
text" (the islands, not the ocean). The embedding distance ranks the proposals
to find which plausible continuation matches the target.

We test multiple anchor sizes (0, 1, 2, 3 known prefix tokens) and sentences.
"""
import argparse
import gc
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer


SENTENCES = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "Mary had a little lamb",
    "Four score and seven years ago",
    "I love cats and dogs",
    "the big red car drove fast",
]


def get_pooled_embedding(model, tokens, device):
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


def beam_search_anchored(
    model, target_emb, prefix_tokens, n_total, device, eot,
    beam_width=8, lm_top_k=20
):
    """Beam search starting from a fixed prefix.

    At each step:
      - For each beam (a partial sequence), get the LM's top lm_top_k
        candidate next tokens
      - Extend each beam with each candidate
      - Forward-pass each extended beam to compute its current embedding
      - Compute distance to target as the score
      - Keep top beam_width beams by distance

    Return the final beam_width beams sorted by final distance.
    """
    n_to_generate = n_total - len(prefix_tokens)
    if n_to_generate <= 0:
        # Nothing to generate
        full = [eot] + list(prefix_tokens)
        emb = get_pooled_embedding(model, full, device)
        d = float(np.sum((target_emb - emb) ** 2))
        return [(list(prefix_tokens), d)]

    beams = [(list(prefix_tokens), 0.0)]  # (sequence, log_prob_so_far)

    for step in range(n_to_generate):
        # For each current beam, get LM next-token candidates
        candidates = []
        for seq, lp in beams:
            full = [eot] + seq
            with torch.no_grad():
                out = model(torch.tensor([full]).to(device))
            logits = out.logits.squeeze(0)[-1]
            log_probs = F.log_softmax(logits, dim=-1)
            top_vals, top_ids = torch.topk(log_probs, k=lm_top_k)
            for i in range(lm_top_k):
                new_seq = seq + [int(top_ids[i])]
                new_lp = lp + float(top_vals[i])
                candidates.append((new_seq, new_lp))

        # Score each candidate by embedding distance
        scored = []
        for seq, lp in candidates:
            full = [eot] + seq
            emb = get_pooled_embedding(model, full, device)
            d = float(np.sum((target_emb - emb) ** 2))
            scored.append((seq, lp, d))

        # Keep top beam_width by distance
        scored.sort(key=lambda x: x[2])
        beams = [(s, lp) for s, lp, _ in scored[:beam_width]]

    # Final ranking by distance
    final = []
    for seq, lp in beams:
        full = [eot] + seq
        emb = get_pooled_embedding(model, full, device)
        d = float(np.sum((target_emb - emb) ** 2))
        final.append((seq, d))
    final.sort(key=lambda x: x[1])
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--lm-top-k", type=int, default=20)
    parser.add_argument("--anchor-sizes", default="0,1,2,3")
    args = parser.parse_args()

    anchor_sizes = [int(x) for x in args.anchor_sizes.split(",")]

    device = "cpu"
    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True).to(device)
    model.eval()
    eot = tokenizer.eos_token_id

    print("Building L1 vocab (for per-position anchor)...")
    v_l1 = build_vocab_pos1(model, 1, device, eot)

    summary = {}  # (anchor_size, sentence) -> pos_match

    for sent in SENTENCES:
        print(f"\n{'='*70}")
        print(f"Sentence: {sent!r}")
        true_tokens = [eot] + tokenizer.encode(sent)
        true_trailing = true_tokens[1:]
        n_t = len(true_trailing)
        true_strs = [tokenizer.decode([t]) for t in true_trailing]
        target_emb = get_pooled_embedding(model, true_tokens, device)

        # Get the perfect L1 anchor
        l1_anchor = per_position_decompose(model, true_tokens, 1, v_l1, device)
        # Verify it's correct
        l1_correct = sum(1 for a, b in zip(l1_anchor, true_trailing) if a == b)
        print(f"  L1 per-position recovers {l1_correct}/{n_t} correctly")

        for anchor_size in anchor_sizes:
            if anchor_size > n_t:
                continue
            prefix = list(l1_anchor[:anchor_size])
            print(f"\n  --- Anchor size {anchor_size}: prefix = {[tokenizer.decode([t]) for t in prefix]} ---")

            results = beam_search_anchored(
                model, target_emb, prefix, n_t, device, eot,
                beam_width=args.beam_width, lm_top_k=args.lm_top_k,
            )

            best_seq, best_dist = results[0]
            best_strs = [tokenizer.decode([t]) for t in best_seq]
            pos_match = sum(1 for a, b in zip(best_seq, true_trailing) if a == b)
            print(f"     best beam: {best_strs}")
            print(f"     pos-match: {pos_match}/{n_t}, dist: {best_dist:.4f}")
            if pos_match == n_t:
                print(f"     *** PERFECT ***")

            # Show top 3 beams for context
            for i, (seq, d) in enumerate(results[:3]):
                tok_str = [tokenizer.decode([t]) for t in seq]
                pm = sum(1 for a, b in zip(seq, true_trailing) if a == b)
                marker = "*" if i == 0 else " "
                print(f"     {marker} beam {i+1}: {tok_str}  match={pm}/{n_t}  dist={d:.4f}")

            summary[(anchor_size, sent)] = pos_match

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY: pos-match per anchor size")
    print(f"{'='*70}")
    header = f"  {'sentence':40s}"
    for sz in anchor_sizes:
        header += f"  anc={sz}"
    print(header)
    for sent in SENTENCES:
        line = f"  {sent[:38]:40s}"
        true_n = len(tokenizer.encode(sent))
        for sz in anchor_sizes:
            if sz > true_n:
                line += "  n/a"
            else:
                pm = summary.get((sz, sent), 0)
                line += f"  {pm}/{true_n}"
        print(line)

    del v_l1
    gc.collect()


if __name__ == "__main__":
    main()
