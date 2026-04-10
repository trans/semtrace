#!/usr/bin/env python3
"""Generate-and-verify attack using Llama's own generate + embed APIs.

The model is both generator and verifier:
  - /api/generate proposes continuations (constrained to plausible English)
  - /api/embed scores each continuation against the target

Pipeline:
  1. Start with an empty or short prefix
  2. Generate N diverse continuations from that prefix (high temperature)
  3. For each, embed the full candidate and score by cosine to target
  4. Keep the top-K best prefixes
  5. Extend each by generating again
  6. Repeat until target length reached
  7. Final ranking by cosine

This is beam search where the beam proposals come from the LLM itself.
Fully black-box: no weights, no hidden states, just two API endpoints.
"""
import json
import subprocess
import numpy as np


def embed(text, model="llama3.2:3b"):
    resp = subprocess.run(
        ['curl', '-s', 'http://localhost:11434/api/embed', '-d',
         json.dumps({'model': model, 'input': text})],
        capture_output=True, text=True)
    return np.array(json.loads(resp.stdout)['embeddings'][0])


def generate_continuation(prefix, model="llama3.2:3b", max_tokens=8, temperature=1.2):
    """Generate a short continuation of the prefix."""
    resp = subprocess.run(
        ['curl', '-s', 'http://localhost:11434/api/generate', '-d',
         json.dumps({'model': model, 'prompt': prefix, 'stream': False,
                     'options': {'temperature': temperature,
                                 'num_predict': max_tokens,
                                 'top_k': 50, 'top_p': 0.95}})],
        capture_output=True, text=True)
    return json.loads(resp.stdout)['response'].strip()


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


TARGETS = [
    "the cat sat on the mat",
    "she walked slowly to the river",
    "Mary had a little lamb",
]


def generate_diverse(prefix, n_samples=8, max_tokens=8, temperatures=[0.8, 1.0, 1.2, 1.5]):
    """Generate diverse continuations by varying temperature."""
    candidates = set()
    for temp in temperatures:
        for _ in range(max(1, n_samples // len(temperatures))):
            cont = generate_continuation(prefix, max_tokens=max_tokens, temperature=temp)
            if cont:
                # Take just the first sentence/clause
                full = (prefix + " " + cont).strip()
                # Truncate at punctuation or newline
                for stop in ['.', '!', '?', '\n', ',']:
                    if stop in cont:
                        cont = cont[:cont.index(stop)]
                        break
                full = (prefix + " " + cont).strip()
                candidates.add(full)
    return list(candidates)


def main():
    print("Generate-and-verify attack on Llama 3.2 3B\n")

    for target_sent in TARGETS:
        print(f"\n{'='*70}")
        print(f"TARGET: {target_sent!r}")
        target = embed(target_sent)
        target_words = target_sent.split()
        target_len = len(target_words)
        print(f"  ({target_len} words)")

        # ============================================================
        # Strategy 1: Word-by-word beam search
        # Generate one word at a time, score full prefix by cosine
        # ============================================================
        print(f"\n  Strategy 1: Word-by-word beam (generate short, score by cosine)")

        beam_width = 6
        beams = [("", 0.0)]  # (prefix, cosine_to_target)

        for word_pos in range(target_len):
            new_beams = []
            tokens_to_generate = 2  # just 1-2 words

            for prefix, _ in beams:
                # Generate diverse short continuations
                continuations = generate_diverse(
                    prefix if prefix else "The",
                    n_samples=8, max_tokens=tokens_to_generate,
                    temperatures=[0.5, 0.8, 1.0, 1.3]
                )

                for full in continuations:
                    # Trim to approximately word_pos+1 words
                    words = full.split()[:word_pos + 2]
                    trimmed = ' '.join(words)
                    e = embed(trimmed)
                    c = cos(e, target)
                    new_beams.append((trimmed, c))

            # Keep top beam_width
            new_beams.sort(key=lambda x: -x[1])
            beams = new_beams[:beam_width]

            best_text, best_cos = beams[0]
            print(f"    word {word_pos+1}: cos={best_cos:.4f}  \"{best_text}\"")

        # Final ranking
        print(f"\n  Final beams:")
        for text, c in beams[:5]:
            overlap = len(set(text.lower().split()) & set(target_sent.lower().split()))
            print(f"    cos={c:.4f}  words={overlap}/{target_len}  \"{text}\"")

        # ============================================================
        # Strategy 2: Full-sentence generation + verification
        # Generate complete sentences, score each
        # ============================================================
        print(f"\n  Strategy 2: Full-sentence generation + cosine verification")

        # Use the best beam prefix as a seed for full-sentence generation
        seed = beams[0][0].split()[0] if beams[0][0] else "The"

        full_candidates = set()
        for temp in [0.5, 0.7, 0.9, 1.0, 1.2, 1.4]:
            for _ in range(3):
                cont = generate_continuation(seed, max_tokens=target_len * 2, temperature=temp)
                full = (seed + " " + cont).strip()
                # Trim to ~target_len words
                words = full.split()[:target_len + 2]
                full_candidates.add(' '.join(words))

        # Also try generating from the top-3 beam prefixes
        for prefix, _ in beams[:3]:
            remaining = target_len - len(prefix.split())
            if remaining > 0:
                for temp in [0.5, 0.8, 1.0, 1.3]:
                    cont = generate_continuation(prefix, max_tokens=remaining * 2, temperature=temp)
                    full = (prefix + " " + cont).strip()
                    words = full.split()[:target_len + 1]
                    full_candidates.add(' '.join(words))

        # Score all candidates
        scored = []
        for c in full_candidates:
            e = embed(c)
            s = cos(e, target)
            scored.append((c, s))
        scored.sort(key=lambda x: -x[1])

        print(f"  {len(full_candidates)} candidates scored")
        print(f"  Top 5:")
        for text, c in scored[:5]:
            overlap = len(set(text.lower().split()) & set(target_sent.lower().split()))
            print(f"    cos={c:.4f}  words={overlap}/{target_len}  \"{text}\"")

        # Overall best across both strategies
        all_results = beams + scored
        all_results.sort(key=lambda x: -x[1])
        best_text, best_cos = all_results[0]
        overlap = len(set(best_text.lower().split()) & set(target_sent.lower().split()))

        print(f"\n  BEST OVERALL: cos={best_cos:.4f}  words={overlap}/{target_len}")
        print(f"    recovered: \"{best_text}\"")
        print(f"    true:      \"{target_sent}\"")


if __name__ == "__main__":
    main()
