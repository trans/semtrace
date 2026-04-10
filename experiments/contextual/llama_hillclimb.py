#!/usr/bin/env python3
"""Production attack pipeline on Llama via Ollama:

  Stage 1: Thematic probing — embed ~200 words, find top-20 by cosine
           to target. These are "theme words" (not exact, but topically
           related).
  Stage 2: LLM candidate generation — ask Mistral to generate 20 short
           sentences using the theme words.
  Stage 3: Verification — embed each candidate, find the closest to target.
  Stage 4: Hill-climbing — from the best candidate, try replacing each word
           with alternatives, keep replacements that increase cosine.
  Stage 5: Repeat until convergence or budget exhausted.
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


def generate(prompt, model="mistral:7b", max_tokens=200):
    resp = subprocess.run(
        ['curl', '-s', 'http://localhost:11434/api/generate', '-d',
         json.dumps({'model': model, 'prompt': prompt, 'stream': False,
                     'options': {'temperature': 0.8, 'num_predict': max_tokens}})],
        capture_output=True, text=True)
    return json.loads(resp.stdout)['response'].strip()


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


TARGETS = [
    "the cat sat on the mat",
    "she walked slowly to the river",
]


def main():
    print("Production attack pipeline on Llama 3.2 3B\n")

    vocab_words = list(set('''
    the a an is was are be have has had do does did will would can could
    may might shall should must need not no yes and but or if then so
    in on at to from by with of about for up down out over under into
    I you he she it we they me him her us them my your his its our
    this that these those who what which where when how why
    cat dog sat mat ran park little lamb had go went sure snow bird
    white fleece everywhere love cats dogs big red car drove fast tree
    old man woman fished boat sun rose slowly morning village sleepy
    children played colorful kite water fire king queen house door road
    happy sad run walk truck blue flower moon score seven years ago
    four new good great small put get make take know think want give
    river walked slowly came saw said back still just also too very
    much more most some any all every each both many few long time
    night day book read open hand eye face black green brown young
    city home food work life world people think feel look find help
    '''.split()))

    for target_sent in TARGETS:
        print(f"\n{'='*70}")
        print(f"TARGET: {target_sent!r}")
        print(f"{'='*70}")

        target = embed(target_sent)

        # ============================================================
        # Stage 1: Thematic probing
        # ============================================================
        print(f"\n  Stage 1: Thematic probing ({len(vocab_words)} words)...")
        word_scores = []
        for w in vocab_words:
            e = embed(w)
            word_scores.append((w, cos(e, target)))
        word_scores.sort(key=lambda x: -x[1])

        theme_words = [w for w, _ in word_scores[:20]]
        print(f"  Top 20 theme words: {theme_words}")

        # ============================================================
        # Stage 2: LLM candidate generation
        # ============================================================
        print(f"\n  Stage 2: Generate candidates from theme words...")
        prompt = (
            f"Generate 15 different short English sentences (5-10 words each) "
            f"that are related to these themes/words: {', '.join(theme_words[:10])}.\n"
            f"Output one sentence per line, nothing else.\n\n"
        )
        raw = generate(prompt)
        candidates = [line.strip().strip('0123456789.-) ') for line in raw.split('\n')
                       if line.strip() and len(line.strip()) > 10]
        candidates = candidates[:15]
        # Also add the target itself as a control (would an attacker be this lucky?)
        # candidates.append(target_sent)

        print(f"  Generated {len(candidates)} candidates")
        for c in candidates[:5]:
            print(f"    {c}")
        if len(candidates) > 5:
            print(f"    ...and {len(candidates)-5} more")

        # ============================================================
        # Stage 3: Verification — find the closest candidate
        # ============================================================
        print(f"\n  Stage 3: Verify candidates by cosine to target...")
        scored = []
        for c in candidates:
            e = embed(c)
            s = cos(e, target)
            scored.append((c, s))
        scored.sort(key=lambda x: -x[1])

        print(f"  Top 5 candidates:")
        for c, s in scored[:5]:
            print(f"    cos={s:.4f}  {c}")

        best_sent = scored[0][0]
        best_cos = scored[0][1]
        print(f"\n  Best starting point: cos={best_cos:.4f}")
        print(f"    \"{best_sent}\"")

        # ============================================================
        # Stage 4: Hill-climbing via word replacement
        # ============================================================
        print(f"\n  Stage 4: Hill-climbing (word-by-word replacement)...")
        current = best_sent
        current_cos = best_cos
        current_words = current.split()

        # Replacement candidates for each position: try theme words + common words
        replacement_pool = list(set(theme_words + [
            'the', 'a', 'an', 'on', 'in', 'at', 'to', 'from', 'by', 'with',
            'of', 'and', 'but', 'or', 'is', 'was', 'sat', 'sit', 'ran',
            'walked', 'slowly', 'fast', 'big', 'small', 'old', 'new',
            'cat', 'dog', 'mat', 'rug', 'floor', 'chair', 'table',
            'river', 'park', 'road', 'house', 'boat', 'tree',
            'she', 'he', 'it', 'they', 'we', 'I', 'her', 'his',
            'little', 'very', 'quite', 'slowly', 'quickly',
        ]))

        for round_num in range(5):
            improved = False
            for pos in range(len(current_words)):
                best_replacement = None
                best_new_cos = current_cos

                for replacement in replacement_pool:
                    new_words = list(current_words)
                    new_words[pos] = replacement
                    new_sent = ' '.join(new_words)
                    new_emb = embed(new_sent)
                    new_cos = cos(new_emb, target)

                    if new_cos > best_new_cos:
                        best_new_cos = new_cos
                        best_replacement = replacement

                if best_replacement is not None:
                    old_word = current_words[pos]
                    current_words[pos] = best_replacement
                    current = ' '.join(current_words)
                    current_cos = best_new_cos
                    print(f"    round {round_num+1}: pos {pos} '{old_word}' → "
                          f"'{best_replacement}' → cos={current_cos:.4f}  \"{current}\"")
                    improved = True
                    break  # one improvement per round, then re-assess all positions

            if not improved:
                print(f"    round {round_num+1}: no improvement found")
                break

        print(f"\n  RESULT: cos={current_cos:.4f}")
        print(f"    recovered: \"{current}\"")
        print(f"    true:      \"{target_sent}\"")

        # Word overlap
        true_words = set(target_sent.lower().split())
        rec_words = set(current.lower().split())
        overlap = len(true_words & rec_words)
        print(f"    word overlap: {overlap}/{len(true_words)}")
        if current.lower().strip() == target_sent.lower().strip():
            print(f"    *** PERFECT RECOVERY ***")


if __name__ == "__main__":
    main()
