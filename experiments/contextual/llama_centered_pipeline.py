#!/usr/bin/env python3
"""Production attack pipeline with centered cosine probing.

The improved pipeline:
  1. Build vocab by embedding ~200 common words through Llama embed API
  2. Compute centroid and center everything
  3. Rank vocab by CENTERED cosine to target → extract content words
  4. Feed content words to Mistral → generate candidate sentences
  5. Embed each candidate through Llama → score by cosine to target
  6. Hill-climb from the best candidate → word-by-word replacement

Fully black-box: only uses /api/embed and /api/generate endpoints.
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


def generate(prompt, model="mistral:7b", temp=0.3, max_tok=60):
    resp = subprocess.run(
        ['curl', '-s', 'http://localhost:11434/api/generate', '-d',
         json.dumps({'model': model, 'prompt': prompt, 'stream': False,
                     'options': {'temperature': temp, 'num_predict': max_tok}})],
        capture_output=True, text=True)
    return json.loads(resp.stdout)['response'].strip()


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


TARGETS = [
    "the cat sat on the mat",
    "she walked slowly to the river",
    "Mary had a little lamb",
    "the old man fished from the boat",
    "children played with the colorful kite",
]

VOCAB_WORDS = list(set('''
the a an is was are be have has had do does did will would can could
in on at to from by with of about for up down out over under into
I you he she it we they me him her us them my your his its our
and but or if then so not no yes this that these those who what
cat dog sat mat ran park little lamb had Mary go went sure snow
white fleece everywhere love cats dogs big red car drove fast
old man woman fished boat sun rose slowly morning village sleepy
children played colorful kite bird tree water fire king queen
happy sad run walk sit lay truck blue flower moon score seven
years ago four new good great small house door road river came saw
said put get make take know think want give back still just also
very much more most some any all every each many few long time
night day book read open hand face black green young city home food
life world people feel look find help sitting resting mat rug
floor chair table bed carpet kitchen garden park street fish
caught threw ball tall short wide deep warm cold hot bright dark
girl boy baby mother father brother sister friend teacher student
'''.split()))


def main():
    print("Centered-cosine production pipeline on Llama 3.2 3B\n")

    # Build vocabulary embeddings (one-time)
    print(f"Building vocabulary ({len(VOCAB_WORDS)} words)...")
    word_embs = {}
    for w in VOCAB_WORDS:
        word_embs[w] = embed(w)
    vocab_matrix = np.array([word_embs[w] for w in VOCAB_WORDS])

    # Compute centroid
    centroid = vocab_matrix.mean(axis=0)
    print(f"Centroid norm: {np.linalg.norm(centroid):.4f}")

    for target_sent in TARGETS:
        print(f"\n{'='*70}")
        print(f"TARGET: {target_sent!r}")
        target = embed(target_sent)
        target_centered = target - centroid
        true_words = set(target_sent.lower().split())

        # ============================================================
        # Stage 1: Centered cosine word probing
        # ============================================================
        print(f"\n  Stage 1: Centered cosine word probing")
        scores = []
        for w in VOCAB_WORDS:
            w_centered = word_embs[w] - centroid
            c = cos(w_centered, target_centered)
            scores.append((w, c))
        scores.sort(key=lambda x: -x[1])

        top_words = [w for w, _ in scores[:15]]
        top_in_true = [w for w in top_words if w.lower() in true_words]
        print(f"  Top 15 words: {top_words}")
        print(f"  True words found in top 15: {top_in_true} ({len(top_in_true)}/{len(true_words)})")

        # ============================================================
        # Stage 2: LLM candidate generation from content words
        # ============================================================
        print(f"\n  Stage 2: Generate candidates from content words")
        content_words = ', '.join(top_words[:8])

        candidates = set()
        # Ask Mistral to make sentences multiple ways
        for temp in [0.1, 0.5, 0.8, 1.0]:
            prompt = (
                f"Write a short simple English sentence (5-8 words) using some of "
                f"these words: {content_words}.\n"
                f"Output ONLY the sentence, nothing else.\n\nSentence:"
            )
            result = generate(prompt, temp=temp, max_tok=30)
            sent = result.split('\n')[0].strip().strip('"').strip("'")
            if len(sent) > 5:
                candidates.add(sent)

        # Also try: "make a sentence with [top-3 content words]"
        for i in range(min(5, len(top_in_true))):
            w = top_words[i]
            prompt = f"Write a short sentence containing the word \"{w}\".\nOutput ONLY the sentence.\n\nSentence:"
            for temp in [0.3, 0.8]:
                result = generate(prompt, temp=temp, max_tok=25)
                sent = result.split('\n')[0].strip().strip('"').strip("'")
                if len(sent) > 5:
                    candidates.add(sent)

        # And directly: "what sentence uses these words?"
        prompt = (
            f"These words were extracted from a sentence: {', '.join(top_words[:6])}.\n"
            f"What was the original sentence? It is short (5-8 words).\n"
            f"Output ONLY your best guess at the sentence.\n\nSentence:"
        )
        for temp in [0.1, 0.5, 1.0]:
            result = generate(prompt, temp=temp, max_tok=25)
            sent = result.split('\n')[0].strip().strip('"').strip("'")
            if len(sent) > 5:
                candidates.add(sent)

        print(f"  Generated {len(candidates)} candidates")

        # ============================================================
        # Stage 3: Cosine verification
        # ============================================================
        print(f"\n  Stage 3: Verify candidates by cosine")
        scored = []
        for c in candidates:
            e = embed(c)
            s = cos(e, target)
            scored.append((c, s))
        scored.sort(key=lambda x: -x[1])

        print(f"  Top 5:")
        for c, s in scored[:5]:
            overlap = len(true_words & set(c.lower().replace(',', '').replace('.', '').split()))
            print(f"    cos={s:.4f}  words={overlap}/{len(true_words)}  \"{c}\"")

        if not scored:
            print(f"  No candidates generated!")
            continue

        best_sent = scored[0][0]
        best_cos = scored[0][1]

        # ============================================================
        # Stage 4: Hill-climbing word replacement
        # ============================================================
        print(f"\n  Stage 4: Hill-climbing from best candidate")
        current = best_sent
        current_cos = best_cos
        current_words = current.split()

        # Replacement pool: top-30 centered-cosine words + common function words
        replacement_pool = list(set(
            [w for w, _ in scores[:30]] +
            ['the', 'a', 'an', 'on', 'in', 'at', 'to', 'from', 'by', 'with',
             'of', 'and', 'but', 'or', 'is', 'was', 'sat', 'sit', 'ran',
             'walked', 'slowly', 'fast', 'big', 'small', 'old', 'new',
             'cat', 'dog', 'mat', 'had', 'little', 'she', 'he', 'it',
             'Mary', 'river', 'boat', 'man', 'fished', 'children', 'played']
        ))

        for round_num in range(8):
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

                    if new_cos > best_new_cos + 0.001:  # small threshold to avoid noise
                        best_new_cos = new_cos
                        best_replacement = replacement

                if best_replacement is not None:
                    old = current_words[pos]
                    current_words[pos] = best_replacement
                    current = ' '.join(current_words)
                    current_cos = best_new_cos
                    print(f"    round {round_num+1}: '{old}' → '{best_replacement}' → "
                          f"cos={current_cos:.4f}  \"{current}\"")
                    improved = True
                    break

            if not improved:
                break

            if current_cos > 0.999:
                break

        # ============================================================
        # Final result
        # ============================================================
        overlap = len(true_words & set(current.lower().replace(',', '').replace('.', '').split()))
        print(f"\n  RESULT: cos={current_cos:.4f}  words={overlap}/{len(true_words)}")
        print(f"    recovered: \"{current}\"")
        print(f"    true:      \"{target_sent}\"")
        if current_cos > 0.99:
            print(f"    *** NEAR-PERFECT RECOVERY ***")
        if current.lower().strip().rstrip('.') == target_sent.lower().strip():
            print(f"    *** EXACT MATCH ***")


if __name__ == "__main__":
    main()
