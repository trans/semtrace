require "../src/semtrace"

# Battery Benchmark
#
# Three fixed test batteries for cross-model comparison:
#   A: Simple vocabulary (common single-token words)
#   B: Complex vocabulary (mixed common + multi-syllable)
#   C: Multi-sentence with punctuation
#
# Requires a BPE tokenizer (merges.txt) — GPT-2 models only.
#
# Usage:
#   bin/batteries [--data DIR] [--battery A|B|C|all]

module Semtrace
  module BatteryBench
    SIMPLE = [
      "the cat sat on the mat",
      "the dog and the cat are friends",
      "the big red dog chased the small black cat across the yard",
      "the old man walked down the long and dusty road that led to the small town by the river",
      "the young girl and her older brother went to the park near the lake and they played with the dog and the cat and then they all went home and had dinner together with the rest of the family",
      "in the morning the sun rose over the hills and the birds began to sing in the trees and the wind blew through the fields of green grass and the farmer went out to the barn to feed the animals and milk the cows before he came back to the house for a hot cup of tea",
      "the big brown dog and the small black cat ran fast down the long dirt road to the old farm where the man and the woman lived with the birds and the fish and they all had food and water and the sun was warm and the sky was blue and the grass was green and the wind was cool and the trees were tall and the flowers were red and the air was fresh and the day was good and the night was dark and the moon was full",
      "the big brown dog and the small black cat ran fast down the long dirt road to the old farm where the man and the woman lived with the birds and the fish and they all had food and water and the sun was warm and the sky was blue and the grass was green and the wind was cool and the trees were tall and the flowers were red and the air was fresh and the day was good and the night was dark and the moon was full and the stars were bright and the world was still and the fire was hot and the snow was white and the rain was cold and the ice was hard and the stone was gray and the sand was soft and the sea was deep and the land was wide",
    ]

    COMPLEX = [
      "The old building was undergoing a major renovation.",
      "She carefully reviewed the environmental sustainability report before the meeting.",
      "The new semiconductor chip was surprisingly efficient but the manufacturing process needed improvement.",
      "After the investigation, the researchers published their extraordinary findings about developmental psychology in a well-known international journal last year.",
      "The archaeological team from the university spent three long summers at the remote site before they finally uncovered the sophisticated agricultural tools that changed our understanding of Mediterranean civilizations in the seventeenth century.",
    ]

    MULTI = [
      "Hello, world! How are you?",
      "The cat sat down. It was tired. The dog, however, was not.",
      "She said: \"I don't believe it.\" He replied, \"Well, it's true.\"",
      "First, we went to the store; then, we stopped at the park. Finally, we came home -- exhausted but happy -- and made dinner.",
      "Dr. Smith (the lead researcher) published her findings in Nature. The results, which contradicted previous studies, were surprising. \"We didn't expect this,\" she admitted. However, the data was clear: the hypothesis was wrong. Period.",
    ]

    def self.run
      data_dir = (Path[__DIR__].parent / "data").to_s
      battery = "all"

      i = 0
      while i < ARGV.size
        case ARGV[i]
        when "--data"    then data_dir = ARGV[i + 1]; i += 2
        when "--battery" then battery = ARGV[i + 1]; i += 2
        else i += 1
        end
      end

      embeddings_path = File.join(data_dir, "embeddings.bin")
      vocab_path = File.join(data_dir, "vocab.json")
      merges_path = File.join(data_dir, "merges.txt")
      abort "Missing data files in #{data_dir}" unless File.exists?(embeddings_path) && File.exists?(vocab_path)
      abort "No merges.txt — this benchmark requires a BPE tokenizer (GPT-2 models)" unless File.exists?(merges_path)

      model_name = Path[data_dir].basename
      print "Loading #{model_name}... "
      store = EmbeddingStore.new(embeddings_path, vocab_path)
      puts "#{store.vocab_size} tokens x #{store.dimensions}d"
      decomposer = Decomposer.new(store)
      tokenizer = Tokenizer.new(vocab_path, merges_path)

      batteries = case battery.downcase
                  when "a" then [{"A: Simple Vocabulary", SIMPLE}]
                  when "b" then [{"B: Complex Vocabulary", COMPLEX}]
                  when "c" then [{"C: Multi-Sentence & Punctuation", MULTI}]
                  else
                    [
                      {"A: Simple Vocabulary", SIMPLE},
                      {"B: Complex Vocabulary", COMPLEX},
                      {"C: Multi-Sentence & Punctuation", MULTI},
                    ]
                  end

      batteries.each do |label, sentences|
        puts "\n--- #{label} ---"
        puts "#{"N".rjust(5)}  #{"Accuracy".rjust(8)}  #{"Miss".rjust(5)}  #{"Extra".rjust(5)}  Sentence"
        puts "-" * 90

        sentences.each do |text|
          ids = tokenizer.encode(text)
          n = ids.size
          original = ids.map { |id| store.token_for(id) }.sort

          result = decomposer.trace_sentence(ids, max_steps: n + 10)
          recovered = result.tokens.sort

          if original == recovered
            puts "#{n.to_s.rjust(5)}  #{"100.0%".rjust(8)}  #{"0".rjust(5)}  #{"0".rjust(5)}  #{text[0, 60]}"
          else
            missing = (original - recovered).size
            extra = (recovered - original).size
            pct = "%.1f%%" % ((n - missing) * 100.0 / n)
            puts "#{n.to_s.rjust(5)}  #{pct.rjust(8)}  #{missing.to_s.rjust(5)}  #{extra.to_s.rjust(5)}  #{text[0, 60]}"
          end
        end
      end

      store.close
    end
  end
end

Semtrace::BatteryBench.run
