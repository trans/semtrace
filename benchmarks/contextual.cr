require "../src/semtrace"

# Contextual Embedding Benchmark
#
# Tests decomposition of contextual (forward-pass) embeddings from Ollama
# against the static token embedding matrix from the same model.
#
# Requires Ollama running with the target model loaded.
#
# Usage:
#   bin/contextual [--data DIR] [--model llama3.2:3b] [--text "sentence"]

module Semtrace
  module ContextualBench
    def self.run
      data_dir = (Path[__DIR__].parent / "data" / "llama-3-2-3b-instruct").to_s
      model = "llama3.2:3b"
      texts = [] of String

      i = 0
      while i < ARGV.size
        case ARGV[i]
        when "--data"  then data_dir = ARGV[i + 1]; i += 2
        when "--model" then model = ARGV[i + 1]; i += 2
        when "--text"  then texts << ARGV[i + 1]; i += 2
        else i += 1
        end
      end

      if texts.empty?
        texts = [
          "the cat sat on the mat",
          "hello world",
          "The king and the queen",
          "I like dogs and cats",
        ]
      end

      embeddings_path = File.join(data_dir, "embeddings.bin")
      vocab_path = File.join(data_dir, "vocab.json")
      abort "Missing data files in #{data_dir}" unless File.exists?(embeddings_path) && File.exists?(vocab_path)

      model_name = Path[data_dir].basename
      print "Loading #{model_name} (brute-force mode)... "
      store = EmbeddingStore.new(embeddings_path, vocab_path, skip_index: true)
      puts "#{store.vocab_size} tokens x #{store.dimensions}d"
      decomposer = Decomposer.new(store)

      texts.each do |text|
        puts "\n=== \"#{text}\" ==="

        # Get contextual embedding
        print "  Ollama embed... "
        ctx = Ollama.embed(text, model)
        puts "#{ctx.size}d (norm: #{"%.4f" % EmbeddingStore.norm(ctx)})"

        if ctx.size != store.dimensions
          puts "  ERROR: Dimension mismatch"
          next
        end

        # Nearest static tokens
        puts "  Nearest static tokens:"
        results = store.search(ctx, k: 10)
        results.each_with_index do |r, j|
          puts "    #{(j+1).to_s.rjust(2)}. #{store.token_for(r.key).inspect.ljust(15)} (dist: #{"%.4f" % r.distance})"
        end

        # Greedy decomposition
        result = decomposer.decompose(ctx, max_steps: 20)
        puts "  Decomposed (#{result.tokens.size}): #{result.tokens.map(&.inspect).join(", ")}"
        puts "  Final residual: #{"%.4f" % result.final_residual_norm}"
      end

      store.close
    end
  end
end

Semtrace::ContextualBench.run
