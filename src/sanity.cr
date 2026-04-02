require "./semtrace"

# Sanity Check Benchmark
#
# Tests basic decomposition correctness:
#   - Single token round-trips
#   - Pair composition
#   - Concept arithmetic
#
# Usage:
#   bin/sanity [--data DIR]

module Semtrace
  module SanityBench
    SINGLES = %w[cat dog the hello world man woman king]

    PAIRS = [
      ["cat", "dog"],
      ["man", "woman"],
      ["king", "queen"],
    ]

    ARITHMETIC = [
      {positive: [" king", " woman"], negative: [" man"], label: "king - man + woman"},
      {positive: [" Paris", " Germany"], negative: [" France"], label: "Paris - France + Germany"},
      {positive: [" bigger", " cold"], negative: [" big"], label: "bigger - big + cold"},
    ]

    def self.run
      data_dir = (Path[__DIR__].parent / "data").to_s

      i = 0
      while i < ARGV.size
        case ARGV[i]
        when "--data" then data_dir = ARGV[i + 1]; i += 2
        else i += 1
        end
      end

      embeddings_path = File.join(data_dir, "embeddings.bin")
      vocab_path = File.join(data_dir, "vocab.json")
      abort "Missing data files in #{data_dir}" unless File.exists?(embeddings_path) && File.exists?(vocab_path)

      model_name = Path[data_dir].basename
      print "Loading #{model_name}... "
      store = EmbeddingStore.new(embeddings_path, vocab_path)
      puts "#{store.vocab_size} tokens x #{store.dimensions}d"
      decomposer = Decomposer.new(store)

      # Single token round-trips
      puts "\n--- Single Token Round-Trip ---"
      ok = 0
      total = 0
      SINGLES.each do |token|
        [token, " #{token}"].each do |t|
          next unless store.vocab.includes?(t)
          total += 1
          result = decomposer.trace_single(t)
          if result.tokens == [t] && result.final_residual_norm < 0.001
            ok += 1
          else
            puts "  FAIL: #{t.inspect} -> #{result.tokens.map(&.inspect).join(", ")} (residual: #{"%.4f" % result.final_residual_norm})"
          end
        end
      end
      puts "  #{ok}/#{total} exact round-trips"

      # Pair composition
      puts "\n--- Pair Composition ---"
      pair_ok = 0
      pair_total = 0
      PAIRS.each do |pair|
        tokens = pair.map { |t| store.vocab.includes?(" #{t}") ? " #{t}" : t }
        next unless tokens.all? { |t| store.vocab.includes?(t) }
        pair_total += 1
        begin
          result = decomposer.trace_tokens(tokens)
          if result.tokens.sort == tokens.sort && result.final_residual_norm < 0.001
            pair_ok += 1
          else
            puts "  FAIL: #{tokens.map(&.inspect).join(" + ")} -> #{result.tokens.map(&.inspect).join(", ")}"
          end
        rescue e
          puts "  SKIP: #{tokens.map(&.inspect).join(" + ")} — #{e.message}"
        end
      end
      puts "  #{pair_ok}/#{pair_total} exact pair decompositions"

      # Concept arithmetic
      puts "\n--- Concept Arithmetic ---"
      ARITHMETIC.each do |ex|
        begin
          result = decomposer.arithmetic(positive: ex[:positive], negative: ex[:negative])
          puts "  #{ex[:label]} -> #{result.tokens.map(&.inspect).join(", ")} (residual: #{"%.4f" % result.final_residual_norm})"
        rescue e
          puts "  #{ex[:label]}: SKIP — #{e.message}"
        end
      end

      store.close
    end
  end
end

Semtrace::SanityBench.run
