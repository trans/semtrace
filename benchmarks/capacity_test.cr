require "../src/semtrace"

# Capacity Test
#
# Measures the maximum number of unique tokens recoverable from a
# bag-of-words sum in a given embedding space. Isolates the pure
# geometric capacity question by using well-trained common tokens.
#
# Usage:
#   bin/capacity_test [--data DIR] [--trials N] [--max N]

module Semtrace
  module CapacityTest
    def self.run
      data_dir = (Path[__DIR__].parent / "data").to_s
      trials = 5
      max_n = 500
      test_ns = [10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500]

      i = 0
      while i < ARGV.size
        case ARGV[i]
        when "--data"   then data_dir = ARGV[i + 1]; i += 2
        when "--trials" then trials = ARGV[i + 1].to_i; i += 2
        when "--max"    then max_n = ARGV[i + 1].to_i; i += 2
        else i += 1
        end
      end

      test_ns = test_ns.select { |n| n <= max_n }

      embeddings_path = File.join(data_dir, "embeddings.bin")
      vocab_path = File.join(data_dir, "vocab.json")
      abort "Missing data files in #{data_dir}" unless File.exists?(embeddings_path) && File.exists?(vocab_path)

      model_name = Path[data_dir].basename
      print "Loading #{model_name}... "
      store = EmbeddingStore.new(embeddings_path, vocab_path)
      puts "#{store.vocab_size} tokens x #{store.dimensions}d"
      decomposer = Decomposer.new(store)

      # Select well-trained tokens: filter by norm (not too small, not too large)
      # and skip the first ~256 tokens (control chars, byte tokens)
      puts "Selecting well-trained tokens..."
      norms = [] of {Int32, Float32}
      (256...store.vocab_size).each do |i|
        n = EmbeddingStore.norm(store.vector_for(i))
        norms << {i, n}
      end

      # Use tokens with norms in the middle 80% — not outliers
      norms.sort_by! { |_, n| n }
      lo = norms.size // 10
      hi = norms.size * 9 // 10
      good_tokens = norms[lo...hi].map { |id, _| id }

      avg_norm = good_tokens.sum { |id| EmbeddingStore.norm(store.vector_for(id)) } / good_tokens.size
      puts "  Pool: #{good_tokens.size} tokens (norms in middle 80%, avg: #{"%.3f" % avg_norm})"

      puts "\n=== Capacity Test: #{store.dimensions}d ==="
      puts "  Trials: #{trials}"
      puts "  Search: HNSW (cosine, f16)"
      puts

      puts "#{"N".rjust(5)}  #{"Exact".rjust(8)}  #{"Sem<0.5".rjust(8)}  #{"Sem<0.7".rjust(8)}  #{"Recov".rjust(6)}"
      puts "-" * 42

      test_ns.each do |n|
        next if n > good_tokens.size

        sum_exact = 0
        sum_sem5 = 0
        sum_sem7 = 0
        sum_recovered = 0

        trials.times do |trial|
          rng = Random.new((n.to_u64 * 1000 + trial.to_u64))
          token_ids = good_tokens.sample(n, rng)
          tid_set = token_ids.to_set

          target = Array(Float32).new(store.dimensions, 0.0_f32)
          token_ids.each { |id| target = EmbeddingStore.add(target, store.vector_for(id)) }

          result = decomposer.decompose(target, max_steps: n + 15)
          recovered_set = result.token_ids.to_set
          sum_recovered += result.tokens.size

          # Exact matches
          exact = (tid_set & recovered_set).size
          sum_exact += exact

          # Semantic matches for missing tokens
          missing = tid_set - recovered_set
          missing.each do |mid|
            best_dist = Float32::MAX
            recovered_set.each do |rid|
              dist = USearch::Index.distance(
                store.vector_for(mid).to_a,
                store.vector_for(rid).to_a,
                :cos
              )
              best_dist = dist if dist < best_dist
            end
            sum_sem5 += 1 if best_dist < 0.5
            sum_sem7 += 1 if best_dist < 0.7
          end
        end

        total = n * trials
        pct_exact = sum_exact * 100.0 / total
        pct_sem5 = (sum_exact + sum_sem5) * 100.0 / total
        pct_sem7 = (sum_exact + sum_sem7) * 100.0 / total
        avg_recov = sum_recovered // trials

        puts "#{n.to_s.rjust(5)}  #{("%.1f%%" % pct_exact).rjust(8)}  #{("%.1f%%" % pct_sem5).rjust(8)}  #{("%.1f%%" % pct_sem7).rjust(8)}  #{avg_recov.to_s.rjust(6)}"
      end

      store.close
    end
  end
end

Semtrace::CapacityTest.run
