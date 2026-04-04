require "../../src/semtrace"

# Capacity Test
#
# Measures unique token recovery at increasing N for multiple modes:
#   - Raw: sum raw vectors, search raw index
#   - NormVecs: sum normalized vectors, search raw index (cosine is scale-invariant)
#   - NormSum: sum raw vectors then normalize the sum, search raw index
#   - NormAll: sum normalized vectors then normalize the sum, search raw index
#
# Uses well-trained tokens from the middle 80% of the norm distribution.
#
# Usage:
#   bin/capacity_test [--data DIR] [--trials N] [--max N]

module Semtrace
  module CapacityTest
    def self.run
      data_dir = (Path[__DIR__].parent.parent / "data").to_s
      trials = 5
      test_ns = [10, 25, 50, 75, 100, 150, 200, 300, 500]

      i = 0
      while i < ARGV.size
        case ARGV[i]
        when "--data"   then data_dir = ARGV[i + 1]; i += 2
        when "--trials" then trials = ARGV[i + 1].to_i; i += 2
        when "--max"    then max_n = ARGV[i + 1].to_i; test_ns = test_ns.select { |n| n <= max_n }; i += 2
        when "--ns"     then test_ns = ARGV[i + 1].split(",").map(&.to_i); i += 2
        else i += 1
        end
      end

      embeddings_path = File.join(data_dir, "embeddings.bin")
      vocab_path = File.join(data_dir, "vocab.json")
      abort "Missing data files in #{data_dir}" unless File.exists?(embeddings_path) && File.exists?(vocab_path)

      model_name = Path[data_dir].basename
      print "Loading #{model_name} (with normalized index)... "
      store = EmbeddingStore.new(embeddings_path, vocab_path, build_norm_index: true)
      puts "#{store.vocab_size} tokens x #{store.dimensions}d"
      decomposer = Decomposer.new(store)
      dims = store.dimensions

      # Select well-trained tokens
      puts "Selecting well-trained tokens..."
      norms = [] of {Int32, Float32}
      (256...store.vocab_size).each do |i|
        n = EmbeddingStore.norm(store.vector_for(i))
        norms << {i, n}
      end
      norms.sort_by! { |_, n| n }
      lo = norms.size // 10
      hi = norms.size * 9 // 10
      good_tokens = norms[lo...hi].map { |id, _| id }
      avg_norm = good_tokens[0, 100].sum { |id| EmbeddingStore.norm(store.vector_for(id)) } / 100
      puts "  Pool: #{good_tokens.size} tokens (avg norm: #{"%.3f" % avg_norm})"

      puts "\n=== Capacity Test: #{dims}d, #{store.vocab_size} vocab ==="
      puts "  Trials: #{trials}"
      puts "  Search: HNSW (cosine, f16)"
      puts
      puts "#{"N".rjust(5)}  #{"Raw".rjust(8)}  #{"NrmVec".rjust(8)}  #{"NrmSum".rjust(8)}  #{"NrmAll".rjust(8)}"
      puts "-" * 43

      test_ns.each do |n|
        next if n > good_tokens.size

        sum_raw = 0
        sum_nv = 0
        sum_ns = 0
        sum_na = 0

        trials.times do |trial|
          rng = Random.new((n.to_u64 * 1000 + trial.to_u64))
          token_ids = good_tokens.sample(n, rng)
          tid_set = token_ids.to_set

          # Raw target: sum of raw vectors
          target_raw = Array(Float32).new(dims, 0.0_f32)
          token_ids.each { |id| target_raw = EmbeddingStore.add(target_raw, store.vector_for(id)) }

          # NormVecs target: sum of normalized vectors
          target_nv = Array(Float32).new(dims, 0.0_f32)
          token_ids.each do |id|
            vec = store.vector_for(id).to_a
            vec_norm = EmbeddingStore.norm(vec)
            target_nv = EmbeddingStore.add(target_nv, vec.map { |v| v / vec_norm }) if vec_norm > 1e-10
          end

          # NormSum target: sum raw vectors, then normalize the sum
          raw_norm = EmbeddingStore.norm(target_raw)
          target_ns = raw_norm > 1e-10 ? target_raw.map { |v| v / raw_norm } : target_raw

          # NormAll target: sum normalized vectors, then normalize the sum
          nv_norm = EmbeddingStore.norm(target_nv)
          target_na = nv_norm > 1e-10 ? target_nv.map { |v| v / nv_norm } : target_nv

          # Raw: raw index, raw subtraction
          r_raw = decomposer.decompose(target_raw, max_steps: n + 15)
          sum_raw += (tid_set & r_raw.token_ids.to_set).size

          # NormVecs: normalized index, normalized subtraction
          r_nv = decomposer.decompose_normalized(target_nv, max_steps: n + 15)
          sum_nv += (tid_set & r_nv.token_ids.to_set).size

          # NormSum: raw sum normalized — search raw index, raw subtraction
          r_ns = decomposer.decompose(target_ns, max_steps: n + 15)
          sum_ns += (tid_set & r_ns.token_ids.to_set).size

          # NormAll: normalized sum normalized — search norm index, norm subtraction
          r_na = decomposer.decompose_normalized(target_na, max_steps: n + 15)
          sum_na += (tid_set & r_na.token_ids.to_set).size
        end

        total = n * trials
        puts "#{n.to_s.rjust(5)}  #{("%.1f%%" % (sum_raw * 100.0 / total)).rjust(8)}  #{("%.1f%%" % (sum_nv * 100.0 / total)).rjust(8)}  #{("%.1f%%" % (sum_ns * 100.0 / total)).rjust(8)}  #{("%.1f%%" % (sum_na * 100.0 / total)).rjust(8)}"
      end

      store.close
    end
  end
end

Semtrace::CapacityTest.run
