require "../src/semtrace"

# Normalization Impact Test
#
# Tests how L2 normalization affects greedy residual decomposition.
# Three modes:
#   --raw       Sum of raw vectors, decomposed against raw vectors (baseline)
#   --norm-vecs Sum of L2-normalized vectors, decomposed against normalized vectors
#   --norm-all  Same as norm-vecs but the sum is also L2-normalized
#   (default: all three)
#
# Uses brute-force cosine search for all modes to ensure fair comparison.
#
# Usage:
#   crystal run benchmarks/norm_test.cr -- [options]
#   # or compile first:
#   crystal build benchmarks/norm_test.cr -o bin/norm_test
#   bin/norm_test --data data/gpt2-xl --trials 5 --max 100

module Semtrace
  module NormTest
    def self.run
      data_dir = (Path[__DIR__].parent / "data").to_s
      trials = 5
      test_ns = [6, 12, 25, 50, 100]
      modes = [:raw, :norm_vecs, :norm_all]

      # Parse args
      i = 0
      while i < ARGV.size
        case ARGV[i]
        when "--data"     then data_dir = ARGV[i + 1]; i += 2
        when "--trials"   then trials = ARGV[i + 1].to_i; i += 2
        when "--max"      then max_n = ARGV[i + 1].to_i; test_ns = test_ns.select { |n| n <= max_n }; i += 2
        when "--raw"      then modes = [:raw]; i += 1
        when "--norm-vecs" then modes = [:norm_vecs]; i += 1
        when "--norm-all" then modes = [:norm_all]; i += 1
        when "--ns"       then test_ns = ARGV[i + 1].split(",").map(&.to_i); i += 2
        else
          STDERR.puts "Unknown option: #{ARGV[i]}"
          STDERR.puts "Usage: norm_test [--data DIR] [--trials N] [--max N] [--ns 6,12,25] [--raw|--norm-vecs|--norm-all]"
          exit 1
        end
      end

      embeddings_path = File.join(data_dir, "embeddings.bin")
      vocab_path = File.join(data_dir, "vocab.json")

      unless File.exists?(embeddings_path) && File.exists?(vocab_path)
        STDERR.puts "Missing data files in #{data_dir}"
        exit 1
      end

      model_name = Path[data_dir].basename
      print "Loading #{model_name} embeddings (brute-force mode)... "
      store = EmbeddingStore.new(embeddings_path, vocab_path, skip_index: true)
      puts "#{store.vocab_size} tokens x #{store.dimensions}d"

      dims = store.dimensions

      # Token range: avoid extremes of the vocabulary
      max_token = [store.vocab_size - 1, 20000].min
      min_token = [1000, store.vocab_size // 10].min
      good_tokens = (min_token..max_token).to_a

      puts "\n=== Normalization Impact Test ==="
      puts "  Model: #{model_name} (#{store.vocab_size} tokens x #{dims}d)"
      puts "  Trials: #{trials}, Token range: #{min_token}..#{max_token}"
      puts "  Modes: #{modes.join(", ")}"
      puts "  Search: brute-force cosine (all modes)"

      # Header
      header = "#{"N".rjust(5)}"
      header += "  #{"Raw".rjust(8)}" if modes.includes?(:raw)
      header += "  #{"NormVecs".rjust(8)}" if modes.includes?(:norm_vecs)
      header += "  #{"NormAll".rjust(8)}" if modes.includes?(:norm_all)
      puts "\n#{header}"
      puts "-" * header.size

      test_ns.each do |n|
        next if n > good_tokens.size

        sum_raw = 0
        sum_nv = 0
        sum_fn = 0

        trials.times do |trial|
          rng = Random.new((42 * (n + 1) + trial).to_u64)
          token_ids = good_tokens.sample(n, rng)
          tid_set = token_ids.to_set

          if modes.includes?(:raw)
            target_raw = Array(Float32).new(dims, 0.0_f32)
            token_ids.each { |tid| target_raw = EmbeddingStore.add(target_raw, store.vector_for(tid)) }
            result = brute_force_decompose(target_raw, store, n + 10, normalize_vecs: false)
            sum_raw += (tid_set & result.to_set).size
          end

          if modes.includes?(:norm_vecs) || modes.includes?(:norm_all)
            target_nv = Array(Float32).new(dims, 0.0_f32)
            token_ids.each do |tid|
              vec = store.vector_for(tid).to_a
              vec_norm = EmbeddingStore.norm(vec)
              normalized = vec.map { |v| v / vec_norm }
              target_nv = EmbeddingStore.add(target_nv, normalized)
            end

            if modes.includes?(:norm_vecs)
              result = brute_force_decompose(target_nv, store, n + 10, normalize_vecs: true)
              sum_nv += (tid_set & result.to_set).size
            end

            if modes.includes?(:norm_all)
              fn_norm = EmbeddingStore.norm(target_nv)
              target_fn = target_nv.map { |v| v / fn_norm }
              result = brute_force_decompose(target_fn, store, n + 10, normalize_vecs: true)
              sum_fn += (tid_set & result.to_set).size
            end
          end
        end

        total = n * trials
        line = "#{n.to_s.rjust(5)}"
        line += "  #{(sum_raw * 100 // total).to_s.rjust(7)}%" if modes.includes?(:raw)
        line += "  #{(sum_nv * 100 // total).to_s.rjust(7)}%" if modes.includes?(:norm_vecs)
        line += "  #{(sum_fn * 100 // total).to_s.rjust(7)}%" if modes.includes?(:norm_all)
        puts line
      end

      store.close
    end

    # Brute-force greedy decomposition with optional vector normalization.
    # Uses the same search method for all modes to ensure fair comparison.
    private def self.brute_force_decompose(
      target : Array(Float32),
      store : EmbeddingStore,
      max_steps : Int32,
      normalize_vecs : Bool
    ) : Array(Int32)
      dims = store.dimensions
      residual = target.dup
      tokens = [] of Int32
      prev_norm = Float32::MAX

      max_steps.times do
        r_norm = EmbeddingStore.norm(residual)
        break if r_norm < 0.001_f32
        break if r_norm > prev_norm
        prev_norm = r_norm

        best_id = -1
        best_sim = -Float32::MAX

        store.vocab_size.times do |i|
          vec = store.vector_for(i)
          vec_norm = EmbeddingStore.norm(vec)
          next if vec_norm < 1e-10_f32

          dot = 0.0_f32
          dims.times { |d| dot += residual.unsafe_fetch(d) * vec.unsafe_fetch(d) }
          sim = dot / (r_norm * vec_norm)

          if sim > best_sim
            best_sim = sim
            best_id = i
          end
        end

        break if best_id < 0
        tokens << best_id

        vec = store.vector_for(best_id)
        if normalize_vecs
          vec_norm = EmbeddingStore.norm(vec)
          dims.times { |d| residual[d] -= vec.unsafe_fetch(d) / vec_norm }
        else
          dims.times { |d| residual[d] -= vec.unsafe_fetch(d) }
        end
      end

      tokens
    end
  end
end

Semtrace::NormTest.run
