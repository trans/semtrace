require "../../src/semtrace"

# Optimization Test
#
# Compares decomposition accuracy across optimization strategies:
#   1. Baseline: greedy k=1, raw vectors (HNSW)
#   2. Lookahead: greedy k=5, raw vectors (HNSW)
#   3. Normalized: greedy k=1, L2-normalized vectors (HNSW)
#   4. Both: greedy k=5, normalized vectors (HNSW)
#
# All modes use HNSW approximate nearest-neighbor search.
# Builds two HNSW indexes: one for raw vectors, one for normalized.
#
# Usage:
#   bin/optimization_test --data data/gpt2-xl --file benchmarks/texts/tale-ch1.txt

module Semtrace
  module OptimizationTest
    def self.run
      data_dir = (Path[__DIR__].parent.parent / "data").to_s
      file_path = ""

      i = 0
      while i < ARGV.size
        case ARGV[i]
        when "--data" then data_dir = ARGV[i + 1]; i += 2
        when "--file" then file_path = ARGV[i + 1]; i += 2
        else i += 1
        end
      end

      abort "Usage: optimization_test --file <text_file> [--data DIR]" if file_path.empty?
      abort "File not found: #{file_path}" unless File.exists?(file_path)

      embeddings_path = File.join(data_dir, "embeddings.bin")
      vocab_path = File.join(data_dir, "vocab.json")
      merges_path = File.join(data_dir, "merges.txt")
      abort "Missing data files in #{data_dir}" unless File.exists?(embeddings_path) && File.exists?(vocab_path)
      abort "No merges.txt" unless File.exists?(merges_path)

      model_name = Path[data_dir].basename
      print "Loading #{model_name} (with normalized index)... "
      store = EmbeddingStore.new(embeddings_path, vocab_path, build_norm_index: true)
      puts "#{store.vocab_size} tokens x #{store.dimensions}d"
      tokenizer = Tokenizer.new(vocab_path, merges_path)

      text = File.read(file_path).strip
      ids = tokenizer.encode(text)
      n = ids.size
      unique_ids = ids.to_set
      dims = store.dimensions

      # Count original tokens
      orig_counts = Hash(Int32, Int32).new(0)
      ids.each { |id| orig_counts[id] += 1 }

      puts "\n=== Optimization Test ==="
      puts "  File: #{Path[file_path].basename}"
      puts "  Total tokens: #{n}, Unique: #{unique_ids.size}"
      puts "  Model: #{model_name} (#{store.vocab_size} x #{dims}d)"
      puts "  Search: HNSW (both raw and normalized indexes)"

      # Build raw target
      target_raw = Array(Float32).new(dims, 0.0_f32)
      ids.each { |id| target_raw = EmbeddingStore.add(target_raw, store.vector_for(id)) }

      # Build normalized target (sum of normalized vectors, NOT normalized sum)
      target_norm = Array(Float32).new(dims, 0.0_f32)
      ids.each { |id| target_norm = EmbeddingStore.add(target_norm, store.norm_vector_for(id)) }

      max_steps = n + 20

      configs = [
        {name: "Baseline (k=1, raw)", lookahead: 1, normalized: false},
        {name: "Lookahead (k=5, raw)", lookahead: 5, normalized: false},
        {name: "Normalized (k=1, norm)", lookahead: 1, normalized: true},
        {name: "Both (k=5, norm)", lookahead: 5, normalized: true},
      ]

      results = [] of {String, Int32, Int32, Int32, Int32, Int32}

      configs.each do |config|
        print "\n  Running: #{config[:name]}... "
        STDOUT.flush

        target = config[:normalized] ? target_norm : target_raw
        t0 = Time.monotonic

        recovered = decompose_hnsw(target, store, max_steps, config[:lookahead], config[:normalized])

        elapsed = Time.monotonic - t0

        recovered_set = recovered.to_set
        unique_found = (unique_ids & recovered_set).size
        unique_missing = (unique_ids - recovered_set).size
        extra = (recovered_set - unique_ids).size

        # Total recovery (count-aware)
        rec_counts = Hash(Int32, Int32).new(0)
        recovered.each { |id| rec_counts[id] += 1 }
        total_matched = 0
        orig_counts.each do |id, count|
          total_matched += [count, rec_counts[id]? || 0].min
        end

        results << {config[:name], total_matched, unique_found, unique_missing, extra, recovered.size}

        puts "#{"%.1f" % elapsed.total_seconds}s"
        puts "    Total: #{total_matched}/#{n} (#{"%.1f" % (total_matched * 100.0 / n)}%)"
        puts "    Unique: #{unique_found}/#{unique_ids.size} (#{"%.1f" % (unique_found * 100.0 / unique_ids.size)}%)"
        puts "    Missing unique: #{unique_missing}, Extra unique: #{extra}"
        puts "    Recovered tokens: #{recovered.size}"
      end

      # Summary table
      puts "\n=== Summary ==="
      puts "#{"Config".ljust(28)} #{"Total".rjust(10)} #{"Unique".rjust(10)} #{"Miss".rjust(6)} #{"Extra".rjust(6)} #{"Recov".rjust(6)}"
      puts "-" * 70
      results.each do |name, total, unique, missing, extra, recov|
        total_pct = "%.1f%%" % (total * 100.0 / n)
        unique_pct = "%.1f%%" % (unique * 100.0 / unique_ids.size)
        puts "#{name.ljust(28)} #{total_pct.rjust(10)} #{unique_pct.rjust(10)} #{missing.to_s.rjust(6)} #{extra.to_s.rjust(6)} #{recov.to_s.rjust(6)}"
      end

      store.close
    end

    # Greedy decomposition using HNSW index with optional normalization and lookahead.
    private def self.decompose_hnsw(
      target : Array(Float32),
      store : EmbeddingStore,
      max_steps : Int32,
      lookahead : Int32,
      normalized : Bool
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

        # Search the appropriate index
        candidates = if normalized
                       store.search_normalized(residual, k: lookahead)
                     else
                       store.search(residual, k: lookahead)
                     end
        break if candidates.empty?

        best_id = if candidates.size == 1
                    candidates.first.key.to_i32
                  else
                    # Lookahead: pick candidate that minimizes next residual norm
                    candidates.min_by do |c|
                      cid = c.key.to_i32
                      vec = normalized ? store.norm_vector_for(cid) : store.vector_for(cid)
                      next_residual = EmbeddingStore.subtract(residual, vec)
                      EmbeddingStore.norm(next_residual)
                    end.key.to_i32
                  end

        vec = normalized ? store.norm_vector_for(best_id) : store.vector_for(best_id)
        residual = EmbeddingStore.subtract(residual, vec)
        tokens << best_id
      end

      tokens
    end
  end
end

Semtrace::OptimizationTest.run
