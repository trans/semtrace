require "../../src/semtrace"
require "json"

# Metric Comparison for Llama (uses pre-tokenized IDs instead of BPE tokenizer)
#
# Compile with --release:
#   crystal build benchmarks/metric_test_llama.cr -o bin/metric_test_llama --release
#
# Usage:
#   bin/metric_test_llama --data data/llama-3-2-3b-instruct --ids data/llama-3-2-3b-instruct/gettysburg_ids.json

module Semtrace
  module MetricTestLlama
    def self.run
      data_dir = ""
      ids_path = ""

      i = 0
      while i < ARGV.size
        case ARGV[i]
        when "--data" then data_dir = ARGV[i + 1]; i += 2
        when "--ids"  then ids_path = ARGV[i + 1]; i += 2
        else i += 1
        end
      end

      abort "Usage: metric_test_llama --data <dir> --ids <ids.json>" if data_dir.empty? || ids_path.empty?

      embeddings_path = File.join(data_dir, "embeddings.bin")
      vocab_path = File.join(data_dir, "vocab.json")
      abort "Missing data" unless File.exists?(embeddings_path) && File.exists?(vocab_path)
      abort "Missing IDs file: #{ids_path}" unless File.exists?(ids_path)

      model_name = Path[data_dir].basename
      print "Loading #{model_name} (skip index — brute-force only)... "
      store = EmbeddingStore.new(embeddings_path, vocab_path, skip_index: true)
      puts "#{store.vocab_size} tokens x #{store.dimensions}d"
      dims = store.dimensions

      # Load pre-tokenized IDs
      ids = Array(Int32).new
      JSON.parse(File.read(ids_path)).as_a.each { |v| ids << v.as_i.to_i32 }
      unique_ids = ids.to_set

      target = Array(Float32).new(dims, 0.0_f32)
      ids.each { |id| target = EmbeddingStore.add(target, store.vector_for(id)) }

      puts "\n=== Metric Comparison: #{Path[ids_path].basename} ==="
      puts "  Tokens: #{ids.size}, Unique: #{unique_ids.size}"
      puts "  Search: brute-force (#{store.vocab_size} vocab x #{dims}d)"
      puts

      [:cosine, :ip].each do |metric|
        t0 = Time.monotonic
        recovered = decompose_bf(target, store, ids.size + 20, metric)
        elapsed = Time.monotonic - t0

        recovered_set = recovered.to_set
        exact = (unique_ids & recovered_set).size

        # Semantic matches
        missing = unique_ids - recovered_set
        sem5 = 0
        sem7 = 0
        missing.each do |mid|
          best = Float32::MAX
          recovered_set.each do |rid|
            d = USearch::Index.distance(store.vector_for(mid).to_a, store.vector_for(rid).to_a, :cos)
            best = d if d < best
          end
          sem5 += 1 if best < 0.5
          sem7 += 1 if best < 0.7
        end

        pct = "%.1f" % (exact * 100.0 / unique_ids.size)
        sem5_pct = "%.1f" % ((exact + sem5) * 100.0 / unique_ids.size)
        sem7_pct = "%.1f" % ((exact + sem7) * 100.0 / unique_ids.size)

        puts "  #{metric.to_s.ljust(8)}  exact: #{pct.rjust(5)}% (#{exact}/#{unique_ids.size})  sem<0.5: #{sem5_pct.rjust(5)}%  sem<0.7: #{sem7_pct.rjust(5)}%  steps: #{recovered.size}  time: #{"%.1f" % elapsed.total_seconds}s"
      end

      store.close
    end

    private def self.decompose_bf(target : Array(Float32), store : EmbeddingStore, max_steps : Int32, metric : Symbol) : Array(Int32)
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
        case metric
        when :cosine
          best_val = -Float32::MAX
          store.vocab_size.times do |i|
            vec = store.vector_for(i)
            vn = EmbeddingStore.norm(vec)
            next if vn < 1e-10_f32
            dot = 0.0_f32
            dims.times { |d| dot += residual.unsafe_fetch(d) * vec.unsafe_fetch(d) }
            sim = dot / (r_norm * vn)
            if sim > best_val; best_val = sim; best_id = i; end
          end
        when :ip
          best_val = -Float32::MAX
          store.vocab_size.times do |i|
            vec = store.vector_for(i)
            dot = 0.0_f32
            dims.times { |d| dot += residual.unsafe_fetch(d) * vec.unsafe_fetch(d) }
            if dot > best_val; best_val = dot; best_id = i; end
          end
        end

        break if best_id < 0
        tokens << best_id
        residual = EmbeddingStore.subtract(residual, store.vector_for(best_id))
      end
      tokens
    end
  end
end

Semtrace::MetricTestLlama.run
