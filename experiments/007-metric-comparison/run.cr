require "../../src/semtrace"

# Metric Comparison Test
#
# Compares cosine, L2, and inner product metrics for greedy residual
# decomposition using brute-force exact search. All three metrics
# use the same search method (linear scan) for fair comparison.
#
# Compile with --release for best performance:
#   crystal build benchmarks/metric_test.cr -o bin/metric_test --release
#
# Usage:
#   bin/metric_test --data data/gpt2-xl --file benchmarks/texts/tale-ch1.txt
#   bin/metric_test --data data/gpt2-xl --builtin gettysburg

module Semtrace
  module MetricTest
    GETTYSBURG = <<-TEXT
    Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this. But, in a larger sense, we can not dedicate — we can not consecrate — we can not hallow — this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here. It is rather for us to be here dedicated to the great task remaining before us — that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion — that we here highly resolve that these dead shall not have died in vain — that this nation, under God, shall have a new birth of freedom — and that government of the people, by the people, for the people, shall not perish from the earth.
    TEXT

    MARY = <<-TEXT
    Mary had a little lamb, its fleece was white as snow, and everywhere that Mary went, the lamb was sure to go. It followed her to school one day, which was against the rules. It made the children laugh and play to see a lamb at school.
    TEXT

    def self.run
      data_dir = (Path[__DIR__].parent / "data").to_s
      text = ""
      label = ""

      i = 0
      while i < ARGV.size
        case ARGV[i]
        when "--data" then data_dir = ARGV[i + 1]; i += 2
        when "--file"
          path = ARGV[i + 1]
          abort "File not found: #{path}" unless File.exists?(path)
          text = File.read(path).strip
          label = Path[path].basename
          i += 2
        when "--builtin"
          case ARGV[i + 1]
          when "gettysburg" then text = GETTYSBURG.strip; label = "Gettysburg Address"
          when "mary" then text = MARY.strip; label = "Mary Had a Little Lamb"
          else abort "Unknown: #{ARGV[i + 1]}. Available: gettysburg, mary"
          end
          i += 2
        else i += 1
        end
      end

      text = GETTYSBURG.strip if text.empty?
      label = "Gettysburg Address" if label.empty?

      embeddings_path = File.join(data_dir, "embeddings.bin")
      vocab_path = File.join(data_dir, "vocab.json")
      merges_path = File.join(data_dir, "merges.txt")
      abort "Missing data" unless File.exists?(embeddings_path) && File.exists?(vocab_path)
      abort "No merges.txt" unless File.exists?(merges_path)

      model_name = Path[data_dir].basename
      print "Loading #{model_name} (skip index — brute-force only)... "
      store = EmbeddingStore.new(embeddings_path, vocab_path, skip_index: true)
      puts "#{store.vocab_size} tokens x #{store.dimensions}d"
      tokenizer = Tokenizer.new(vocab_path, merges_path)
      dims = store.dimensions

      ids = tokenizer.encode(text)
      unique_ids = ids.to_set
      target = Array(Float32).new(dims, 0.0_f32)
      ids.each { |id| target = EmbeddingStore.add(target, store.vector_for(id)) }

      puts "\n=== Metric Comparison: #{label} ==="
      puts "  Tokens: #{ids.size}, Unique: #{unique_ids.size}"
      puts "  Search: brute-force (exact)"
      puts

      [:cosine, :l2, :ip].each do |metric|
        t0 = Time.monotonic
        recovered = decompose_bf(target, store, ids.size + 20, metric)
        elapsed = Time.monotonic - t0

        recovered_set = recovered.to_set
        exact = (unique_ids & recovered_set).size

        # Semantic matches for missing tokens
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
        when :l2
          best_val = Float32::MAX
          store.vocab_size.times do |i|
            vec = store.vector_for(i)
            dist = 0.0_f32
            dims.times { |d|
              diff = residual.unsafe_fetch(d) - vec.unsafe_fetch(d)
              dist += diff * diff
            }
            if dist < best_val; best_val = dist; best_id = i; end
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

Semtrace::MetricTest.run
