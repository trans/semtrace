require "../src/semtrace"
require "json"

# Union Metric Test
#
# Runs cosine, L2, and inner product independently, then unions the
# recovered token sets. Reports exact match, case-insensitive match,
# and the still-missing tokens.
#
# Compile with --release:
#   crystal build benchmarks/union_test.cr -o bin/union_test --release

module Semtrace
  module UnionTest
    def self.run
      data_dir = (Path[__DIR__].parent / "data").to_s
      token_ids = nil.as(Array(Int32)?)
      file_path = ""
      merges_path_override = ""

      i = 0
      while i < ARGV.size
        case ARGV[i]
        when "--data" then data_dir = ARGV[i + 1]; i += 2
        when "--file" then file_path = ARGV[i + 1]; i += 2
        when "--ids"
          ids_path = ARGV[i + 1]
          abort "Missing IDs file" unless File.exists?(ids_path)
          token_ids = Array(Int32).new
          JSON.parse(File.read(ids_path)).as_a.each { |v| token_ids.not_nil! << v.as_i.to_i32 }
          i += 2
        else i += 1
        end
      end

      embeddings_path = File.join(data_dir, "embeddings.bin")
      vocab_path = File.join(data_dir, "vocab.json")
      merges_path = File.join(data_dir, "merges.txt")
      abort "Missing data" unless File.exists?(embeddings_path) && File.exists?(vocab_path)

      model_name = Path[data_dir].basename
      print "Loading #{model_name} (brute-force)... "
      store = EmbeddingStore.new(embeddings_path, vocab_path, skip_index: true)
      puts "#{store.vocab_size} tokens x #{store.dimensions}d"
      dims = store.dimensions

      # Get token IDs
      if token_ids.nil?
        abort "No merges.txt for BPE tokenization" unless File.exists?(merges_path)
        abort "Provide --file <path>" if file_path.empty?
        tokenizer = Tokenizer.new(vocab_path, merges_path)
        text = File.read(file_path).strip
        token_ids = tokenizer.encode(text)
      end

      ids = token_ids.not_nil!
      unique_ids = ids.to_set

      target = Array(Float32).new(dims, 0.0_f32)
      ids.each { |id| target = EmbeddingStore.add(target, store.vector_for(id)) }

      puts "\n=== Union Metric Test ==="
      puts "  Tokens: #{ids.size}, Unique: #{unique_ids.size}"
      puts "  Search: brute-force (#{store.vocab_size} x #{dims}d)"

      # Run each metric
      results = Hash(Symbol, Array(Int32)).new
      [:cosine, :l2, :ip].each do |metric|
        print "  Running #{metric}... "
        STDOUT.flush
        t0 = Time.monotonic
        r = decompose_bf(target.dup, store, ids.size + 20, metric)
        elapsed = Time.monotonic - t0
        results[metric] = r
        puts "#{r.size} steps, #{"%.1f" % elapsed.total_seconds}s"
      end

      # Build sets
      cos_set = results[:cosine].to_set
      l2_set = results[:l2].to_set
      ip_set = results[:ip].to_set
      union_set = cos_set | l2_set | ip_set

      # Case-insensitive lookup
      orig_lower = Hash(String, Set(Int32)).new { |h, k| h[k] = Set(Int32).new }
      unique_ids.each { |id| orig_lower[store.token_for(id).downcase.strip] << id }

      # Compute metrics
      puts "\n#{"Method".ljust(15)} #{"Exact".rjust(10)} #{"Case-Ins".rjust(10)} #{"IDs Found".rjust(10)} #{"Extran".rjust(8)}"
      puts "-" * 56

      [{" Cosine", cos_set}, {" L2", l2_set}, {" IP", ip_set}, {" UNION", union_set}].each do |name, rset|
        exact = (unique_ids & rset).size
        ci = ci_match(rset, unique_ids, orig_lower, store)
        extra = rset.size - exact  # unique IDs not in original (exact)
        total = unique_ids.size
        puts "#{name.ljust(15)} #{("%.1f%%" % (exact * 100.0 / total)).rjust(10)} #{("%.1f%%" % (ci * 100.0 / total)).rjust(10)} #{rset.size.to_s.rjust(10)} #{extra.to_s.rjust(8)}"
      end

      # What's still missing?
      union_matched = Set(Int32).new
      union_set.each do |rid|
        if unique_ids.includes?(rid)
          union_matched << rid
        else
          key = store.token_for(rid).downcase.strip
          if orig_ids = orig_lower[key]?
            orig_ids.each { |oid| union_matched << oid }
          end
        end
      end
      still_missing = unique_ids - union_matched
      if still_missing.any?
        puts "\n  Still missing after union + case-insensitive (#{still_missing.size}):"
        still_missing.each { |id| puts "    #{store.token_for(id).inspect}" }
      else
        puts "\n  ALL TOKENS RECOVERED (exact or case-insensitive)"
      end

      store.close
    end

    private def self.ci_match(recovered_set, unique_ids, orig_lower, store) : Int32
      matched = Set(Int32).new
      recovered_set.each do |rid|
        if unique_ids.includes?(rid)
          matched << rid
        else
          key = store.token_for(rid).downcase.strip
          if orig_ids = orig_lower[key]?
            orig_ids.each { |oid| matched << oid }
          end
        end
      end
      matched.size
    end

    private def self.decompose_bf(residual : Array(Float32), store : EmbeddingStore, max_steps : Int32, metric : Symbol) : Array(Int32)
      dims = store.dimensions
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
            vec = store.vector_for(i); vn = EmbeddingStore.norm(vec)
            next if vn < 1e-10_f32
            dot = 0.0_f32
            dims.times { |d| dot += residual.unsafe_fetch(d) * vec.unsafe_fetch(d) }
            sim = dot / (r_norm * vn)
            if sim > best_val; best_val = sim; best_id = i; end
          end
        when :l2
          best_val = Float32::MAX
          store.vocab_size.times do |i|
            vec = store.vector_for(i)
            dist = 0.0_f32
            dims.times { |d| diff = residual.unsafe_fetch(d) - vec.unsafe_fetch(d); dist += diff * diff }
            if dist < best_val; best_val = dist; best_id = i; end
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

Semtrace::UnionTest.run
