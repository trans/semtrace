module Semtrace
  # Shared real-text decomposition benchmark.
  # Used by experiments 001-003 and any future real-text test.
  module RealText
    def self.run(text : String, label : String, data_dir : String, trace : Bool = false)
      embeddings_path = File.join(data_dir, "embeddings.bin")
      vocab_path = File.join(data_dir, "vocab.json")
      merges_path = File.join(data_dir, "merges.txt")
      abort "Missing data files in #{data_dir}" unless File.exists?(embeddings_path) && File.exists?(vocab_path)
      abort "No merges.txt — requires BPE tokenizer" unless File.exists?(merges_path)

      model_name = Path[data_dir].basename
      print "Loading #{model_name}... "
      store = EmbeddingStore.new(embeddings_path, vocab_path)
      puts "#{store.vocab_size} tokens x #{store.dimensions}d"
      decomposer = Decomposer.new(store)
      tokenizer = Tokenizer.new(vocab_path, merges_path)

      ids = tokenizer.encode(text.strip)
      token_strs = ids.map { |id| store.token_for(id) }
      unique_ids = ids.to_set

      puts "\n=== Real Text: #{label} ==="
      puts "  Total tokens: #{ids.size}"
      puts "  Unique tokens: #{unique_ids.size}"
      puts "  Unique ratio: #{"%.1f%%" % (unique_ids.size * 100.0 / ids.size)}"

      counts = Hash(Int32, Int32).new(0)
      ids.each { |id| counts[id] += 1 }
      sorted = counts.to_a.sort_by { |_, c| -c }
      puts "  Most repeated:"
      sorted[0, 5].each do |id, count|
        puts "    #{store.token_for(id).inspect.ljust(12)} x#{count}"
      end
      puts "  Tokens appearing once: #{sorted.count { |_, c| c == 1 }}"

      target = Array(Float32).new(store.dimensions, 0.0_f32)
      ids.each { |id| target = EmbeddingStore.add(target, store.vector_for(id)) }

      if trace
        puts "\n  Decomposing (trace mode)..."
        puts "  #{"Step".rjust(5)}  #{"Norm".rjust(10)}  #{"Token".ljust(18)} Hit?"
        puts "  #{"-" * 42}"

        residual = target.dup
        result_tokens = [] of String
        result_ids = [] of Int32
        prev_norm = Float32::MAX
        step = 0

        (ids.size + 20).times do
          norm = EmbeddingStore.norm(residual)
          break if norm < 0.01 || norm > prev_norm
          prev_norm = norm

          results = store.search(residual, k: 1)
          break if results.empty?
          best_id = results.first.key.to_i32
          token = store.token_for(best_id)
          hit = unique_ids.includes?(best_id) ? "HIT" : "miss"

          result_tokens << token
          result_ids << best_id

          puts "  #{step.to_s.rjust(5)}  #{"%.4f" % norm}  #{token.inspect.ljust(18)} #{hit}"

          residual = EmbeddingStore.subtract(residual, store.vector_for(best_id))
          step += 1
        end

        final_norm = EmbeddingStore.norm(residual)
        puts "  #{"".rjust(5)}  #{"%.4f" % final_norm}  (final residual)"

        result = TraceResult.new(
          tokens: result_tokens,
          token_ids: result_ids,
          residual_norms: [] of Float32,
          final_residual_norm: final_norm,
        )
      else
        puts "\n  Decomposing..."
        result = decomposer.decompose(target, max_steps: ids.size + 20)
      end

      original_sorted = token_strs.sort
      recovered_sorted = result.tokens.sort
      if original_sorted == recovered_sorted
        total_missing = 0
        total_extra = 0
      else
        total_missing = (original_sorted - recovered_sorted).size
        total_extra = (recovered_sorted - original_sorted).size
      end
      total_accuracy = (ids.size - total_missing) * 100.0 / ids.size

      recovered_id_set = result.token_ids.to_set
      unique_found = (unique_ids & recovered_id_set).size
      unique_accuracy = unique_found * 100.0 / unique_ids.size

      missing_unique = unique_ids - recovered_id_set
      missing_tokens = missing_unique.map { |id| {store.token_for(id), counts[id]} }
        .sort_by { |_, c| -c }

      puts "\n--- Results ---"
      puts "  Total token recovery:  #{"%.1f%%" % total_accuracy} (#{ids.size - total_missing}/#{ids.size}, #{total_missing} missing, #{total_extra} extra)"
      puts "  Unique token recovery: #{"%.1f%%" % unique_accuracy} (#{unique_found}/#{unique_ids.size})"
      puts "  Recovered tokens: #{result.tokens.size}"
      puts "  Final residual: #{"%.4f" % result.final_residual_norm}"

      if missing_tokens.any?
        puts "\n  Missing unique tokens (#{missing_tokens.size}):"
        missing_tokens.each do |tok, count|
          puts "    #{tok.inspect.ljust(15)} (appeared #{count}x)"
        end
      else
        puts "\n  All unique tokens recovered!"
      end

      store.close
    end

    # Parse common CLI args: --data, --trace, --file, --builtin
    def self.parse_args(builtins : Hash(String, {String, String}))
      data_dir = ""
      text = ""
      label = ""
      trace = false

      i = 0
      while i < ARGV.size
        case ARGV[i]
        when "--data"  then data_dir = ARGV[i + 1]; i += 2
        when "--trace" then trace = true; i += 1
        when "--file"
          path = ARGV[i + 1]
          abort "File not found: #{path}" unless File.exists?(path)
          text = File.read(path)
          label = Path[path].basename
          i += 2
        when "--builtin"
          key = ARGV[i + 1]
          if entry = builtins[key]?
            text = entry[0]
            label = entry[1]
          else
            abort "Unknown builtin: #{key}. Available: #{builtins.keys.join(", ")}"
          end
          i += 2
        else i += 1
        end
      end

      {data_dir: data_dir, text: text, label: label, trace: trace}
    end
  end
end
