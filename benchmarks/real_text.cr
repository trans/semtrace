require "../src/semtrace"

# Real Text Benchmark
#
# Tests decomposition on actual prose — no synthetic word lists.
# Reports both total and unique token recovery.
#
# Usage:
#   bin/real_text [--data DIR] [--file path/to/text] [--builtin gettysburg|maryhadalittlelamb]

module Semtrace
  module RealTextBench
    MARY_HAD_A_LITTLE_LAMB = <<-TEXT
    Mary had a little lamb, its fleece was white as snow, and everywhere that Mary went, the lamb was sure to go. It followed her to school one day, which was against the rules. It made the children laugh and play to see a lamb at school.
    TEXT

    GETTYSBURG = <<-TEXT
    Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this. But, in a larger sense, we can not dedicate — we can not consecrate — we can not hallow — this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here. It is rather for us to be here dedicated to the great task remaining before us — that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion — that we here highly resolve that these dead shall not have died in vain — that this nation, under God, shall have a new birth of freedom — and that government of the people, by the people, for the people, shall not perish from the earth.
    TEXT

    def self.run
      data_dir = (Path[__DIR__].parent / "data").to_s
      text = ""
      label = ""

      i = 0
      while i < ARGV.size
        case ARGV[i]
        when "--data"    then data_dir = ARGV[i + 1]; i += 2
        when "--file"
          path = ARGV[i + 1]
          abort "File not found: #{path}" unless File.exists?(path)
          text = File.read(path)
          label = Path[path].basename
          i += 2
        when "--builtin"
          case ARGV[i + 1]
          when "gettysburg" then text = GETTYSBURG; label = "Gettysburg Address"
          when "maryhadalittlelamb" then text = MARY_HAD_A_LITTLE_LAMB; label = "Mary Had a Little Lamb"
          else abort "Unknown builtin: #{ARGV[i + 1]}. Available: gettysburg, maryhadalittlelamb"
          end
          i += 2
        else i += 1
        end
      end

      text = GETTYSBURG if text.empty?
      label = "Gettysburg Address" if label.empty?

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

      # Tokenize
      ids = tokenizer.encode(text.strip)
      token_strs = ids.map { |id| store.token_for(id) }
      unique_ids = ids.to_set

      puts "\n=== Real Text: #{label} ==="
      puts "  Total tokens: #{ids.size}"
      puts "  Unique tokens: #{unique_ids.size}"
      puts "  Unique ratio: #{"%.1f%%" % (unique_ids.size * 100.0 / ids.size)}"

      # Show token distribution
      counts = Hash(Int32, Int32).new(0)
      ids.each { |id| counts[id] += 1 }
      sorted = counts.to_a.sort_by { |_, c| -c }
      puts "  Most repeated:"
      sorted[0, 5].each do |id, count|
        puts "    #{store.token_for(id).inspect.ljust(12)} x#{count}"
      end
      singles = sorted.count { |_, c| c == 1 }
      puts "  Tokens appearing once: #{singles}"

      # Compose bag-of-words vector
      puts "\n  Decomposing..."
      target = Array(Float32).new(store.dimensions, 0.0_f32)
      ids.each { |id| target = EmbeddingStore.add(target, store.vector_for(id)) }

      result = decomposer.decompose(target, max_steps: ids.size + 20)

      # Total token recovery (with duplicates)
      original_sorted = token_strs.sort
      recovered_sorted = result.tokens.sort
      if original_sorted == recovered_sorted
        total_accuracy = 100.0
        total_missing = 0
        total_extra = 0
      else
        total_missing = (original_sorted - recovered_sorted).size
        total_extra = (recovered_sorted - original_sorted).size
        total_accuracy = (ids.size - total_missing) * 100.0 / ids.size
      end

      # Unique token recovery (set-based)
      recovered_id_set = result.token_ids.to_set
      unique_found = (unique_ids & recovered_id_set).size
      unique_accuracy = unique_found * 100.0 / unique_ids.size

      # Unique tokens that were NOT recovered
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
  end
end

Semtrace::RealTextBench.run
