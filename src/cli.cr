require "./semtrace"
require "option_parser"

module Semtrace
  module CLI
    DATA_DIR = Path[__DIR__].parent / "data"

    def self.run
      data_dir = DATA_DIR.to_s
      command = ""
      args = [] of String

      # Parse --data flag before anything else
      i = 0
      raw_args = ARGV.to_a
      while i < raw_args.size
        if raw_args[i] == "--data" && i + 1 < raw_args.size
          data_dir = raw_args[i + 1]
          i += 2
        else
          if command.empty? && !raw_args[i].starts_with?("-")
            command = raw_args[i]
          else
            args << raw_args[i]
          end
          i += 1
        end
      end

      embeddings_path = File.join(data_dir, "embeddings.bin")
      vocab_path = File.join(data_dir, "vocab.json")
      merges_path = File.join(data_dir, "merges.txt")

      # Commands that don't need embeddings loaded
      case command
      when "prepare"
        model = args[0]? || "gpt2"
        Prepare.run(DATA_DIR.to_s, model: model)
        return
      when "extract-gguf"
        gguf_path = args[0]?
        abort "Usage: semtrace extract-gguf <path-to-gguf-file>" unless gguf_path
        abort "File not found: #{gguf_path}" unless File.exists?(gguf_path)
        extract_gguf(gguf_path)
        return
      when "list"
        list_models
        return
      end

      unless File.exists?(embeddings_path) && File.exists?(vocab_path)
        STDERR.puts "Missing data files in #{data_dir}."
        STDERR.puts "  For GPT-2:  bin/semtrace prepare [gpt2|gpt2-medium|gpt2-large|gpt2-xl]"
        STDERR.puts "  For GGUF:   bin/semtrace extract-gguf <path>"
        STDERR.puts "  Then:       bin/semtrace --data <dir> <command>"
        exit 1
      end

      model_name = Path[data_dir].basename
      # Use lightweight mode (no HNSW index) for commands that don't need full search
      skip_index = command == "contextual" || command == "calibrate"
      print "Loading #{model_name} embeddings#{skip_index ? " (brute-force mode)" : ""}... "
      store = EmbeddingStore.new(embeddings_path, vocab_path, skip_index: skip_index)
      puts "#{store.vocab_size} tokens x #{store.dimensions}d"

      tokenizer = if File.exists?(merges_path)
                    Tokenizer.new(vocab_path, merges_path)
                  end

      decomposer = Decomposer.new(store)

      case command
      when "single"
        token = args[0]? || "cat"
        run_single(decomposer, token)
      when "compose"
        abort "Usage: semtrace compose <token1> <token2> ..." if args.empty?
        run_compose(decomposer, args)
      when "sentence"
        text = args.join(" ")
        abort "Usage: semtrace sentence <text>" if text.empty?
        abort "Tokenizer not available (missing merges.txt)" unless tokenizer
        run_sentence(decomposer, tokenizer, text)
      when "arithmetic"
        run_arithmetic_examples(decomposer)
      when "midpoint"
        abort "Usage: semtrace midpoint <tokenA> <tokenB>" unless args.size >= 2
        run_midpoint(decomposer, args[0], args[1])
      when "stress"
        abort "Tokenizer not available (missing merges.txt)" unless tokenizer
        run_stress_test(decomposer, tokenizer)
      when "benchmark"
        run_benchmark(decomposer, store, tokenizer)
      when "sweep"
        abort "Tokenizer not available (missing merges.txt)" unless tokenizer
        run_sweep(decomposer, store, tokenizer, args)
      when "contextual"
        text = args.join(" ")
        abort "Usage: semtrace contextual <text>" if text.empty?
        model = "llama3.2:3b"  # TODO: make configurable
        run_contextual(decomposer, store, text, model)
      when "calibrate"
        model = args[0]? || "llama3.2:3b"
        run_calibrate(store, model)
      when "norm-test"
        run_norm_test(decomposer, store)
      when "interactive"
        run_interactive(decomposer, tokenizer)
      else
        run_all_tests(decomposer, tokenizer)
      end

      store.close
    end

    # Lists available model data directories.
    private def self.list_models
      puts "Available models in #{DATA_DIR}:"
      Dir.each_child(DATA_DIR.to_s) do |name|
        dir = DATA_DIR / name
        next unless File.directory?(dir.to_s)
        emb = dir / "embeddings.bin"
        if File.exists?(emb.to_s)
          # Read header to get dimensions
          File.open(emb.to_s, "rb") do |f|
            header = Bytes.new(8)
            f.read(header)
            vocab = IO::ByteFormat::LittleEndian.decode(Int32, header[0, 4])
            dims = IO::ByteFormat::LittleEndian.decode(Int32, header[4, 4])
            size_mb = File.size(emb.to_s) / (1024.0 * 1024.0)
            puts "  #{name.ljust(30)} #{vocab} tokens x #{dims}d  (#{"%.0f" % size_mb} MB)"
          end
        end
      end

      # Also check root data dir (GPT-2 default location)
      emb = DATA_DIR / "embeddings.bin"
      if File.exists?(emb.to_s)
        File.open(emb.to_s, "rb") do |f|
          header = Bytes.new(8)
          f.read(header)
          vocab = IO::ByteFormat::LittleEndian.decode(Int32, header[0, 4])
          dims = IO::ByteFormat::LittleEndian.decode(Int32, header[4, 4])
          size_mb = File.size(emb.to_s) / (1024.0 * 1024.0)
          puts "  #{"(default)".ljust(30)} #{vocab} tokens x #{dims}d  (#{"%.0f" % size_mb} MB)"
        end
      end
    end

    # =========================================================================
    # Benchmark suite — standardized tests for cross-model comparison
    # =========================================================================

    BENCHMARK_SINGLES = %w[cat dog the hello world man woman king]

    BENCHMARK_PAIRS = [
      ["cat", "dog"],
      ["man", "woman"],
      ["king", "queen"],
    ]

    # Battery A: Simple vocabulary, single clause, common words only
    BENCHMARK_SIMPLE = [
      "the cat sat on the mat",
      "the dog and the cat are friends",
      "the big red dog chased the small black cat across the yard",
      "the old man walked down the long and dusty road that led to the small town by the river",
      "the young girl and her older brother went to the park near the lake and they played with the dog and the cat and then they all went home and had dinner together with the rest of the family",
      "in the morning the sun rose over the hills and the birds began to sing in the trees and the wind blew through the fields of green grass and the farmer went out to the barn to feed the animals and milk the cows before he came back to the house for a hot cup of tea",
      "the big brown dog and the small black cat ran fast down the long dirt road to the old farm where the man and the woman lived with the birds and the fish and they all had food and water and the sun was warm and the sky was blue and the grass was green and the wind was cool and the trees were tall and the flowers were red and the air was fresh and the day was good and the night was dark and the moon was full",
      "the big brown dog and the small black cat ran fast down the long dirt road to the old farm where the man and the woman lived with the birds and the fish and they all had food and water and the sun was warm and the sky was blue and the grass was green and the wind was cool and the trees were tall and the flowers were red and the air was fresh and the day was good and the night was dark and the moon was full and the stars were bright and the world was still and the fire was hot and the snow was white and the rain was cold and the ice was hard and the stone was gray and the sand was soft and the sea was deep and the land was wide",
    ]

    # Battery B: Mixed vocabulary — common words alongside multi-syllable,
    # technical, and compound words that produce BPE subword fragments
    BENCHMARK_COMPLEX = [
      "The old building was undergoing a major renovation.",
      "She carefully reviewed the environmental sustainability report before the meeting.",
      "The new semiconductor chip was surprisingly efficient but the manufacturing process needed improvement.",
      "After the investigation, the researchers published their extraordinary findings about developmental psychology in a well-known international journal last year.",
      "The archaeological team from the university spent three long summers at the remote site before they finally uncovered the sophisticated agricultural tools that changed our understanding of Mediterranean civilizations in the seventeenth century.",
    ]

    # Battery C: Multi-sentence, punctuation, mixed structure
    BENCHMARK_MULTI = [
      "Hello, world! How are you?",
      "The cat sat down. It was tired. The dog, however, was not.",
      "She said: \"I don't believe it.\" He replied, \"Well, it's true.\"",
      "First, we went to the store; then, we stopped at the park. Finally, we came home -- exhausted but happy -- and made dinner.",
      "Dr. Smith (the lead researcher) published her findings in Nature. The results, which contradicted previous studies, were surprising. \"We didn't expect this,\" she admitted. However, the data was clear: the hypothesis was wrong. Period.",
    ]

    private def self.run_benchmark(d : Decomposer, store : EmbeddingStore, tok : Tokenizer?)
      puts "\n=== BENCHMARK: #{store.vocab_size} tokens x #{store.dimensions}d ==="

      # 1. Single token round-trip
      puts "\n--- Single Token Round-Trip ---"
      single_ok = 0
      single_total = 0
      BENCHMARK_SINGLES.each do |token|
        # Try both bare and space-prefixed forms
        [token, " #{token}"].each do |t|
          next unless store.vocab.includes?(t)
          single_total += 1
          result = d.trace_single(t)
          if result.tokens == [t] && result.final_residual_norm < 0.001
            single_ok += 1
          else
            puts "  FAIL: #{t.inspect} -> #{result.tokens.map(&.inspect).join(", ")} (residual: #{"%.4f" % result.final_residual_norm})"
          end
        end
      end
      puts "  #{single_ok}/#{single_total} exact round-trips"

      # 2. Pair composition
      puts "\n--- Pair Composition ---"
      pair_ok = 0
      pair_total = 0
      BENCHMARK_PAIRS.each do |pair|
        # Try space-prefixed
        tokens = pair.map { |t| store.vocab.includes?(" #{t}") ? " #{t}" : t }
        next unless tokens.all? { |t| store.vocab.includes?(t) }
        pair_total += 1
        begin
          result = d.trace_tokens(tokens)
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

      # 3. Sentence decomposition (requires tokenizer)
      if tok
        {
          {"A: Simple Vocabulary", BENCHMARK_SIMPLE},
          {"B: Complex Vocabulary (mixed)", BENCHMARK_COMPLEX},
          {"C: Multi-Sentence & Punctuation", BENCHMARK_MULTI},
        }.each do |label, sentences|
          puts "\n--- #{label} ---"
          puts "#{"N".rjust(5)}  #{"Accuracy".rjust(8)}  #{"Miss".rjust(5)}  #{"Extra".rjust(5)}  Sentence"
          puts "-" * 90

          sentences.each do |text|
            ids = tok.encode(text)
            n = ids.size
            original = ids.map { |id| store.token_for(id) }.sort

            result = d.trace_sentence(ids, max_steps: n + 10)
            recovered = result.tokens.sort

            if original == recovered
              puts "#{n.to_s.rjust(5)}  #{"100.0%".rjust(8)}  #{"0".rjust(5)}  #{"0".rjust(5)}  #{text[0, 60]}"
            else
              missing = (original - recovered).size
              extra = (recovered - original).size
              pct = "%.1f%%" % ((n - missing) * 100.0 / n)
              puts "#{n.to_s.rjust(5)}  #{pct.rjust(8)}  #{missing.to_s.rjust(5)}  #{extra.to_s.rjust(5)}  #{text[0, 60]}"
            end
          end
        end
      else
        puts "\n--- Sentence Decomposition: SKIPPED (no tokenizer) ---"
        puts "  This model has no BPE merges file. Use 'sentence' command with manual token IDs."
      end

      # 4. Embedding space statistics
      puts "\n--- Embedding Space Stats ---"
      # Sample norms
      norms = [] of Float32
      sample_size = [store.vocab_size, 1000].min
      sample_size.times do |i|
        norms << EmbeddingStore.norm(store.vector_for(i))
      end
      norms.sort!
      avg = norms.sum / norms.size
      puts "  Token embedding norms (sample of #{sample_size}):"
      puts "    min: #{"%.4f" % norms.first}, max: #{"%.4f" % norms.last}, avg: #{"%.4f" % avg}"
      puts "    median: #{"%.4f" % norms[norms.size // 2]}"

      puts "\n=== END BENCHMARK ==="
    end

    # =========================================================================
    # Individual commands
    # =========================================================================

    private def self.run_single(d : Decomposer, token : String)
      puts "\n--- E(#{token.inspect}) ---"
      result = d.trace_single(token)
      print_result(result)
    end

    private def self.run_compose(d : Decomposer, tokens : Array(String))
      label = tokens.map { |t| "E(#{t.inspect})" }.join(" + ")
      puts "\n--- #{label} ---"
      result = d.trace_tokens(tokens)
      print_result(result)
    end

    private def self.run_sentence(d : Decomposer, tok : Tokenizer, text : String)
      ids = tok.encode(text)
      token_strs = ids.map { |id| d.@store.token_for(id) }

      puts "\n--- Sentence: #{text.inspect} ---"
      puts "  Tokenized (#{ids.size} tokens): #{token_strs.map(&.inspect).join(", ")}"
      puts "  IDs: #{ids.join(", ")}"
      puts "  Decomposing bag-of-words embedding..."

      result = d.trace_sentence(ids, max_steps: ids.size + 10)
      print_result(result)

      original_set = token_strs.sort
      recovered_set = result.tokens.sort
      if original_set == recovered_set
        puts "  >> Exact match (same token multiset)"
      else
        missing = original_set - recovered_set
        extra = recovered_set - original_set
        puts "  >> Original tokens: #{original_set.map(&.inspect).join(", ")}"
        puts "  >> Recovered tokens: #{recovered_set.map(&.inspect).join(", ")}"
        puts "  >> Missing: #{missing.map(&.inspect).join(", ")}" unless missing.empty?
        puts "  >> Extra: #{extra.map(&.inspect).join(", ")}" unless extra.empty?
      end
    end

    private def self.run_arithmetic_examples(d : Decomposer)
      puts "\n=== Concept Arithmetic ==="

      examples = [
        {positive: [" king", " woman"], negative: [" man"], label: "king - man + woman"},
        {positive: [" Paris", " Germany"], negative: [" France"], label: "Paris - France + Germany"},
        {positive: [" bigger", " cold"], negative: [" big"], label: "bigger - big + cold"},
      ]

      examples.each do |ex|
        puts "\n--- #{ex[:label]} ---"
        begin
          result = d.arithmetic(positive: ex[:positive], negative: ex[:negative])
          print_result(result)
        rescue e
          puts "  Skipped: #{e.message}"
        end
      end
    end

    private def self.run_midpoint(d : Decomposer, a : String, b : String)
      puts "\n--- Midpoint: #{a.inspect} ↔ #{b.inspect} ---"
      result = d.midpoint(a, b)
      print_result(result)
    end

    private def self.run_interactive(d : Decomposer, tok : Tokenizer?)
      puts "\nInteractive mode. Commands:"
      puts "  single <token>           - decompose a single token's embedding"
      puts "  compose <t1> <t2> ...    - sum tokens, then decompose"
      puts "  sentence <text>          - tokenize, sum, and decompose"
      puts "  midpoint <t1> <t2>       - decompose midpoint between two tokens"
      puts "  quit                     - exit"
      puts

      loop do
        print "> "
        line = gets
        break unless line
        line = line.strip
        next if line.empty?
        break if line == "quit" || line == "exit"

        parts = line.split(" ", 2)
        cmd = parts[0]

        begin
          case cmd
          when "single"
            run_single(d, parts[1]? || "cat")
          when "compose"
            tokens = (parts[1]? || "").split
            if tokens.empty?
              puts "Usage: compose <token1> <token2> ..."
            else
              run_compose(d, tokens)
            end
          when "sentence"
            text = parts[1]?
            if text && tok
              run_sentence(d, tok, text)
            elsif !tok
              puts "Tokenizer not available (missing merges.txt)"
            else
              puts "Usage: sentence <text>"
            end
          when "midpoint"
            args = (parts[1]? || "").split
            if args.size == 2
              run_midpoint(d, args[0], args[1])
            else
              puts "Usage: midpoint <token1> <token2>"
            end
          else
            puts "Unknown command: #{cmd}"
          end
        rescue ex
          puts "  Error: #{ex.message}"
        end
      end
    end

    # Reproducible word list for sweep text generation (seeded by position).
    SWEEP_WORDS = %w[
      the dog and cat ran fast down long dirt road to old farm where man
      woman lived with birds fish all had food water sun was warm sky blue
      grass green wind cool trees tall flowers red air fresh day good night
      dark moon full stars bright world still fire hot snow white rain cold
      ice hard stone gray sand soft sea deep land wide river high hill low
      field dry cloud light path home door wall roof floor chair table book
      pen ink lamp bed hat coat shoe ring bell song drum flag rope ship cart
      wheel plow seed crop barn gate fence yard well pond cave fort town gold
      clay silk wool corn wheat hay oil salt iron tin lead coal dust fog dew
    ]

    private def self.run_sweep(d : Decomposer, store : EmbeddingStore, tok : Tokenizer, args : Array(String))
      # Parse sweep arguments
      min_n = 10
      max_n = 200
      step = 25
      lookahead = 1
      trials = 3
      battery = "simple"  # simple, complex, multi

      i = 0
      while i < args.size
        case args[i]
        when "--min"    then min_n = args[i + 1].to_i; i += 2
        when "--max"    then max_n = args[i + 1].to_i; i += 2
        when "--step"   then step = args[i + 1].to_i; i += 2
        when "--lookahead", "-k" then lookahead = args[i + 1].to_i; i += 2
        when "--trials" then trials = args[i + 1].to_i; i += 2
        when "--battery" then battery = args[i + 1]; i += 2
        else
          STDERR.puts "Unknown sweep arg: #{args[i]}"
          STDERR.puts "Usage: semtrace sweep [--min N] [--max N] [--step N] [--lookahead K] [--trials T] [--battery simple|complex|multi]"
          exit 1
        end
      end

      puts "\n=== SWEEP: #{store.vocab_size} tokens x #{store.dimensions}d ==="
      puts "  Range: #{min_n}..#{max_n} (step #{step}), lookahead=#{lookahead}, trials=#{trials}, battery=#{battery}"
      puts
      puts "#{"N".rjust(5)}  #{"Accuracy".rjust(8)}  #{"Miss".rjust(6)}  #{"Extra".rjust(6)}  #{"Over%".rjust(6)}  #{"Recov".rjust(6)}"
      puts "-" * 45

      n = min_n
      while n <= max_n
        total_original = 0
        total_missing = 0
        total_extra = 0
        total_recovered = 0

        trials.times do |trial|
          text = generate_sweep_text(battery, n, trial)
          ids = tok.encode(text)
          actual_n = ids.size
          original = ids.map { |id| store.token_for(id) }.sort

          result = d.trace_sentence(ids, max_steps: actual_n + 10, lookahead: lookahead)
          recovered = result.tokens.sort

          missing = (original - recovered).size
          extra = (recovered - original).size

          total_original += actual_n
          total_missing += missing
          total_extra += extra
          total_recovered += result.tokens.size
        end

        avg_n = total_original // trials
        avg_miss = total_missing.to_f / trials
        avg_extra = total_extra.to_f / trials
        avg_recovered = total_recovered.to_f / trials
        accuracy = (total_original - total_missing) * 100.0 / total_original
        overshoot = total_extra * 100.0 / total_original

        puts "#{avg_n.to_s.rjust(5)}  #{("%.1f%%" % accuracy).rjust(8)}  #{("%.1f" % avg_miss).rjust(6)}  #{("%.1f" % avg_extra).rjust(6)}  #{("%.1f%%" % overshoot).rjust(6)}  #{("%.0f" % avg_recovered).rjust(6)}"

        n += step
      end
    end

    # Generates reproducible test text for a given battery, target token count, and trial.
    private def self.generate_sweep_text(battery : String, target_n : Int, trial : Int) : String
      seed = target_n * 1000 + trial
      rng = Random.new(seed.to_u64)

      case battery
      when "complex"
        generate_complex_text(rng, target_n)
      when "multi"
        generate_multi_text(rng, target_n)
      else
        generate_simple_text(rng, target_n)
      end
    end

    private def self.generate_simple_text(rng : Random, target_n : Int) : String
      # Build text from simple word list, shuffled by seed
      words = SWEEP_WORDS.dup
      String.build do |s|
        (target_n * 2).times do |i|
          s << " " if i > 0
          s << words[rng.rand(words.size)]
        end
      end
    end

    private def self.generate_complex_text(rng : Random, target_n : Int) : String
      # Mix simple words with complex multi-syllable words
      complex = %w[
        environmental investigation pharmaceutical representative
        semiconductor manufacturing psychological extraordinary
        archaeological sophisticated international developmental
        unprecedented sustainability characteristics university
        understanding civilization approximately technological
        communication infrastructure unfortunately revolutionary
      ]
      simple = %w[the a an is was are were had has been to in of and for on with at by from]

      String.build do |s|
        (target_n * 2).times do |i|
          s << " " if i > 0
          if rng.rand(3) == 0
            s << complex[rng.rand(complex.size)]
          else
            s << simple[rng.rand(simple.size)]
          end
        end
      end
    end

    private def self.generate_multi_text(rng : Random, target_n : Int) : String
      # Generate text with punctuation and sentence boundaries
      words = SWEEP_WORDS.dup
      punctuation = [",", ";", ":", ".", "!", "?", " --", " (", ")"]

      String.build do |s|
        (target_n * 2).times do |i|
          if i > 0 && rng.rand(8) == 0
            p = punctuation[rng.rand(punctuation.size)]
            s << p
            s << " " unless p.ends_with?(" ")
          else
            s << " " if i > 0
          end
          s << words[rng.rand(words.size)]
        end
        s << "."
      end
    end

    private def self.run_norm_test(d : Decomposer, store : EmbeddingStore)
      puts "\n=== Normalization Impact Test ==="
      puts "  #{store.vocab_size} tokens x #{store.dimensions}d"
      dims = store.dimensions
      trials = 5

      # Sample from tokens in a safe range
      max_token = [store.vocab_size - 1, 20000].min
      min_token = [1000, store.vocab_size // 10].min
      good_tokens = (min_token..max_token).to_a

      puts "  Trials: #{trials}, Token range: #{min_token}..#{max_token}"
      puts "\n#{"N".rjust(5)}  #{"Raw".rjust(8)}  #{"NormVecs".rjust(8)}  #{"NormAll".rjust(8)}"
      puts "-" * 35

      [6, 12, 25, 50, 100].each do |n|
        next if n > good_tokens.size

        sum_raw = 0
        sum_nv = 0
        sum_fn = 0

        trials.times do |trial|
          rng = Random.new((42 * (n + 1) + trial).to_u64)
          token_ids = good_tokens.sample(n, rng)
          tid_set = token_ids.to_set

          # === Raw baseline ===
          target_raw = Array(Float32).new(dims, 0.0_f32)
          token_ids.each { |tid| target_raw = EmbeddingStore.add(target_raw, store.vector_for(tid)) }
          result_raw = d.decompose(target_raw, max_steps: n + 10)
          sum_raw += (tid_set & result_raw.token_ids.to_set).size

          # === Normalized vecs, unnormalized sum ===
          target_nv = Array(Float32).new(dims, 0.0_f32)
          token_ids.each do |tid|
            vec = store.vector_for(tid).to_a
            vec_norm = EmbeddingStore.norm(vec)
            normalized = vec.map { |v| v / vec_norm }
            target_nv = EmbeddingStore.add(target_nv, normalized)
          end
          result_nv = decompose_normalized(target_nv, store, n + 10)
          sum_nv += (tid_set & result_nv.to_set).size

          # === Fully normalized ===
          target_fn_norm = EmbeddingStore.norm(target_nv)
          target_fn = target_nv.map { |v| v / target_fn_norm }
          result_fn = decompose_normalized(target_fn, store, n + 10)
          sum_fn += (tid_set & result_fn.to_set).size
        end

        total = n * trials
        pct_raw = sum_raw * 100 // total
        pct_nv = sum_nv * 100 // total
        pct_fn = sum_fn * 100 // total

        puts "#{n.to_s.rjust(5)}  #{pct_raw.to_s.rjust(7)}%  #{pct_nv.to_s.rjust(7)}%  #{pct_fn.to_s.rjust(7)}%"
      end
    end

    # Greedy decomposition against L2-normalized token embeddings
    private def self.decompose_normalized(target : Array(Float32), store : EmbeddingStore, max_steps : Int32) : Array(Int32)
      dims = store.dimensions
      residual = target.dup
      tokens = [] of Int32
      prev_norm = Float32::MAX

      max_steps.times do
        r_norm = EmbeddingStore.norm(residual)
        break if r_norm < 0.001_f32
        break if r_norm > prev_norm
        prev_norm = r_norm

        # Find nearest normalized token to residual (by cosine)
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

        # Subtract the NORMALIZED token vector
        vec = store.vector_for(best_id)
        vec_norm = EmbeddingStore.norm(vec)
        dims.times { |d| residual[d] -= vec.unsafe_fetch(d) / vec_norm }
      end

      tokens
    end

    private def self.run_calibrate(store : EmbeddingStore, model : String)
      puts "\n=== Static vs Contextual Calibration ==="
      puts "  Model: #{model}"
      puts "  Vocab: #{store.vocab_size} tokens x #{store.dimensions}d"

      # Pick diverse common tokens that are likely single-token in most models
      test_words = %w[the cat dog sat on man woman king hello world
                      big small red blue run walk good bad is and
                      water fire sun moon day night house tree book car]

      puts "  Embedding #{test_words.size} individual tokens through Ollama..."
      contextual = Ollama.embed_batch(test_words, model)

      puts "\n#{"Token".ljust(10)} #{"Static Norm".rjust(12)} #{"Ctx Norm".rjust(10)} #{"Cos Dist".rjust(10)} #{"Cos Sim".rjust(10)}"
      puts "-" * 55

      sims = [] of Float64
      found = 0

      test_words.each_with_index do |word, i|
        tid = store.vocab.index(word)
        unless tid
          puts "#{word.ljust(10)} — not in vocab"
          next
        end
        found += 1

        static = store.vector_for(tid)
        ctx = contextual[i]

        s_norm = EmbeddingStore.norm(static)
        c_norm = EmbeddingStore.norm(ctx)

        dot = 0.0_f64
        store.dimensions.times do |d|
          dot += static[d].to_f64 * ctx[d].to_f64
        end
        cos_sim = dot / (s_norm * c_norm)
        cos_dist = 1.0 - cos_sim
        sims << cos_sim

        puts "#{word.ljust(10)} #{("%.4f" % s_norm).rjust(12)} #{("%.4f" % c_norm).rjust(10)} #{("%.4f" % cos_dist).rjust(10)} #{("%.4f" % cos_sim).rjust(10)}"
      end

      if found > 0
        avg_sim = sims.sum / sims.size
        min_sim = sims.min
        max_sim = sims.max
        puts "\n  Tokens compared: #{found}"
        puts "  Cosine similarity: avg=#{"%.4f" % avg_sim}, min=#{"%.4f" % min_sim}, max=#{"%.4f" % max_sim}"

        if avg_sim > 0.5
          puts "  => Moderate correlation — linear mapping may work"
        elsif avg_sim > 0.1
          puts "  => Weak correlation — mapping will be challenging"
        else
          puts "  => Near-orthogonal — spaces are fundamentally different"
        end
      end
    end

    private def self.run_contextual(d : Decomposer, store : EmbeddingStore, text : String, model : String)
      puts "\n=== Contextual Embedding Decomposition ==="
      puts "  Model: #{model}"
      puts "  Text: #{text.inspect}"
      puts "  Static index: #{store.vocab_size} tokens x #{store.dimensions}d"

      # Get contextual embedding from Ollama forward pass
      print "  Getting contextual embedding from Ollama... "
      ctx_embedding = Ollama.embed(text, model)
      puts "#{ctx_embedding.size}d"

      if ctx_embedding.size != store.dimensions
        puts "  ERROR: Dimension mismatch! Contextual=#{ctx_embedding.size}d, Static=#{store.dimensions}d"
        return
      end

      # Compare: bag-of-words static sum vs contextual
      # First, find nearest static token to the contextual embedding
      puts "\n--- Nearest static tokens to contextual embedding ---"
      nearest = store.search(ctx_embedding, k: 10)
      nearest.each_with_index do |r, i|
        puts "  #{(i+1).to_s.rjust(2)}. #{store.token_for(r.key).inspect.ljust(15)} (distance: #{"%.4f" % r.distance})"
      end

      # Decompose the contextual embedding
      puts "\n--- Greedy decomposition of contextual embedding ---"
      result = d.decompose(ctx_embedding, max_steps: 30)
      puts "  Tokens (#{result.tokens.size}): #{result.tokens.map(&.inspect).join(", ")}"
      puts "  Norms: #{result.residual_norms.map { |n| "%.4f" % n }.join(" → ")}"
      puts "  Final residual: #{"%.4f" % result.final_residual_norm}"

      # Also show what the bag-of-words decomposition would give for comparison
      # Embed each word individually and sum for a static baseline
      puts "\n--- Comparison: individual token embeddings ---"
      words = text.split
      puts "  Embedding each word individually via Ollama..."
      word_embeddings = Ollama.embed_batch(words, model)

      # How far is the sentence embedding from the sum of word embeddings?
      word_sum = Array(Float32).new(store.dimensions, 0.0_f32)
      word_embeddings.each do |we|
        word_sum = EmbeddingStore.add(word_sum, we)
      end
      ctx_vs_sum = EmbeddingStore.norm(EmbeddingStore.subtract(ctx_embedding, word_sum))
      ctx_norm = EmbeddingStore.norm(ctx_embedding)
      sum_norm = EmbeddingStore.norm(word_sum)

      puts "  Contextual embedding norm: #{"%.4f" % ctx_norm}"
      puts "  Sum of word embeddings norm: #{"%.4f" % sum_norm}"
      puts "  Distance (contextual vs sum): #{"%.4f" % ctx_vs_sum}"
      puts "  Relative distance: #{"%.4f" % (ctx_vs_sum / ctx_norm)}"

      # Also decompose the contextual sum-of-words for comparison
      puts "\n--- Decomposition of contextual word sum ---"
      result2 = d.decompose(word_sum, max_steps: 30)
      puts "  Tokens (#{result2.tokens.size}): #{result2.tokens.map(&.inspect).join(", ")}"
      puts "  Final residual: #{"%.4f" % result2.final_residual_norm}"
    end

    private def self.run_stress_test(d : Decomposer, tok : Tokenizer)
      sentences = [
        "I like cats",
        "The cat sat on the mat",
        "The king and the queen sat on the throne",
        "The big red dog chased the small black cat across the yard",
        "I wanted to go to the store but it was closed so I went home instead",
        "The old man walked down the long and dusty road that led to the small town by the river",
        "She told him that she would not be able to come to the party on Friday because she had to work late at the office that night",
        "The young girl and her older brother went to the park near the lake and they played with the dog and the cat and then they all went home and had dinner together with the rest of the family",
        "In the morning the sun rose over the hills and the birds began to sing in the trees and the wind blew through the fields of green grass and the farmer went out to the barn to feed the animals and milk the cows before he came back to the house for a hot cup of tea",
      ]

      puts "\n=== Stress Test: Greedy (k=1) vs Lookahead (k=5) ==="
      puts "#{"Tokens".rjust(6)}  #{"Greedy".ljust(20)}  #{"Lookahead k=5".ljust(20)}  Sentence"
      puts "-" * 100

      sentences.each do |text|
        ids = tok.encode(text)
        n = ids.size
        token_strs = ids.map { |id| d.@store.token_for(id) }
        original_set = token_strs.sort

        r1 = d.trace_sentence(ids, max_steps: n + 10, lookahead: 1)
        set1 = r1.tokens.sort
        match1 = original_set == set1

        r5 = d.trace_sentence(ids, max_steps: n + 10, lookahead: 5)
        set5 = r5.tokens.sort
        match5 = original_set == set5

        greedy_str = if match1
                       "EXACT".ljust(20)
                     else
                       missing1 = (original_set - set1).size
                       "#{n - missing1}/#{n} (#{missing1} wrong)".ljust(20)
                     end

        lookahead_str = if match5
                          "EXACT".ljust(20)
                        else
                          missing5 = (original_set - set5).size
                          "#{n - missing5}/#{n} (#{missing5} wrong)".ljust(20)
                        end

        short = text.size > 50 ? text[0, 47] + "..." : text
        puts "#{n.to_s.rjust(6)}  #{greedy_str}  #{lookahead_str}  #{short}"
      end
    end

    private def self.run_all_tests(d : Decomposer, tok : Tokenizer?)
      puts "\n=== Sanity Checks ==="

      %w[cat dog the hello].each do |token|
        run_single(d, token)
      end

      puts "\n=== Composition ==="
      [
        ["cat", "sat"],
        [" king", " queen"],
        ["hello", " world"],
      ].each do |tokens|
        begin
          run_compose(d, tokens)
        rescue ex
          puts "  Skipped: #{ex.message}"
        end
      end

      run_arithmetic_examples(d)

      puts "\n=== Midpoints ==="
      run_midpoint(d, "cat", "dog")
      run_midpoint(d, " hot", " cold")

      if tok
        puts "\n=== Sentence Decomposition ==="
        ["The cat sat on the mat",
         "Hello world",
         "King and queen",
        ].each do |text|
          begin
            run_sentence(d, tok, text)
          rescue ex
            puts "  Skipped (#{text.inspect}): #{ex.message}"
          end
        end
      end
    end

    # =========================================================================
    # GGUF extraction
    # =========================================================================

    private def self.extract_gguf(gguf_path : String)
      puts "Reading GGUF header: #{gguf_path}"
      info = GGUF.read_info(gguf_path)

      puts "Model: #{info[:metadata]["general.name"]? || "unknown"}"
      puts "Architecture: #{info[:metadata]["general.architecture"]? || "unknown"}"
      puts "Tensors: #{info[:header].n_tensors}"

      emb = info[:tensors].find { |t| t.name == "token_embd.weight" }
      unless emb
        puts "ERROR: token_embd.weight not found. Available tensors:"
        info[:tensors].each { |t| puts "  #{t.name}: #{t.shape} (#{t.type})" }
        exit 1
      end

      dims = emb.shape[0].to_i32
      vocab_size = emb.shape[1].to_i32
      puts "Found token_embd.weight: #{vocab_size} tokens x #{dims} dims (#{emb.type})"

      total_elements = vocab_size.to_i64 * dims
      size_mb = total_elements * 4 / (1024 * 1024)
      puts "Will dequantize to float32: ~#{size_mb} MB"

      model_name = (info[:metadata]["general.name"]?.try(&.as(String)) || "model")
        .downcase.gsub(/[^a-z0-9]+/, "-").strip("-")
      out_dir = (DATA_DIR / model_name).to_s
      Dir.mkdir_p(out_dir)

      embeddings_path = File.join(out_dir, "embeddings.bin")
      vocab_path = File.join(out_dir, "vocab.json")

      puts "Extracting embeddings to #{embeddings_path}..."
      File.open(embeddings_path, "wb") do |outf|
        header = Bytes.new(8)
        IO::ByteFormat::LittleEndian.encode(vocab_size, header[0, 4])
        IO::ByteFormat::LittleEndian.encode(dims, header[4, 4])
        outf.write(header)

        data = GGUF.read_tensor_f32(gguf_path, emb, info[:data_offset])
        outf.write(data.unsafe_slice_of(UInt8))
      end

      out_mb = File.size(embeddings_path) / (1024.0 * 1024.0)
      puts "Wrote #{embeddings_path} (#{"%.1f" % out_mb} MB)"

      puts "Extracting vocabulary..."
      extract_gguf_vocab(gguf_path, vocab_path, vocab_size)

      puts "Done. To use: bin/semtrace --data #{out_dir}"
    end

    private def self.extract_gguf_vocab(gguf_path : String, vocab_path : String, expected_size : Int32)
      tokens = [] of String

      File.open(gguf_path, "rb") do |f|
        magic = read_gguf_u32(f)
        version = read_gguf_u32(f)
        n_tensors = read_gguf_i64(f)
        n_kv = read_gguf_i64(f)

        n_kv.times do
          key = read_gguf_string(f)
          vtype = read_gguf_u32(f)

          if key == "tokenizer.ggml.tokens" && vtype == 9
            atype = read_gguf_u32(f)
            alen = read_gguf_u64(f)
            if atype == 8
              alen.times { tokens << read_gguf_string(f) }
            else
              alen.times { skip_gguf_value(f, GGUF::ValueType.new(atype)) }
            end
          else
            skip_gguf_value(f, GGUF::ValueType.new(vtype))
          end
        end
      end

      if tokens.empty?
        puts "WARNING: Could not extract vocabulary from GGUF metadata."
        puts "You may need to provide vocab.json manually."
        return
      end

      vocab = Hash(String, String).new
      tokens.each_with_index { |t, i| vocab[i.to_s] = t }
      File.write(vocab_path, vocab.to_json)
      puts "Wrote #{vocab_path} (#{tokens.size} tokens)"
    end

    private def self.read_gguf_u32(f : IO) : UInt32
      buf = Bytes.new(4)
      f.read_fully(buf)
      IO::ByteFormat::LittleEndian.decode(UInt32, buf)
    end

    private def self.read_gguf_i64(f : IO) : Int64
      buf = Bytes.new(8)
      f.read_fully(buf)
      IO::ByteFormat::LittleEndian.decode(Int64, buf)
    end

    private def self.read_gguf_u64(f : IO) : UInt64
      buf = Bytes.new(8)
      f.read_fully(buf)
      IO::ByteFormat::LittleEndian.decode(UInt64, buf)
    end

    private def self.read_gguf_string(f : IO) : String
      len = read_gguf_u64(f)
      buf = Bytes.new(len)
      f.read_fully(buf)
      String.new(buf)
    end

    private def self.skip_gguf_value(f : IO, vtype : GGUF::ValueType)
      case vtype
      when .uint8?, .int8?, .bool? then f.skip(1)
      when .uint16?, .int16?       then f.skip(2)
      when .uint32?, .int32?, .float32? then f.skip(4)
      when .uint64?, .int64?, .float64? then f.skip(8)
      when .string?
        len = read_gguf_u64(f)
        f.skip(len)
      when .array?
        atype = GGUF::ValueType.new(read_gguf_u32(f))
        alen = read_gguf_u64(f)
        alen.times { skip_gguf_value(f, atype) }
      end
    end

    private def self.print_result(result : TraceResult)
      puts "  Tokens: #{result.tokens.map(&.inspect).join(", ")}"
      puts "  IDs:    #{result.token_ids.join(", ")}"
      puts "  Residual norms: #{result.residual_norms.map { |n| "%.4f" % n }.join(" → ")}"
      puts "  Final residual: #{"%.6f" % result.final_residual_norm}"
    end
  end
end

Semtrace::CLI.run
