require "../src/semtrace"

# Parameterized Sweep Benchmark
#
# Generates reproducible test text at increasing token counts and measures
# decomposition accuracy. Uses seeded random text generation for reproducibility.
#
# Usage:
#   bin/sweep [--data DIR] [--min N] [--max N] [--step N] [--trials T]
#             [--lookahead K] [--battery simple|complex|multi]

module Semtrace
  module SweepBench
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

    def self.run
      data_dir = (Path[__DIR__].parent / "data").to_s
      min_n = 50
      max_n = 500
      step = 50
      lookahead = 1
      trials = 3
      battery = "simple"

      i = 0
      while i < ARGV.size
        case ARGV[i]
        when "--data"      then data_dir = ARGV[i + 1]; i += 2
        when "--min"       then min_n = ARGV[i + 1].to_i; i += 2
        when "--max"       then max_n = ARGV[i + 1].to_i; i += 2
        when "--step"      then step = ARGV[i + 1].to_i; i += 2
        when "--trials"    then trials = ARGV[i + 1].to_i; i += 2
        when "--lookahead", "-k" then lookahead = ARGV[i + 1].to_i; i += 2
        when "--battery"   then battery = ARGV[i + 1]; i += 2
        else
          STDERR.puts "Unknown option: #{ARGV[i]}"
          exit 1
        end
      end

      embeddings_path = File.join(data_dir, "embeddings.bin")
      vocab_path = File.join(data_dir, "vocab.json")
      merges_path = File.join(data_dir, "merges.txt")
      abort "Missing data files in #{data_dir}" unless File.exists?(embeddings_path) && File.exists?(vocab_path)
      abort "No merges.txt — this benchmark requires a BPE tokenizer" unless File.exists?(merges_path)

      model_name = Path[data_dir].basename
      print "Loading #{model_name}... "
      store = EmbeddingStore.new(embeddings_path, vocab_path)
      puts "#{store.vocab_size} tokens x #{store.dimensions}d"
      decomposer = Decomposer.new(store)
      tokenizer = Tokenizer.new(vocab_path, merges_path)

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
          text = generate_text(battery, n, trial)
          ids = tokenizer.encode(text)
          actual_n = ids.size
          original = ids.map { |id| store.token_for(id) }.sort

          result = decomposer.trace_sentence(ids, max_steps: actual_n + 10, lookahead: lookahead)
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

      store.close
    end

    private def self.generate_text(battery : String, target_n : Int, trial : Int) : String
      seed = target_n * 1000 + trial
      rng = Random.new(seed.to_u64)

      case battery
      when "complex"
        generate_complex(rng, target_n)
      when "multi"
        generate_multi(rng, target_n)
      else
        generate_simple(rng, target_n)
      end
    end

    private def self.generate_simple(rng : Random, target_n : Int) : String
      String.build do |s|
        (target_n * 2).times do |i|
          s << " " if i > 0
          s << SWEEP_WORDS[rng.rand(SWEEP_WORDS.size)]
        end
      end
    end

    private def self.generate_complex(rng : Random, target_n : Int) : String
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
          s << (rng.rand(3) == 0 ? complex[rng.rand(complex.size)] : simple[rng.rand(simple.size)])
        end
      end
    end

    private def self.generate_multi(rng : Random, target_n : Int) : String
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
          s << SWEEP_WORDS[rng.rand(SWEEP_WORDS.size)]
        end
        s << "."
      end
    end
  end
end

Semtrace::SweepBench.run
