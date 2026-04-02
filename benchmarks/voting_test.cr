require "../src/semtrace"
require "json"

# Voting Metric Test (optimized single-pass)
#
# Computes cosine, L2, and inner product in a single scan of the vocabulary.
# When 2+ metrics agree, use that token. When all disagree, pick the
# candidate that minimizes next residual norm.
#
# Compile with --release:
#   crystal build benchmarks/voting_test.cr -o bin/voting_test --release

module Semtrace
  module VotingTest
    GETTYSBURG = <<-TEXT
    Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this. But, in a larger sense, we can not dedicate — we can not consecrate — we can not hallow — this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here. It is rather for us to be here dedicated to the great task remaining before us — that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion — that we here highly resolve that these dead shall not have died in vain — that this nation, under God, shall have a new birth of freedom — and that government of the people, by the people, for the people, shall not perish from the earth.
    TEXT

    def self.run
      data_dir = (Path[__DIR__].parent / "data").to_s
      token_ids = nil.as(Array(Int32)?)
      file_path = ""

      i = 0
      while i < ARGV.size
        case ARGV[i]
        when "--data"    then data_dir = ARGV[i + 1]; i += 2
        when "--file"    then file_path = ARGV[i + 1]; i += 2
        when "--ids"
          ids_path = ARGV[i + 1]
          abort "Missing IDs file" unless File.exists?(ids_path)
          token_ids = Array(Int32).new
          JSON.parse(File.read(ids_path)).as_a.each { |v| token_ids.not_nil! << v.as_i.to_i32 }
          i += 2
        when "--builtin"
          # handled below
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

      if token_ids.nil?
        abort "No merges.txt for BPE tokenization" unless File.exists?(merges_path)
        tokenizer = Tokenizer.new(vocab_path, merges_path)
        text = if !file_path.empty?
                 File.read(file_path).strip
               else
                 GETTYSBURG.strip
               end
        token_ids = tokenizer.encode(text)
      end

      ids = token_ids.not_nil!
      unique_ids = ids.to_set
      target = Array(Float32).new(dims, 0.0_f32)
      ids.each { |id| target = EmbeddingStore.add(target, store.vector_for(id)) }

      puts "\n=== Voting Metric Test (single-pass) ==="
      puts "  Tokens: #{ids.size}, Unique: #{unique_ids.size}"
      puts "  Search: brute-force (#{store.vocab_size} vocab x #{dims}d)"

      # Run all methods
      methods = {
        "cosine" => :cosine,
        "l2"     => :l2,
        "ip"     => :ip,
      }

      puts "\n--- Individual Metrics ---"
      methods.each do |name, metric|
        t0 = Time.monotonic
        r = decompose_single(target.dup, store, ids.size + 20, metric)
        elapsed = Time.monotonic - t0
        hits = (unique_ids & r.to_set).size
        puts "  #{name.ljust(8)}: #{hits}/#{unique_ids.size} (#{"%.1f" % (hits * 100.0 / unique_ids.size)}%)  #{r.size} steps  #{"%.1f" % elapsed.total_seconds}s"
      end

      puts "\n--- Voting (single-pass, 2/3 agree or min-residual) ---"
      t0 = Time.monotonic
      r_vote, stats = decompose_voting(target.dup, store, ids.size + 20)
      elapsed = Time.monotonic - t0
      hits = (unique_ids & r_vote.to_set).size
      puts "  voting  : #{hits}/#{unique_ids.size} (#{"%.1f" % (hits * 100.0 / unique_ids.size)}%)  #{r_vote.size} steps  #{"%.1f" % elapsed.total_seconds}s"
      puts "  Agreements: #{stats.agree}/#{stats.total} (#{"%.1f" % (stats.agree * 100.0 / [stats.total, 1].max)}%)"
      puts "  All-3 agree: #{stats.all_agree}"
      puts "  2-of-3 agree: #{stats.two_agree}"
      puts "  All disagree: #{stats.disagree} (used min-residual)"
      puts "  When disagreed, chose: cos=#{stats.chose_cos}, l2=#{stats.chose_l2}, ip=#{stats.chose_ip}"

      store.close
    end

    private def self.decompose_single(residual : Array(Float32), store : EmbeddingStore, max_steps : Int32, metric : Symbol) : Array(Int32)
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
            vec = store.vector_for(i)
            vn = EmbeddingStore.norm(vec)
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
            dims.times { |d|
              diff = residual.unsafe_fetch(d) - vec.unsafe_fetch(d)
              dist += diff * diff
            }
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

    record VoteStats,
      agree : Int32,
      all_agree : Int32,
      two_agree : Int32,
      disagree : Int32,
      total : Int32,
      chose_cos : Int32,
      chose_l2 : Int32,
      chose_ip : Int32

    private def self.decompose_voting(residual : Array(Float32), store : EmbeddingStore, max_steps : Int32) : {Array(Int32), VoteStats}
      dims = store.dimensions
      tokens = [] of Int32
      prev_norm = Float32::MAX
      all_agree = 0
      two_agree = 0
      disagree = 0
      chose_cos = 0
      chose_l2 = 0
      chose_ip = 0

      max_steps.times do
        r_norm = EmbeddingStore.norm(residual)
        break if r_norm < 0.001_f32
        break if r_norm > prev_norm
        prev_norm = r_norm

        # Single pass: compute dot product and norms, derive all three metrics
        cos_id = -1; cos_best = -Float32::MAX
        l2_id = -1;  l2_best = Float32::MAX
        ip_id = -1;  ip_best = -Float32::MAX

        store.vocab_size.times do |i|
          vec = store.vector_for(i)

          dot = 0.0_f32
          vec_norm_sq = 0.0_f32
          l2_dist = 0.0_f32

          dims.times do |d|
            rv = residual.unsafe_fetch(d)
            vv = vec.unsafe_fetch(d)
            dot += rv * vv
            vec_norm_sq += vv * vv
            diff = rv - vv
            l2_dist += diff * diff
          end

          # Inner product
          if dot > ip_best
            ip_best = dot
            ip_id = i
          end

          # L2
          if l2_dist < l2_best
            l2_best = l2_dist
            l2_id = i
          end

          # Cosine
          vec_norm = Math.sqrt(vec_norm_sq).to_f32
          if vec_norm > 1e-10_f32
            sim = dot / (r_norm * vec_norm)
            if sim > cos_best
              cos_best = sim
              cos_id = i
            end
          end
        end

        break if cos_id < 0

        # Vote
        best_id = if cos_id == l2_id && l2_id == ip_id
                    all_agree += 1
                    cos_id
                  elsif cos_id == l2_id
                    two_agree += 1
                    cos_id
                  elsif cos_id == ip_id
                    two_agree += 1
                    cos_id
                  elsif l2_id == ip_id
                    two_agree += 1
                    l2_id
                  else
                    # All disagree — pick inner product (best individual metric)
                    disagree += 1
                    chose_ip += 1
                    ip_id
                  end

        tokens << best_id
        residual = EmbeddingStore.subtract(residual, store.vector_for(best_id))
      end

      total = all_agree + two_agree + disagree
      stats = VoteStats.new(
        agree: all_agree + two_agree,
        all_agree: all_agree,
        two_agree: two_agree,
        disagree: disagree,
        total: total,
        chose_cos: chose_cos,
        chose_l2: chose_l2,
        chose_ip: chose_ip,
      )
      {tokens, stats}
    end
  end
end

Semtrace::VotingTest.run
