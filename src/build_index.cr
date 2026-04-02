require "./semtrace"

# Build and save a USearch HNSW index for a model's embeddings.
# Run once, then subsequent benchmarks can load the pre-built index.
#
# Usage:
#   bin/build_index --data data/llama-3-2-3b-instruct [--norm]

module Semtrace
  module BuildIndex
    def self.run
      data_dir = ""
      build_norm = false

      i = 0
      while i < ARGV.size
        case ARGV[i]
        when "--data" then data_dir = ARGV[i + 1]; i += 2
        when "--norm" then build_norm = true; i += 1
        else i += 1
        end
      end

      abort "Usage: build_index --data <dir> [--norm]" if data_dir.empty?

      embeddings_path = File.join(data_dir, "embeddings.bin")
      index_path = File.join(data_dir, "index.usearch")
      norm_index_path = File.join(data_dir, "index_norm.usearch")

      abort "Missing embeddings.bin in #{data_dir}" unless File.exists?(embeddings_path)

      # Read header only first
      f = File.open(embeddings_path, "rb")
      header = Bytes.new(8)
      f.read(header)
      vocab_size = IO::ByteFormat::LittleEndian.decode(Int32, header[0, 4])
      dims = IO::ByteFormat::LittleEndian.decode(Int32, header[4, 4])

      puts "Embeddings: #{vocab_size} tokens x #{dims}d"
      puts "Estimated memory: ~#{"%.0f" % (vocab_size.to_f * dims * 2 / 1024 / 1024)} MB (index) + #{"%.0f" % (vocab_size.to_f * dims * 4 / 1024 / 1024)} MB (vectors)"

      # Load vectors
      print "Loading vectors... "
      STDOUT.flush
      total_floats = vocab_size.to_i64 * dims
      vectors = Slice(Float32).new(total_floats)
      f.read_fully(vectors.unsafe_slice_of(UInt8))
      f.close
      puts "done"

      # Build raw index
      metric_name = ARGV.includes?("--ip") ? "ip" : "cos"
      metric = ARGV.includes?("--ip") ? USearch::MetricKind::IP : USearch::MetricKind::Cos

      print "Building HNSW index (#{vocab_size} vectors x #{dims}d, #{metric_name})... "
      STDOUT.flush

      index = USearch::Index.new(
        dimensions: dims,
        metric: metric,
        quantization: :f16,
        connectivity: 16,
        expansion_add: 128,
        expansion_search: 64,
      )
      index.reserve(vocab_size)

      last_pct = -1
      vocab_size.times do |i|
        offset = i.to_i64 * dims
        vec = vectors[offset, dims]
        index.add(i.to_u64, vec)

        pct = (i * 100) // vocab_size
        if pct != last_pct && pct % 10 == 0
          print "#{pct}%... "
          STDOUT.flush
          last_pct = pct
        end
      end
      puts "100%"

      save_path = ARGV.includes?("--ip") ? File.join(data_dir, "index_ip.usearch") : index_path
      print "Saving to #{save_path}... "
      index.save(save_path)
      size_mb = File.size(index_path) / (1024.0 * 1024.0)
      puts "#{"%.1f" % size_mb} MB"
      index.close

      # Optionally build normalized index
      if build_norm
        print "Building normalized HNSW index... "
        STDOUT.flush

        norm_index = USearch::Index.new(
          dimensions: dims,
          metric: :cos,
          quantization: :f16,
          connectivity: 16,
          expansion_add: 128,
          expansion_search: 64,
        )
        norm_index.reserve(vocab_size)

        norm_vec = Array(Float32).new(dims, 0.0_f32)
        vocab_size.times do |i|
          offset = i.to_i64 * dims
          vec = vectors[offset, dims]
          n = EmbeddingStore.norm(vec)
          inv_n = n > 1e-10_f32 ? 1.0_f32 / n : 0.0_f32
          dims.times { |d| norm_vec[d] = vec[d] * inv_n }
          norm_index.add(i.to_u64, norm_vec)

          pct = (i * 100) // vocab_size
          if pct != last_pct && pct % 10 == 0
            print "#{pct}%... "
            STDOUT.flush
            last_pct = pct
          end
        end
        puts "100%"

        print "Saving to #{norm_index_path}... "
        norm_index.save(norm_index_path)
        size_mb = File.size(norm_index_path) / (1024.0 * 1024.0)
        puts "#{"%.1f" % size_mb} MB"
        norm_index.close
      end

      puts "Done."
    end
  end
end

Semtrace::BuildIndex.run
