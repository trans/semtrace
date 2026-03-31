require "json"
require "usearch"

module Semtrace
  # Loads and indexes a static token embedding matrix for nearest-neighbor search.
  #
  # Binary format (from extract_embeddings.py):
  #   Header: vocab_size (u32 LE), dimensions (u32 LE)
  #   Data:   vocab_size * dimensions float32 values, row-major
  class EmbeddingStore
    getter vocab_size : Int32
    getter dimensions : Int32
    getter vocab : Array(String)

    @index : USearch::Index
    @norm_index : USearch::Index?
    @vectors : Slice(Float32)
    @norm_vectors : Slice(Float32)?

    def initialize(embeddings_path : String, vocab_path : String, skip_index : Bool = false, build_norm_index : Bool = false)
      # Load vocabulary
      vocab_json = JSON.parse(File.read(vocab_path))
      @vocab = Array(String).new(vocab_json.size) { |i| vocab_json[i.to_s].as_s }

      # Load binary embedding matrix
      f = File.open(embeddings_path, "rb")
      begin
        header = Bytes.new(8)
        f.read(header)
        @vocab_size = IO::ByteFormat::LittleEndian.decode(Int32, header[0, 4])
        @dimensions = IO::ByteFormat::LittleEndian.decode(Int32, header[4, 4])

        raise "Vocab size mismatch: file has #{@vocab_size}, vocab.json has #{@vocab.size}" unless @vocab_size == @vocab.size

        # Read all vectors into a contiguous buffer
        total_floats = @vocab_size.to_i64 * @dimensions
        @vectors = Slice(Float32).new(total_floats)
        byte_slice = @vectors.unsafe_slice_of(UInt8)
        f.read_fully(byte_slice)
      ensure
        f.close
      end

      if skip_index
        # Lightweight mode — no HNSW index, use brute-force search
        @index = USearch::Index.new(dimensions: @dimensions, metric: :cos)
      else
        # Build USearch HNSW index for raw vectors
        @index = USearch::Index.new(
          dimensions: @dimensions,
          metric: :cos,
          quantization: :f16,
          connectivity: 32,
          expansion_add: 128,
          expansion_search: 256,
        )
        @index.reserve(@vocab_size)

        @vocab_size.times do |i|
          vec = vector_for(i)
          @index.add(i.to_u64, vec)
        end
      end

      @skip_index = skip_index

      # Optionally build a second index with L2-normalized vectors
      if build_norm_index && !skip_index
        print "(building normalized index...) "
        STDOUT.flush

        norm_vecs = Slice(Float32).new(@vocab_size.to_i64 * @dimensions)
        @vocab_size.times do |i|
          vec = vector_for(i)
          n = EmbeddingStore.norm(vec)
          inv_n = n > 1e-10_f32 ? 1.0_f32 / n : 0.0_f32
          offset = i.to_i64 * @dimensions
          @dimensions.times { |d| norm_vecs[offset + d] = vec[d] * inv_n }
        end
        @norm_vectors = norm_vecs

        ni = USearch::Index.new(
          dimensions: @dimensions,
          metric: :cos,
          quantization: :f16,
          connectivity: 32,
          expansion_add: 128,
          expansion_search: 256,
        )
        ni.reserve(@vocab_size)

        @vocab_size.times do |i|
          offset = i.to_i64 * @dimensions
          ni.add(i.to_u64, norm_vecs[offset, @dimensions])
        end
        @norm_index = ni
      end
    end

    # Returns the raw embedding vector for a token ID.
    def vector_for(token_id : Int) : Slice(Float32)
      offset = token_id.to_i64 * @dimensions
      @vectors[offset, @dimensions]
    end

    # Returns the token string for an ID.
    def token_for(token_id : Int) : String
      @vocab[token_id]
    end

    # Returns the L2-normalized embedding vector for a token ID.
    # Only available if build_norm_index was true.
    def norm_vector_for(token_id : Int) : Slice(Float32)
      nv = @norm_vectors
      raise "Normalized vectors not available (pass build_norm_index: true)" unless nv
      offset = token_id.to_i64 * @dimensions
      nv[offset, @dimensions]
    end

    # Whether a normalized index is available.
    def has_norm_index? : Bool
      !@norm_index.nil?
    end

    # Finds the k nearest tokens to a query vector.
    def search(query : Array(Float32) | Slice(Float32), k : Int = 1) : Array(USearch::SearchResult)
      if @skip_index
        brute_force_search(query, k)
      else
        @index.search(query, k)
      end
    end

    # Finds the k nearest normalized tokens to a query vector.
    def search_normalized(query : Array(Float32) | Slice(Float32), k : Int = 1) : Array(USearch::SearchResult)
      ni = @norm_index
      raise "Normalized index not available (pass build_norm_index: true)" unless ni
      ni.search(query, k)
    end

    # Linear scan nearest-neighbor search (no index needed).
    private def brute_force_search(query : Array(Float32) | Slice(Float32), k : Int) : Array(USearch::SearchResult)
      # Compute cosine distance to every token
      query_norm = EmbeddingStore.norm(query)
      return [] of USearch::SearchResult if query_norm < 1e-10

      # Use a simple top-k selection
      results = Array({UInt64, Float32}).new(k) { {0_u64, Float32::MAX} }
      worst_dist = Float32::MAX

      @vocab_size.times do |i|
        vec = vector_for(i)
        # Cosine distance = 1 - cosine_similarity
        dot = 0.0_f32
        vec_norm_sq = 0.0_f32
        @dimensions.times do |d|
          dot += query.unsafe_fetch(d) * vec.unsafe_fetch(d)
          vec_norm_sq += vec.unsafe_fetch(d) * vec.unsafe_fetch(d)
        end
        vec_norm = Math.sqrt(vec_norm_sq).to_f32
        next if vec_norm < 1e-10_f32

        cos_dist = 1.0_f32 - dot / (query_norm * vec_norm)

        if cos_dist < worst_dist
          # Replace the worst entry
          worst_idx = 0
          results.each_with_index do |(_, d), j|
            if d > results[worst_idx][1]
              worst_idx = j
            end
          end
          results[worst_idx] = {i.to_u64, cos_dist}
          worst_dist = results.max_of { |_, d| d }
        end
      end

      results.sort_by! { |_, d| d }
      results.map { |key, dist| USearch::SearchResult.new(key, dist) }
    end

    # Computes the L2 (Euclidean) norm of a vector.
    def self.norm(vec : Array(Float32) | Slice(Float32)) : Float32
      sum = 0.0_f32
      vec.each { |v| sum += v * v }
      Math.sqrt(sum).to_f32
    end

    # Subtracts vector b from vector a, returning a new array.
    def self.subtract(a : Array(Float32) | Slice(Float32), b : Array(Float32) | Slice(Float32)) : Array(Float32)
      Array(Float32).new(a.size) { |i| a[i] - b[i] }
    end

    # Adds vector b to vector a, returning a new array.
    def self.add(a : Array(Float32) | Slice(Float32), b : Array(Float32) | Slice(Float32)) : Array(Float32)
      Array(Float32).new(a.size) { |i| a[i] + b[i] }
    end

    def close
      @index.close
      @norm_index.try(&.close)
    end
  end

  # Loads GPT-2's positional embedding matrix (wpe).
  class PositionalStore
    getter max_positions : Int32
    getter dimensions : Int32

    @vectors : Slice(Float32)

    def initialize(positions_path : String)
      f = File.open(positions_path, "rb")
      begin
        header = Bytes.new(8)
        f.read(header)
        @max_positions = IO::ByteFormat::LittleEndian.decode(Int32, header[0, 4])
        @dimensions = IO::ByteFormat::LittleEndian.decode(Int32, header[4, 4])

        total_floats = @max_positions.to_i64 * @dimensions
        @vectors = Slice(Float32).new(total_floats)
        f.read_fully(@vectors.unsafe_slice_of(UInt8))
      ensure
        f.close
      end
    end

    # Returns the positional embedding for a given position.
    def vector_for(position : Int) : Slice(Float32)
      raise "Position #{position} out of range (max #{@max_positions})" if position >= @max_positions
      offset = position.to_i64 * @dimensions
      @vectors[offset, @dimensions]
    end
  end
end
