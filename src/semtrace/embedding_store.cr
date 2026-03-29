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
    @vectors : Slice(Float32)

    def initialize(embeddings_path : String, vocab_path : String)
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

      # Build USearch HNSW index
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

    # Returns the raw embedding vector for a token ID.
    def vector_for(token_id : Int) : Slice(Float32)
      offset = token_id.to_i64 * @dimensions
      @vectors[offset, @dimensions]
    end

    # Returns the token string for an ID.
    def token_for(token_id : Int) : String
      @vocab[token_id]
    end

    # Finds the k nearest tokens to a query vector.
    def search(query : Array(Float32) | Slice(Float32), k : Int = 1) : Array(USearch::SearchResult)
      @index.search(query, k)
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
