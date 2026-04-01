module Semtrace
  # Result of a greedy residual decomposition.
  record TraceResult,
    tokens : Array(String),
    token_ids : Array(Int32),
    residual_norms : Array(Float32),
    final_residual_norm : Float32

  # Greedy residual decomposition of embedding vectors.
  #
  # Given a target vector v, iteratively finds the nearest token embedding,
  # subtracts it, and repeats until the residual norm falls below epsilon
  # or max_steps is reached.
  class Decomposer
    def initialize(@store : EmbeddingStore)
    end

    # Decomposes a target vector into a sequence of tokens.
    # When lookahead > 1, evaluates the top-k candidates at each step and
    # picks the one that minimizes the residual norm after subtraction.
    def decompose(
      target : Array(Float32) | Slice(Float32),
      epsilon : Float32 = 0.01_f32,
      max_steps : Int32 = 20,
      lookahead : Int32 = 1
    ) : TraceResult
      residual = target.is_a?(Slice) ? target.to_a : target.dup
      tokens = [] of String
      token_ids = [] of Int32
      residual_norms = [] of Float32

      prev_norm = Float32::MAX

      max_steps.times do |step|
        norm = EmbeddingStore.norm(residual)
        residual_norms << norm
        break if norm < epsilon

        # Stop if the residual is growing — we've exhausted the signal
        if norm > prev_norm
          break
        end
        prev_norm = norm

        # Find nearest token(s) to current residual
        candidates = @store.search(residual, k: lookahead)
        break if candidates.empty?

        best = if candidates.size == 1
                 candidates.first
               else
                 # Lookahead: pick the candidate that minimizes residual norm after subtraction
                 candidates.min_by do |c|
                   next_residual = EmbeddingStore.subtract(residual, @store.vector_for(c.key.to_i32))
                   EmbeddingStore.norm(next_residual)
                 end
               end

        best_id = best.key.to_i32
        tokens << @store.token_for(best_id)
        token_ids << best_id
        residual = EmbeddingStore.subtract(residual, @store.vector_for(best_id))
      end

      final_norm = EmbeddingStore.norm(residual)

      TraceResult.new(
        tokens: tokens,
        token_ids: token_ids,
        residual_norms: residual_norms,
        final_residual_norm: final_norm,
      )
    end

    # Composes a vector by summing token embeddings, then decomposes it.
    # This is the round-trip test: does trace(compose(tokens)) ≈ tokens?
    def trace_tokens(token_names : Array(String), **opts) : TraceResult
      # Look up token IDs by string
      composed = Array(Float32).new(@store.dimensions, 0.0_f32)

      token_names.each do |name|
        # Find the token ID by matching vocab
        token_id = @store.vocab.index(name)
        raise "Token not found in vocabulary: #{name.inspect}" unless token_id

        vec = @store.vector_for(token_id)
        composed = EmbeddingStore.add(composed, vec)
      end

      decompose(composed, **opts)
    end

    # Decomposes a single token's embedding (sanity check: should return itself).
    def trace_single(token_name : String, **opts) : TraceResult
      trace_tokens([token_name], **opts)
    end

    # Concept arithmetic: sum positive tokens, subtract negative tokens, decompose.
    # Example: arithmetic(positive: [" king", " woman"], negative: [" man"])
    #   = E(" king") - E(" man") + E(" woman")
    def arithmetic(positive : Array(String), negative : Array(String) = [] of String, **opts) : TraceResult
      vec = Array(Float32).new(@store.dimensions, 0.0_f32)

      positive.each do |name|
        token_id = @store.vocab.index(name)
        raise "Token not found in vocabulary: #{name.inspect}" unless token_id
        vec = EmbeddingStore.add(vec, @store.vector_for(token_id))
      end

      negative.each do |name|
        token_id = @store.vocab.index(name)
        raise "Token not found in vocabulary: #{name.inspect}" unless token_id
        vec = EmbeddingStore.subtract(vec, @store.vector_for(token_id))
      end

      decompose(vec, **opts)
    end

    # Decomposes a bag-of-words sentence embedding (sum of static token embeddings).
    def trace_sentence(token_ids : Array(Int32), **opts) : TraceResult
      vec = Array(Float32).new(@store.dimensions, 0.0_f32)
      token_ids.each do |id|
        vec = EmbeddingStore.add(vec, @store.vector_for(id))
      end
      decompose(vec, **opts)
    end

    # Decomposes using L2-normalized token vectors.
    # Requires the store to have a normalized HNSW index (build_norm_index: true).
    # Searches the normalized index AND subtracts normalized vectors.
    def decompose_normalized(
      target : Array(Float32) | Slice(Float32),
      epsilon : Float32 = 0.01_f32,
      max_steps : Int32 = 20,
      lookahead : Int32 = 1
    ) : TraceResult
      raise "Normalized index not available" unless @store.has_norm_index?

      residual = target.is_a?(Slice) ? target.to_a : target.dup
      tokens = [] of String
      token_ids = [] of Int32
      residual_norms = [] of Float32

      prev_norm = Float32::MAX

      max_steps.times do
        norm = EmbeddingStore.norm(residual)
        residual_norms << norm
        break if norm < epsilon
        break if norm > prev_norm
        prev_norm = norm

        candidates = @store.search_normalized(residual, k: lookahead)
        break if candidates.empty?

        best = if candidates.size == 1
                 candidates.first
               else
                 candidates.min_by do |c|
                   next_residual = EmbeddingStore.subtract(residual, @store.norm_vector_for(c.key.to_i32))
                   EmbeddingStore.norm(next_residual)
                 end
               end

        best_id = best.key.to_i32
        tokens << @store.token_for(best_id)
        token_ids << best_id
        residual = EmbeddingStore.subtract(residual, @store.norm_vector_for(best_id))
      end

      TraceResult.new(
        tokens: tokens,
        token_ids: token_ids,
        residual_norms: residual_norms,
        final_residual_norm: EmbeddingStore.norm(residual),
      )
    end

    # Midpoint between two tokens.
    def midpoint(token_a : String, token_b : String, **opts) : TraceResult
      id_a = @store.vocab.index(token_a)
      id_b = @store.vocab.index(token_b)
      raise "Token not found: #{token_a.inspect}" unless id_a
      raise "Token not found: #{token_b.inspect}" unless id_b

      vec_a = @store.vector_for(id_a)
      vec_b = @store.vector_for(id_b)
      mid = Array(Float32).new(@store.dimensions) { |i| (vec_a[i] + vec_b[i]) / 2.0_f32 }

      decompose(mid, **opts)
    end
  end
end
