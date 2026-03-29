require "json"

module Semtrace
  # GPT-2 BPE tokenizer.
  #
  # Splits text into tokens matching GPT-2's vocabulary using byte-pair encoding.
  class Tokenizer
    @encoder : Hash(String, Int32)  # token string -> token ID
    @merges : Array({String, String})
    @bpe_ranks : Hash({String, String}, Int32)
    @byte_encoder : Hash(UInt8, Char)

    def initialize(vocab_path : String, merges_path : String)
      # Load vocab (our format: id -> decoded token string)
      # We need the inverse: encoded token string -> id
      vocab_json = JSON.parse(File.read(vocab_path))
      byte_decoder = Prepare.build_byte_decoder
      byte_encoder = byte_decoder.invert

      @byte_encoder = byte_encoder
      @encoder = Hash(String, Int32).new

      vocab_json.as_h.each do |id_str, token_val|
        decoded_str = token_val.as_s
        # Re-encode back to BPE form for matching
        encoded = encode_to_bpe_form(decoded_str, byte_encoder)
        @encoder[encoded] = id_str.to_i
      end

      # Load BPE merges
      lines = File.read_lines(merges_path)
      @merges = [] of {String, String}
      @bpe_ranks = Hash({String, String}, Int32).new

      lines.each_with_index do |line, i|
        next if line.starts_with?("#") || line.strip.empty?
        parts = line.split
        next unless parts.size == 2
        pair = {parts[0], parts[1]}
        @merges << pair
        @bpe_ranks[pair] = @bpe_ranks.size
      end
    end

    # Tokenizes text into token IDs.
    def encode(text : String) : Array(Int32)
      tokens = [] of Int32
      bpe_tokens(text).each do |token_str|
        if id = @encoder[token_str]?
          tokens << id
        end
      end
      tokens
    end

    # Tokenizes text into token strings (BPE-encoded form).
    def tokenize(text : String) : Array(String)
      bpe_tokens(text)
    end

    private def bpe_tokens(text : String) : Array(String)
      result = [] of String

      # Split text into words (GPT-2 pattern: optional space + word chars or punctuation)
      # Simplified pattern matching GPT-2's regex
      words = split_into_words(text)

      words.each do |word|
        # Encode each byte to its BPE unicode character
        encoded = String.build do |s|
          word.each_byte do |b|
            if ch = @byte_encoder[b]?
              s << ch
            else
              s << b.chr
            end
          end
        end

        # Apply BPE merges
        bpe_result = apply_bpe(encoded)
        result.concat(bpe_result)
      end

      result
    end

    # Splits text into "words" following GPT-2's tokenization pattern.
    # Each word includes its leading space if present.
    private def split_into_words(text : String) : Array(String)
      words = [] of String
      i = 0
      chars = text

      while i < chars.size
        word = String.build do |s|
          # Include leading space as part of the word
          if chars[i] == ' ' && i + 1 < chars.size
            s << chars[i]
            i += 1
          end

          if i < chars.size
            if chars[i].letter? || chars[i] == '\''
              # Collect word characters (letters and apostrophes)
              while i < chars.size && (chars[i].letter? || chars[i] == '\'')
                s << chars[i]
                i += 1
              end
            elsif chars[i].number?
              # Collect digits
              while i < chars.size && chars[i].number?
                s << chars[i]
                i += 1
              end
            elsif !chars[i].whitespace?
              # Single punctuation/symbol character
              s << chars[i]
              i += 1
            else
              # Standalone whitespace (not leading space)
              s << chars[i]
              i += 1
            end
          end
        end

        words << word unless word.empty?
      end

      words
    end

    # Applies BPE merges to a single encoded word, returning BPE tokens.
    private def apply_bpe(word : String) : Array(String)
      # Start with individual characters as tokens
      pieces = word.chars.map(&.to_s)
      return pieces if pieces.size <= 1

      loop do
        # Find the highest-priority merge pair
        best_pair : {String, String}? = nil
        best_rank = Int32::MAX

        (pieces.size - 1).times do |i|
          pair = {pieces[i], pieces[i + 1]}
          if rank = @bpe_ranks[pair]?
            if rank < best_rank
              best_rank = rank
              best_pair = pair
            end
          end
        end

        break unless best_pair

        # Apply the merge
        merged = best_pair.not_nil!
        new_pieces = [] of String
        i = 0
        while i < pieces.size
          if i < pieces.size - 1 && pieces[i] == merged[0] && pieces[i + 1] == merged[1]
            new_pieces << (merged[0] + merged[1])
            i += 2
          else
            new_pieces << pieces[i]
            i += 1
          end
        end

        pieces = new_pieces
        break if pieces.size == 1
      end

      pieces
    end

    private def encode_to_bpe_form(decoded : String, byte_encoder : Hash(UInt8, Char)) : String
      String.build do |s|
        decoded.each_byte do |b|
          if ch = byte_encoder[b]?
            s << ch
          else
            s << b.chr
          end
        end
      end
    end
  end
end
