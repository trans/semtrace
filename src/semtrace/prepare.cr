require "http/client"
require "json"
require "uri"

module Semtrace
  # Downloads GPT-2 weights from HuggingFace and extracts the embedding matrix.
  module Prepare
    GPT2_MODELS = {
      "gpt2"        => {repo: "openai-community/gpt2", label: "GPT-2 Small (768d)"},
      "gpt2-medium" => {repo: "openai-community/gpt2-medium", label: "GPT-2 Medium (1024d)"},
      "gpt2-large"  => {repo: "openai-community/gpt2-large", label: "GPT-2 Large (1280d)"},
      "gpt2-xl"     => {repo: "openai-community/gpt2-xl", label: "GPT-2 XL (1600d)"},
    }

    HF_BASE = "https://huggingface.co/openai-community/gpt2/resolve/main"

    SAFETENSORS_URL = "#{HF_BASE}/model.safetensors"
    VOCAB_URL       = "#{HF_BASE}/vocab.json"
    MERGES_URL      = "#{HF_BASE}/merges.txt"

    # Downloads a URL to a local file, following redirects.
    # Shows progress for large files.
    def self.download(url : String, dest : String)
      uri = URI.parse(url)
      max_redirects = 5

      max_redirects.times do
        client = HTTP::Client.new(uri)
        client.connect_timeout = 30.seconds
        client.read_timeout = 300.seconds

        begin
          client.get(uri.request_target) do |response|
            case response.status_code
            when 200
              total = response.headers["Content-Length"]?.try(&.to_i64) || 0_i64
              downloaded = 0_i64
              last_pct = -1

              File.open(dest, "wb") do |f|
                buf = Bytes.new(256 * 1024) # 256KB chunks
                while (n = response.body_io.read(buf)) > 0
                  f.write(buf[0, n])
                  downloaded += n
                  if total > 0
                    pct = (downloaded * 100 / total).to_i
                    if pct != last_pct && pct % 5 == 0
                      print "\r  #{downloaded // (1024 * 1024)} / #{total // (1024 * 1024)} MB (#{pct}%)"
                      STDOUT.flush
                      last_pct = pct
                    end
                  end
                end
              end
              puts "\r  #{downloaded // (1024 * 1024)} MB — done.            "
              return
            when 301, 302, 303, 307, 308
              location = response.headers["Location"]
              # Drain the body before closing
              response.body_io.skip_to_end
              new_uri = URI.parse(location)
              # Handle relative redirects
              if new_uri.scheme.nil? || new_uri.scheme.try(&.empty?)
                new_uri.scheme = uri.scheme
                new_uri.host = uri.host
                new_uri.port = uri.port
              end
              uri = new_uri
            else
              raise "HTTP #{response.status_code} downloading #{url}"
            end
          end
        ensure
          client.close
        end
      end

      raise "Too many redirects downloading #{url}"
    end

    # Downloads GPT-2 files and extracts the embedding matrix to our binary format.
    def self.run(data_dir : String, model : String = "gpt2")
      info = GPT2_MODELS[model]?
      unless info
        STDERR.puts "Unknown model: #{model}"
        STDERR.puts "Available: #{GPT2_MODELS.keys.join(", ")}"
        exit 1
      end

      puts "Preparing #{info[:label]}..."
      hf_base = "https://huggingface.co/#{info[:repo]}/resolve/main"

      # Use model-specific subdirectory for non-default models
      out_dir = model == "gpt2" ? data_dir : File.join(data_dir, model)
      Dir.mkdir_p(out_dir)

      safetensors_path = File.join(out_dir, "model.safetensors")
      hf_vocab_path = File.join(out_dir, "hf_vocab.json")
      hf_merges_path = File.join(out_dir, "merges.txt")
      embeddings_path = File.join(out_dir, "embeddings.bin")
      positions_path = File.join(out_dir, "positions.bin")
      vocab_path = File.join(out_dir, "vocab.json")
      merges_path = File.join(out_dir, "merges.txt")

      # Download safetensors if not cached
      unless File.exists?(safetensors_path)
        puts "Downloading model weights..."
        download("#{hf_base}/model.safetensors", safetensors_path)
      else
        puts "Using cached #{safetensors_path}"
      end

      # Download vocab if not cached
      unless File.exists?(hf_vocab_path)
        puts "Downloading vocabulary..."
        download("#{hf_base}/vocab.json", hf_vocab_path)
      else
        puts "Using cached #{hf_vocab_path}"
      end

      # Download merges if not cached
      unless File.exists?(hf_merges_path)
        puts "Downloading BPE merges..."
        download("#{hf_base}/merges.txt", hf_merges_path)
      else
        puts "Using cached #{hf_merges_path}"
      end

      # Parse safetensors to find wte.weight
      puts "Parsing safetensors header..."
      st_info = Safetensors.read_header(safetensors_path)
      wte = st_info[:tensors].find { |t| t.name == "wte.weight" }
      raise "wte.weight tensor not found in safetensors file" unless wte

      vocab_size = wte.shape[0].to_i32
      dimensions = wte.shape[1].to_i32
      puts "Found wte.weight: #{vocab_size} x #{dimensions} (#{wte.dtype})"

      # Read the tensor data
      puts "Extracting embedding matrix..."
      data = Safetensors.read_tensor_f32(safetensors_path, wte)

      # Write our binary format: header (vocab_size u32 LE, dims u32 LE) + float32 data
      File.open(embeddings_path, "wb") do |f|
        header = Bytes.new(8)
        IO::ByteFormat::LittleEndian.encode(vocab_size, header[0, 4])
        IO::ByteFormat::LittleEndian.encode(dimensions, header[4, 4])
        f.write(header)
        f.write(data.unsafe_slice_of(UInt8))
      end

      size_mb = File.size(embeddings_path) / (1024.0 * 1024.0)
      puts "Wrote #{embeddings_path} (#{"%.1f" % size_mb} MB)"

      # Extract positional embeddings (wpe)
      wpe = st_info[:tensors].find { |t| t.name == "wpe.weight" }
      raise "wpe.weight tensor not found in safetensors file" unless wpe

      max_positions = wpe.shape[0].to_i32
      puts "Found wpe.weight: #{max_positions} x #{wpe.shape[1]} (#{wpe.dtype})"

      puts "Extracting positional embeddings..."
      pos_data = Safetensors.read_tensor_f32(safetensors_path, wpe)

      File.open(positions_path, "wb") do |f|
        header = Bytes.new(8)
        IO::ByteFormat::LittleEndian.encode(max_positions, header[0, 4])
        IO::ByteFormat::LittleEndian.encode(dimensions, header[4, 4])
        f.write(header)
        f.write(pos_data.unsafe_slice_of(UInt8))
      end

      pos_mb = File.size(positions_path) / (1024.0 * 1024.0)
      puts "Wrote #{positions_path} (#{"%.1f" % pos_mb} MB)"

      # Invert HuggingFace vocab (string -> id) to our format (id -> string),
      # and decode GPT-2's byte-level BPE encoding back to readable strings.
      puts "Inverting and decoding vocabulary..."
      byte_decoder = build_byte_decoder
      hf_vocab = JSON.parse(File.read(hf_vocab_path))
      inverted = Hash(String, String).new
      hf_vocab.as_h.each do |token_str, token_id|
        inverted[token_id.to_s] = decode_bpe_token(token_str, byte_decoder)
      end

      File.write(vocab_path, inverted.to_json)
      puts "Wrote #{vocab_path} (#{inverted.size} tokens)"

      # Keep merges.txt (it's small, needed for tokenization)
      if hf_merges_path != merges_path && File.exists?(hf_merges_path)
        File.rename(hf_merges_path, merges_path) unless File.exists?(merges_path)
      end

      # Clean up the large safetensors file
      puts "Cleaning up downloaded model file..."
      File.delete(safetensors_path) if File.exists?(safetensors_path)
      File.delete(hf_vocab_path) if File.exists?(hf_vocab_path)

      if model == "gpt2"
        puts "Done. Ready to run: bin/semtrace"
      else
        puts "Done. Ready to run: bin/semtrace --data #{out_dir}"
      end
    end

    # GPT-2 maps raw bytes to unicode characters so the BPE vocabulary only
    # contains printable strings. This builds the reverse mapping:
    # unicode codepoint -> original byte value.
    protected def self.build_byte_decoder : Hash(Char, UInt8)
      # Printable ranges that map to themselves: '!'..'~', '¡'..'¬', '®'..'ÿ'
      bs = [] of Int32
      ('!'.ord..'~'.ord).each { |b| bs << b }
      ('¡'.ord..'¬'.ord).each { |b| bs << b }
      ('®'.ord..'ÿ'.ord).each { |b| bs << b }

      cs = bs.map(&.chr)

      # Everything else (space, control chars, etc.) maps to 256+n
      n = 0
      256.times do |b|
        unless bs.includes?(b)
          bs << b
          cs << (256 + n).chr
          n += 1
        end
      end

      # Invert: unicode char -> byte value
      decoder = Hash(Char, UInt8).new
      bs.each_with_index do |byte_val, i|
        decoder[cs[i]] = byte_val.to_u8
      end
      decoder
    end

    # Decodes a GPT-2 BPE token string back to its actual UTF-8 representation.
    private def self.decode_bpe_token(encoded : String, decoder : Hash(Char, UInt8)) : String
      bytes = [] of UInt8
      encoded.each_char do |ch|
        if b = decoder[ch]?
          bytes << b
        else
          # Unknown char — keep its UTF-8 bytes as-is
          ch.to_s.each_byte { |byte| bytes << byte }
        end
      end
      String.new(Slice.new(bytes.to_unsafe, bytes.size))
    end
  end
end
