module Semtrace
  # Minimal GGUF parser for extracting tensor data from Ollama model files.
  #
  # GGUF format: header → metadata KV pairs → tensor infos → aligned tensor data.
  # Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
  module GGUF
    MAGIC = 0x46554747_u32 # "GGUF" in little-endian

    enum GGMLType : UInt32
      F32  =  0
      F16  =  1
      Q4_0 =  2
      Q4_1 =  3
      Q5_0 =  6
      Q5_1 =  7
      Q8_0 =  8
      Q8_1 =  9
      Q2_K = 10
      Q3_K = 11
      Q4_K = 12
      Q5_K = 13
      Q6_K = 14
      BF16 = 30
    end

    enum ValueType : UInt32
      UINT8   =  0
      INT8    =  1
      UINT16  =  2
      INT16   =  3
      UINT32  =  4
      INT32   =  5
      FLOAT32 =  6
      BOOL    =  7
      STRING  =  8
      ARRAY   =  9
      UINT64  = 10
      INT64   = 11
      FLOAT64 = 12
    end

    record TensorInfo,
      name : String,
      shape : Array(UInt64),
      type : GGMLType,
      offset : UInt64

    record Header,
      version : UInt32,
      n_tensors : Int64,
      n_kv : Int64

    # Reads the GGUF header, metadata, and tensor info without loading tensor data.
    def self.read_info(path : String) : {header: Header, metadata: Hash(String, String | Int64 | UInt64 | Float64 | Bool), tensors: Array(TensorInfo), data_offset: Int64}
      metadata = Hash(String, String | Int64 | UInt64 | Float64 | Bool).new
      tensors = [] of TensorInfo

      File.open(path, "rb") do |f|
        # Header
        magic = read_u32(f)
        raise "Not a GGUF file (magic: 0x#{magic.to_s(16)})" unless magic == MAGIC

        version = read_u32(f)
        raise "Unsupported GGUF version: #{version}" unless version >= 2

        n_tensors = read_i64(f)
        n_kv = read_i64(f)
        header = Header.new(version, n_tensors, n_kv)

        # Metadata
        n_kv.times do
          key = read_string(f)
          value = read_value(f)
          case value
          when String, Int64, UInt64, Float64, Bool
            metadata[key] = value
          end
        end

        # Tensor infos
        n_tensors.times do
          name = read_string(f)
          ndim = read_u32(f)
          shape = Array(UInt64).new(ndim.to_i) { read_u64(f) }
          type = GGMLType.new(read_u32(f))
          offset = read_u64(f)
          tensors << TensorInfo.new(name, shape, type, offset)
        end

        # Calculate data section offset (aligned)
        alignment = (metadata["general.alignment"]?.try(&.as(UInt64 | Int64).to_u64) || 32_u64)
        current_pos = f.pos
        data_offset = align(current_pos.to_i64, alignment.to_i64)

        {header: header, metadata: metadata, tensors: tensors, data_offset: data_offset}
      end
    end

    # Reads a tensor's data and dequantizes to Float32.
    def self.read_tensor_f32(path : String, tensor : TensorInfo, data_offset : Int64) : Slice(Float32)
      num_elements = tensor.shape.reduce(1_i64) { |a, b| a * b.to_i64 }

      File.open(path, "rb") do |f|
        f.seek(data_offset + tensor.offset.to_i64)

        case tensor.type
        when .f32?
          result = Slice(Float32).new(num_elements)
          f.read_fully(result.unsafe_slice_of(UInt8))
          result
        when .f16?
          raw = Bytes.new(num_elements * 2)
          f.read_fully(raw)
          result = Slice(Float32).new(num_elements)
          num_elements.times { |i| result[i] = f16_to_f32(raw[i * 2], raw[i * 2 + 1]) }
          result
        when .bf16?
          raw = Bytes.new(num_elements * 2)
          f.read_fully(raw)
          result = Slice(Float32).new(num_elements)
          num_elements.times { |i| result[i] = bf16_to_f32(raw[i * 2], raw[i * 2 + 1]) }
          result
        when .q8_0?
          dequantize_q8_0(f, num_elements.to_u64)
        when .q4_0?
          dequantize_q4_0(f, num_elements.to_u64)
        when .q4_1?
          dequantize_q4_1(f, num_elements.to_u64)
        when .q6_k?
          dequantize_q6_k(f, num_elements.to_u64)
        else
          raise "Unsupported quantization type for dequantization: #{tensor.type}"
        end
      end
    end

    # Q8_0: blocks of 32, each block = 2 bytes (f16 scale) + 32 bytes (int8 values)
    private def self.dequantize_q8_0(f : IO, num_elements : UInt64) : Slice(Float32)
      block_size = 32_u64
      num_blocks = (num_elements + block_size - 1) // block_size
      result = Slice(Float32).new(num_elements)
      scale_bytes = Bytes.new(2)
      quants = Bytes.new(32)

      num_blocks.times do |b|
        f.read_fully(scale_bytes)
        scale = f16_to_f32(scale_bytes[0], scale_bytes[1])
        f.read_fully(quants)

        block_size.times do |i|
          idx = b * block_size + i
          break if idx >= num_elements
          q = quants[i].to_i8!
          result[idx] = scale * q.to_f32
        end
      end
      result
    end

    # Q4_0: blocks of 32, each block = 2 bytes (f16 scale) + 16 bytes (32 x 4-bit packed)
    private def self.dequantize_q4_0(f : IO, num_elements : UInt64) : Slice(Float32)
      block_size = 32_u64
      num_blocks = (num_elements + block_size - 1) // block_size
      result = Slice(Float32).new(num_elements)
      scale_bytes = Bytes.new(2)
      quants = Bytes.new(16)

      num_blocks.times do |b|
        f.read_fully(scale_bytes)
        scale = f16_to_f32(scale_bytes[0], scale_bytes[1])
        f.read_fully(quants)

        16.times do |j|
          byte = quants[j]
          lo = (byte & 0x0F).to_i32 - 8 # low nibble, signed offset
          hi = (byte >> 4).to_i32 - 8    # high nibble, signed offset

          idx0 = b * block_size + j
          idx1 = b * block_size + j + 16
          result[idx0] = scale * lo.to_f32 if idx0 < num_elements
          result[idx1] = scale * hi.to_f32 if idx1 < num_elements
        end
      end
      result
    end

    # Q4_1: blocks of 32, each block = 2 bytes (f16 scale) + 2 bytes (f16 min) + 16 bytes
    private def self.dequantize_q4_1(f : IO, num_elements : UInt64) : Slice(Float32)
      block_size = 32_u64
      num_blocks = (num_elements + block_size - 1) // block_size
      result = Slice(Float32).new(num_elements)
      header = Bytes.new(4)
      quants = Bytes.new(16)

      num_blocks.times do |b|
        f.read_fully(header)
        scale = f16_to_f32(header[0], header[1])
        min = f16_to_f32(header[2], header[3])
        f.read_fully(quants)

        16.times do |j|
          byte = quants[j]
          lo = (byte & 0x0F).to_f32
          hi = (byte >> 4).to_f32

          idx0 = b * block_size + j
          idx1 = b * block_size + j + 16
          result[idx0] = scale * lo + min if idx0 < num_elements
          result[idx1] = scale * hi + min if idx1 < num_elements
        end
      end
      result
    end

    # Q6_K: super-blocks of 256 elements
    # Layout per super-block: 128 bytes ql (low 4 bits), 64 bytes qh (high 2 bits),
    #                         16 bytes scales (int8), 2 bytes d (f16)
    private def self.dequantize_q6_k(f : IO, num_elements : UInt64) : Slice(Float32)
      block_size = 256_u64
      num_blocks = (num_elements + block_size - 1) // block_size
      result = Slice(Float32).new(num_elements)

      ql = Bytes.new(128)
      qh = Bytes.new(64)
      scales = Bytes.new(16)
      d_bytes = Bytes.new(2)

      num_blocks.times do |b|
        f.read_fully(ql)
        f.read_fully(qh)
        f.read_fully(scales)
        f.read_fully(d_bytes)
        d = f16_to_f32(d_bytes[0], d_bytes[1])

        256.times do |i|
          idx = b * block_size + i
          break if idx >= num_elements

          # Extract 6-bit quantized value
          ql_byte = ql[i // 2]
          q4 = if i.even?
                 (ql_byte & 0x0F).to_i32
               else
                 (ql_byte >> 4).to_i32
               end

          qh_byte = qh[i // 4]
          shift = (i % 4) * 2
          q2 = ((qh_byte >> shift) & 0x03).to_i32

          q = (q4 | (q2 << 4)) - 32 # 6-bit signed: 0..63 -> -32..31

          sc = scales[i // 16].to_i8!.to_f32
          result[idx] = d * sc * q.to_f32
        end
      end
      result
    end

    private def self.f16_to_f32(lo : UInt8, hi : UInt8) : Float32
      # Reinterpret via bit manipulation without overflow-prone arithmetic
      bits = lo.to_u32 | (hi.to_u32 << 8)
      sign = (bits >> 15) & 0x1
      exponent = (bits >> 10) & 0x1f
      mantissa = bits & 0x3ff

      if exponent == 0
        if mantissa == 0
          # Zero (positive or negative)
          f32_bits = sign << 31
        else
          # Subnormal f16 -> normalized f32
          # Shift mantissa up until the implicit 1 bit is in position
          e = 0_i32
          m = mantissa
          loop do
            e -= 1
            m <<= 1
            break if (m & 0x400) != 0
          end
          m &= 0x3ff
          f32_exp = (127 - 15 + 1 + e).clamp(0, 255).to_u32
          f32_bits = (sign << 31) | (f32_exp << 23) | (m << 13)
        end
      elsif exponent == 0x1f
        # Inf or NaN
        f32_bits = (sign << 31) | (0xff_u32 << 23) | (mantissa << 13)
      else
        # Normalized
        f32_exp = (exponent.to_i32 - 15 + 127).to_u32
        f32_bits = (sign << 31) | (f32_exp << 23) | (mantissa << 13)
      end

      f32_bits.unsafe_as(Float32)
    end

    private def self.bf16_to_f32(lo : UInt8, hi : UInt8) : Float32
      # BF16 is just the top 16 bits of a float32
      bits = (hi.to_u32 << 24) | (lo.to_u32 << 16)
      bits.unsafe_as(Float32)
    end

    private def self.read_u32(f : IO) : UInt32
      buf = Bytes.new(4)
      f.read_fully(buf)
      IO::ByteFormat::LittleEndian.decode(UInt32, buf)
    end

    private def self.read_i64(f : IO) : Int64
      buf = Bytes.new(8)
      f.read_fully(buf)
      IO::ByteFormat::LittleEndian.decode(Int64, buf)
    end

    private def self.read_u64(f : IO) : UInt64
      buf = Bytes.new(8)
      f.read_fully(buf)
      IO::ByteFormat::LittleEndian.decode(UInt64, buf)
    end

    private def self.read_string(f : IO) : String
      len = read_u64(f)
      buf = Bytes.new(len)
      f.read_fully(buf)
      String.new(buf)
    end

    private def self.read_value(f : IO) : String | Int64 | UInt64 | Float64 | Float32 | Bool | Nil
      vtype = ValueType.new(read_u32(f))
      case vtype
      when .uint8?   then f.read_byte.not_nil!.to_u64
      when .int8?    then f.read_byte.not_nil!.to_i8!.to_i64
      when .uint16?
        buf = Bytes.new(2); f.read_fully(buf)
        IO::ByteFormat::LittleEndian.decode(UInt16, buf).to_u64
      when .int16?
        buf = Bytes.new(2); f.read_fully(buf)
        IO::ByteFormat::LittleEndian.decode(Int16, buf).to_i64
      when .uint32?  then read_u32(f).to_u64
      when .int32?
        buf = Bytes.new(4); f.read_fully(buf)
        IO::ByteFormat::LittleEndian.decode(Int32, buf).to_i64
      when .float32?
        buf = Bytes.new(4); f.read_fully(buf)
        IO::ByteFormat::LittleEndian.decode(Float32, buf).to_f64
      when .bool?    then f.read_byte.not_nil! != 0
      when .string?  then read_string(f)
      when .uint64?  then read_u64(f)
      when .int64?   then read_i64(f)
      when .float64?
        buf = Bytes.new(8); f.read_fully(buf)
        IO::ByteFormat::LittleEndian.decode(Float64, buf)
      when .array?
        # Skip arrays — read past them without storing
        atype = ValueType.new(read_u32(f))
        alen = read_u64(f)
        alen.times { skip_value(f, atype) }
        nil
      else
        nil
      end
    end

    private def self.skip_value(f : IO, vtype : ValueType)
      case vtype
      when .uint8?, .int8?, .bool? then f.skip(1)
      when .uint16?, .int16?       then f.skip(2)
      when .uint32?, .int32?, .float32? then f.skip(4)
      when .uint64?, .int64?, .float64? then f.skip(8)
      when .string?
        len = read_u64(f)
        f.skip(len)
      when .array?
        atype = ValueType.new(read_u32(f))
        alen = read_u64(f)
        alen.times { skip_value(f, atype) }
      end
    end

    private def self.align(pos : Int64, alignment : Int64) : Int64
      ((pos + alignment - 1) // alignment) * alignment
    end
  end
end
