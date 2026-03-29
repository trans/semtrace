require "json"

module Semtrace
  # Minimal parser for the safetensors binary format.
  #
  # Format:
  #   Bytes 0..7  : u64 LE — header size in bytes
  #   Bytes 8..8+N: JSON header — maps tensor names to {dtype, shape, data_offsets}
  #   Bytes 8+N.. : raw tensor data (offsets are relative to this point)
  module Safetensors
    record TensorInfo,
      name : String,
      dtype : String,
      shape : Array(Int64),
      offset_start : Int64,
      offset_end : Int64

    # Parses the header of a safetensors file and returns tensor metadata.
    def self.read_header(path : String) : {header_size: Int64, tensors: Array(TensorInfo)}
      File.open(path, "rb") do |f|
        # Read header size (u64 LE)
        buf = Bytes.new(8)
        f.read_fully(buf)
        header_size = IO::ByteFormat::LittleEndian.decode(Int64, buf)

        # Read and parse JSON header
        header_bytes = Bytes.new(header_size)
        f.read_fully(header_bytes)
        header_json = JSON.parse(String.new(header_bytes))

        tensors = [] of TensorInfo
        header_json.as_h.each do |name, info|
          next if name == "__metadata__"
          obj = info.as_h
          dtype = obj["dtype"].as_s
          shape = obj["shape"].as_a.map(&.as_i64)
          offsets = obj["data_offsets"].as_a
          tensors << TensorInfo.new(
            name: name,
            dtype: dtype,
            shape: shape,
            offset_start: offsets[0].as_i64,
            offset_end: offsets[1].as_i64,
          )
        end

        {header_size: header_size, tensors: tensors}
      end
    end

    # Reads a specific tensor's raw data as Float32 values.
    # Supports F32 and F16 dtypes (F16 is converted to F32).
    def self.read_tensor_f32(path : String, tensor : TensorInfo) : Slice(Float32)
      data_offset = 8_i64 + read_header_size(path)

      File.open(path, "rb") do |f|
        f.seek(data_offset + tensor.offset_start)
        byte_count = tensor.offset_end - tensor.offset_start

        case tensor.dtype
        when "F32"
          num_floats = byte_count // 4
          result = Slice(Float32).new(num_floats)
          f.read_fully(result.unsafe_slice_of(UInt8))
          result
        when "F16"
          num_floats = byte_count // 2
          raw = Bytes.new(byte_count)
          f.read_fully(raw)
          result = Slice(Float32).new(num_floats)
          num_floats.times do |i|
            result[i] = f16_to_f32(raw[i * 2], raw[i * 2 + 1])
          end
          result
        else
          raise "Unsupported dtype: #{tensor.dtype}"
        end
      end
    end

    private def self.read_header_size(path : String) : Int64
      File.open(path, "rb") do |f|
        buf = Bytes.new(8)
        f.read_fully(buf)
        IO::ByteFormat::LittleEndian.decode(Int64, buf)
      end
    end

    # Converts IEEE 754 half-precision (F16) to single-precision (F32).
    private def self.f16_to_f32(lo : UInt8, hi : UInt8) : Float32
      bits = lo.to_u16 | (hi.to_u16 << 8)
      sign = (bits >> 15) & 0x1
      exponent = (bits >> 10) & 0x1f
      mantissa = bits & 0x3ff

      if exponent == 0
        if mantissa == 0
          f32_bits = sign.to_u32 << 31
        else
          # Subnormal: normalize
          e = -1
          m = mantissa
          while (m & 0x400) == 0
            m <<= 1
            e -= 1
          end
          m &= 0x3ff
          f32_bits = (sign.to_u32 << 31) | ((127 + e).to_u32 << 23) | (m.to_u32 << 13)
        end
      elsif exponent == 0x1f
        # Inf or NaN
        f32_bits = (sign.to_u32 << 31) | (0xff_u32 << 23) | (mantissa.to_u32 << 13)
      else
        f32_bits = (sign.to_u32 << 31) | ((exponent.to_u32 - 15 + 127) << 23) | (mantissa.to_u32 << 13)
      end

      f32_bits.unsafe_as(Float32)
    end
  end
end
