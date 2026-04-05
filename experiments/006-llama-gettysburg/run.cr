require "../../src/semtrace"
require "json"

# Experiment 006: Llama 3.2 3B with HNSW approximate search
#
# Demonstrates that HNSW approximate search significantly underperforms
# brute-force on large vocabularies (128K tokens). Compare with
# Experiment 008 (brute-force) which achieves 100%.
#
# Usage:
#   bin/exp006 --data data/llama-3-2-3b-instruct --ids data/llama-3-2-3b-instruct/gettysburg_ids.json

data_dir = ""
ids_path = ""

i = 0
while i < ARGV.size
  case ARGV[i]
  when "--data" then data_dir = ARGV[i + 1]; i += 2
  when "--ids"  then ids_path = ARGV[i + 1]; i += 2
  else i += 1
  end
end

data_dir = (Path[__DIR__].parent.parent / "data" / "llama-3-2-3b-instruct").to_s if data_dir.empty?
ids_path = File.join(data_dir, "gettysburg_ids.json") if ids_path.empty?

embeddings_path = File.join(data_dir, "embeddings.bin")
vocab_path = File.join(data_dir, "vocab.json")
abort "Missing data in #{data_dir}" unless File.exists?(embeddings_path) && File.exists?(vocab_path)
abort "Missing IDs file: #{ids_path}" unless File.exists?(ids_path)

# Load WITH HNSW index (this is the point — testing approximate search)
print "Loading Llama (with HNSW index)... "
store = Semtrace::EmbeddingStore.new(embeddings_path, vocab_path)
puts "#{store.vocab_size} tokens x #{store.dimensions}d"

decomposer = Semtrace::Decomposer.new(store)

# Load pre-tokenized IDs
ids = Array(Int32).new
JSON.parse(File.read(ids_path)).as_a.each { |v| ids << v.as_i.to_i32 }
unique_ids = ids.to_set

puts "\n=== Experiment 006: Llama HNSW Decomposition ==="
puts "  Tokens: #{ids.size}, Unique: #{unique_ids.size}"
puts "  Search: HNSW approximate (cosine, f16)"

# Build target
target = Array(Float32).new(store.dimensions, 0.0_f32)
ids.each { |id| target = Semtrace::EmbeddingStore.add(target, store.vector_for(id)) }

# Decompose via HNSW
result = decomposer.decompose(target, max_steps: ids.size + 20)

recovered_set = result.token_ids.to_set
exact = (unique_ids & recovered_set).size

puts "\n--- Results (HNSW) ---"
puts "  Unique recovery: #{exact}/#{unique_ids.size} (#{"%.1f" % (exact * 100.0 / unique_ids.size)}%)"
puts "  Recovered tokens: #{result.tokens.size}"
puts "  Final residual: #{"%.4f" % result.final_residual_norm}"

missing = unique_ids - recovered_set
if missing.size <= 20
  puts "\n  Missing unique tokens (#{missing.size}):"
  missing.each { |id| puts "    #{store.token_for(id).inspect}" }
end

puts "\n  NOTE: Compare with Experiment 008 (brute-force) which achieves"
puts "  100% on the same data. The difference is entirely due to HNSW"
puts "  approximation error at 128K vocabulary / 3072 dimensions."

store.close
