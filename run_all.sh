#!/bin/bash
# Run all experiments and capture output.
# Usage: ./run_all.sh [experiment_number]
# Examples:
#   ./run_all.sh          # run all
#   ./run_all.sh 001      # run just experiment 001
#   ./run_all.sh 001 003  # run 001 and 003

set -e
cd "$(dirname "$0")"

run_crystal() {
  local dir="$1" name="$2" args="$3"
  # Only rebuild if source is newer than binary
  if [ ! -f "bin/$name" ] || [ "$dir/run.cr" -nt "bin/$name" ] || [ "src/semtrace.cr" -nt "bin/$name" ]; then
    echo "Building $name..."
    crystal build "$dir/run.cr" -o "bin/$name" --release 2>/dev/null
  fi
  echo "Running $name..."
  eval "bin/$name $args"
}

# Experiment 001: Gettysburg Address
run_001() {
  echo "=== Experiment 001: Gettysburg Address ==="
  echo "Date: $(date)"
  run_crystal experiments/001-gettysburg-address exp001 "--builtin gettysburg"
  echo; echo "--- GPT-2 Medium ---"
  bin/exp001 --data data/gpt2-medium --builtin gettysburg
  echo; echo "--- GPT-2 Large ---"
  bin/exp001 --data data/gpt2-large --builtin gettysburg
  echo; echo "--- GPT-2 XL ---"
  bin/exp001 --data data/gpt2-xl --builtin gettysburg
}

# Experiment 002: Mary Had a Little Lamb
run_002() {
  echo "=== Experiment 002: Mary Had a Little Lamb ==="
  echo "Date: $(date)"
  run_crystal experiments/002-mary-had-a-little-lamb exp002 "--builtin maryhadalittlelamb"
  echo; echo "--- GPT-2 Medium ---"
  bin/exp002 --data data/gpt2-medium --builtin maryhadalittlelamb
  echo; echo "--- GPT-2 Large ---"
  bin/exp002 --data data/gpt2-large --builtin maryhadalittlelamb
  echo; echo "--- GPT-2 XL ---"
  bin/exp002 --data data/gpt2-xl --builtin maryhadalittlelamb
}

# Experiment 003: Tale of Two Cities
run_003() {
  echo "=== Experiment 003: Tale of Two Cities ==="
  echo "Date: $(date)"
  run_crystal experiments/003-tale-of-two-cities exp003 "--file experiments/texts/tale-ch1.txt"
  echo; echo "--- GPT-2 XL ---"
  bin/exp003 --data data/gpt2-xl --file experiments/texts/tale-ch1.txt
}

# Experiment 007: Metric Comparison (brute-force, slow)
run_007() {
  echo "=== Experiment 007: Metric Comparison ==="
  echo "Date: $(date)"
  run_crystal experiments/007-metric-comparison exp007 "--data data/gpt2-xl --builtin gettysburg"
}

# Experiment 009: Union of Metrics (brute-force, slow)
run_009() {
  echo "=== Experiment 009: Union of Metrics ==="
  echo "Date: $(date)"
  run_crystal experiments/009-union-metrics exp009 "--data data/gpt2-xl --file experiments/texts/gettysburg.txt"
}

# Experiment 010: Contextual Embeddings (Python)
run_010() {
  echo "=== Experiment 010: Contextual Embeddings ==="
  echo "Date: $(date)"
  echo "Step 1: Build contextual vocabularies"
  cd experiments/contextual
  python3 build_ctx_vocab.py --layers 6,12
  echo
  echo "Step 2: Decompose"
  python3 decompose.py --text "the cat sat on the mat"
  cd ../..
}

# Experiment 011: Attention Bias (Python, inline)
run_011() {
  echo "=== Experiment 011: Attention Bias Subtraction ==="
  echo "Date: $(date)"
  cd experiments/contextual
  python3 linear_map.py --text "the cat sat on the mat" --layer 6
  cd ../..
}

# Experiment 012: Coordinate Descent (Python)
run_012() {
  echo "=== Experiment 012: Coordinate Descent ==="
  echo "Date: $(date)"
  cd experiments/contextual
  python3 coord_descent.py --text "the cat sat on the mat" --mode both
  echo
  echo "--- Gettysburg Address (static only) ---"
  python3 coord_descent.py --file ../texts/gettysburg.txt --mode static
  cd ../..
}

# Main: run selected or all
if [ $# -eq 0 ]; then
  experiments="001 002 003 007 009 010 011 012"
else
  experiments="$@"
fi

for exp in $experiments; do
  outdir="experiments/${exp}-*/output.txt"
  # Find the actual directory
  expdir=$(ls -d experiments/${exp}-* 2>/dev/null | head -1)
  if [ -z "$expdir" ]; then
    echo "Unknown experiment: $exp"
    continue
  fi
  outfile="$expdir/output.txt"
  echo "Running experiment $exp → $outfile"
  run_${exp} > "$outfile" 2>&1
  echo "  Done. $(wc -l < "$outfile") lines captured."
  echo
done

echo "All done."
