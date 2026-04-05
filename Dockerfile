# Semtrace: Embedding Decomposition Research Environment
#
# Build:  podman build -t semtrace .
# Run:    podman run --rm semtrace 001
# GPU:    podman run --rm --device nvidia.com/gpu=all semtrace 012
# Shell:  podman run --rm -it semtrace bash

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Python deps
RUN pip install --no-cache-dir transformers numpy

# Crystal
RUN apt-get update && \
    apt-get install -y curl ca-certificates build-essential libevent-dev libgc-dev libpcre2-dev libxml2-dev libyaml-dev git && \
    curl -fsSL https://crystal-lang.org/install.sh | bash && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /semtrace

# Copy source first (changes less often → better layer caching)
COPY shard.yml shard.lock* ./
RUN shards install

COPY src/ src/
COPY experiments/ experiments/

# Build the CLI
RUN mkdir -p bin && crystal build src/cli.cr -o bin/semtrace --release

# Download GPT-2 Small embeddings (smallest model, ~150MB)
RUN bin/semtrace prepare gpt2

# Pre-build Crystal experiments
RUN for d in experiments/0*/run.cr; do \
      name=$(basename $(dirname "$d")); \
      num=$(echo "$name" | cut -c1-3); \
      crystal build "$d" -o "bin/exp${num}" --release 2>/dev/null || true; \
    done

# Python contextual vocabs built on first run (cached in /semtrace/experiments/contextual/)

# Copy remaining files
COPY DESIGN.md RESULTS.md PAPER.md BIGWIN.md SEMTRACE-PAPER.md run_all.sh ./
COPY spec/ spec/

ENTRYPOINT ["/semtrace/run_all.sh"]
