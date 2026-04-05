# Semtrace: Recovering Text from Embedding Vectors

Greedy residual decomposition of embedding vectors into constituent tokens. Given a vector in embedding space, find the tokens whose embeddings sum to approximate it.

**Thomas Sawyer & Claude Opus 4.6** · [Paper](SEMTRACE-PAPER.md) · [Results](RESULTS.md) · [Design](DESIGN.md)

## Quick Start (Container)

```bash
# Build (one-time — downloads GPT-2, builds indexes, ~10 min)
podman build -t semtrace .

# Run an experiment
podman run --rm semtrace 001          # Gettysburg Address
podman run --rm semtrace 002          # Mary Had a Little Lamb
podman run --rm semtrace 003          # Tale of Two Cities

# Run with GPU (for Python experiments)
podman run --rm --device nvidia.com/gpu=all semtrace 012

# Interactive shell
podman run --rm -it semtrace bash
```

## Quick Start (Native)

Requires: Crystal >= 1.19, Python 3 with torch + transformers.

```bash
# Install Crystal dependencies
shards install

# Download GPT-2 embeddings (one-time)
crystal build src/cli.cr -o bin/semtrace --release
bin/semtrace prepare gpt2

# Run an experiment
./run_all.sh 001
```

## Experiments

| # | Name | Type | What It Tests |
|---|---|---|---|
| 001 | Gettysburg Address | Crystal | Static decomposition across GPT-2 family |
| 002 | Mary Had a Little Lamb | Crystal | Short text, 100% recovery on Large/XL |
| 003 | Tale of Two Cities | Crystal | Long text capacity limits |
| 004 | Optimization | Crystal | Lookahead, normalization, semantic matching |
| 005 | Capacity Test | Crystal | Random token baseline vs real text |
| 006 | Llama HNSW | Crystal | Llama 3.2 3B with approximate search |
| 007 | Metric Comparison | Crystal | Cosine vs L2 vs inner product |
| 008 | Llama Brute-Force | Crystal | Exact search reveals HNSW was the bottleneck |
| 009 | Union of Metrics | Crystal | Combining three metrics for better recovery |
| 010 | Contextual Embeddings | Python | Layer-by-layer analysis, attention is non-additive |
| 011 | Attention Bias | Python | Constant bias discovery, partial contextual recovery |
| 012 | Coordinate Descent | Python | Iterative optimization, 43%→94% on GPT-2 Small |

## Key Findings

1. **Static decomposition works.** 100% on short text, 90%+ on medium text (GPT-2 XL).
2. **Dimensionality phase transition** at ~1280d — accuracy jumps discontinuously.
3. **Semantic coherence helps 6x** — real text recovers far better than random tokens.
4. **Coordinate descent doubles accuracy** on smaller models (43%→94%).
5. **Attention adds a constant bias** (99.5% of energy) — subtracting it enables contextual decomposition.
6. **Normalization destroys magnitude** — token count and residual signal are lost.
7. **The model remembers** — the barrier is the serving layer, not the model itself.

## Additional Models

```bash
# Download other GPT-2 sizes
bin/semtrace prepare gpt2-medium
bin/semtrace prepare gpt2-large
bin/semtrace prepare gpt2-xl

# Extract Llama from Ollama (requires Ollama with model pulled)
bin/semtrace extract-gguf /path/to/model.gguf

# Run with a specific model
./run_all.sh 001    # uses GPT-2 Small by default
bin/exp001 --data data/gpt2-xl --builtin gettysburg
```

## License

MIT
