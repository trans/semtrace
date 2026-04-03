FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Python dependencies
RUN pip install --no-cache-dir transformers numpy

# Crystal (via snap-like binary install from GitHub releases)
RUN apt-get update && apt-get install -y curl ca-certificates libevent-dev libgc-dev libpcre2-dev libxml2-dev libyaml-dev && \
    curl -fsSL https://crystal-lang.org/install.sh | bash && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /semtrace
