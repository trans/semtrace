# Semtrace experiment runner

image := "semtrace"
data := "./data"
experiments := "./experiments"

# Build the container image
build:
    podman build -t {{image}} experiments/

# Run a Python experiment (e.g., just py contextual/run.py --text "hello world")
py *args:
    podman run --rm \
        -v {{experiments}}:/semtrace \
        -v {{data}}:/data \
        {{image}} {{args}}

# Run a Python experiment with GPU
py-gpu *args:
    podman run --rm \
        --device nvidia.com/gpu=all \
        -v {{experiments}}:/semtrace \
        -v {{data}}:/data \
        {{image}} {{args}}

# Run a Crystal experiment (e.g., just cr experiments/009-union-metrics/run.cr)
cr *args:
    podman run --rm \
        -v .:/semtrace \
        -w /semtrace \
        {{image}} sh -c "crystal run {{args}}"

# Run the contextual embedding experiment
contextual *args:
    podman run --rm \
        --device nvidia.com/gpu=all \
        -v {{experiments}}:/semtrace \
        -v {{data}}:/data \
        {{image}} python3 contextual/run.py {{args}}

# Run contextual on CPU (no GPU required)
contextual-cpu *args:
    podman run --rm \
        -v {{experiments}}:/semtrace \
        -v {{data}}:/data \
        {{image}} python3 contextual/run.py --device cpu {{args}}
