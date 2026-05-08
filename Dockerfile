# The devcontainer should use the developer target and run as root with podman
# or docker with user namespaces.
ARG PYTHON_VERSION=py312
FROM ghcr.io/prefix-dev/pixi as developer

# Add any system dependencies for the developer/build environment here
RUN apt-get update && apt-get install -y --no-install-recommends \
    graphviz \
    libqt5gui5 \
    && rm -rf /var/lib/apt/lists/*

# Install the pixi environment for the specified Python version
RUN pixi install --locked -e $PYTHON_VERSION
