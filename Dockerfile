# The devcontainer should use the developer target and run as root with podman
# or docker with user namespaces.
FROM ghcr.io/prefix-dev/pixi:0.83.0 AS developer

# Add any system dependencies for the developer/build environment here
RUN apt-get update && apt-get install -y --no-install-recommends \
    graphviz \
    libqt5gui5 \
    && rm -rf /var/lib/apt/lists/*
