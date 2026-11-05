FROM ubuntu:26.04

RUN apt-get update && apt-get install -y \
   git \
   build-essential \
   curl \
   && rm -rf /var/lib/apt/lists/*
# Set the working directory
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_CACHE_DIR=/root/.cache/uv

# Keep this uv version in step with setup-uv in the workflows.
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

# Copy the project files to the working directory

# .python-version pins the interpreter. Unpinned, uv downloads the newest
# CPython, so the image silently changes Python whenever one is released -- and
# a release can break the build outright (CPython 3.15: dracopy 2.0.0 ships no
# cp315 wheel and no sdist, so `uv sync` fails). Bump it deliberately once
# dependencies support the new version.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=.python-version,target=.python-version \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --no-dev --frozen --no-install-workspace


COPY . /app

# .git isn't in the build context, so the version comes in as an argument.
# docker-publish.yml sets it; a local build reports this placeholder.
ARG PIXELMAP_VERSION=0.0.0+local

# Install the Python packages using uv
RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PIXELMAP="$PIXELMAP_VERSION" \
    uv sync --no-dev --frozen

# Pre-download the common mouse and rat atlases, plus the registry index, so
# the running container never has to reach the atlas host.  Under
# brainglobe-atlasapi v3 both atlases together are only ~9 MB of compressed
# OME-Zarr, so this is cheap.
#
# NOT best-effort: prefetch_atlases.py exits non-zero if it cannot warm the
# cache, failing the build.  A cold image is what took the server down in
# v1.2.x -- it pushes the download into the single-threaded Bokeh process at
# runtime.  See the script's docstring for the full story.
ENV PATH="/app/.venv/bin:$PATH"
RUN python /app/scripts/prefetch_atlases.py

# Keep a pristine copy of the warmed cache outside ~/.brainglobe.
#
# Production mounts a Docker volume at /root/.brainglobe to persist atlases
# users download at runtime.  A named volume is seeded from the image *only
# when the volume is empty*, so once it exists it permanently masks whatever
# the image baked in -- and a single deploy of a cold image leaves a cold
# volume that shadows every later image, warm or not.  entrypoint.sh copies
# anything missing out of this seed at startup, which makes that trap
# impossible regardless of the volume's history.
ENV BRAINGLOBE_SEED=/opt/brainglobe-seed
RUN cp -a /root/.brainglobe "$BRAINGLOBE_SEED"

# Expose the port
ENV INTERNAL_PORT=5008
ENV EXTERNAL_PORT=5008
ENV NUM_PROCS=1
ENV ADDRESS=0.0.0.0
ENV ALLOW_WEBSOCKET_ORIGIN=*
# glibc's malloc never returns freed arenas to the OS; under repeated
# session bursts idle RSS ratchets upward with zero live sessions (observed
# 277 -> 476 -> 544 -> 592 -> 615 MB). Capping the number of arenas bounds
# how much of that high-water mark each one can hold.
ENV MALLOC_ARENA_MAX=2
EXPOSE ${INTERNAL_PORT}

# Copy the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Copy the Dockerfile to /dockerfile within the container
COPY Dockerfile /Dockerfile

# Health check
#
# /liveness (added via --liveness in entrypoint.sh) returns 200 without
# building a Bokeh session. Hitting / or /app instead builds a full session
# per probe (+27 MB RSS, torn down ~60 s later) -- in production that meant
# a session build/destroy cycle roughly every 31 s with zero real users.
HEALTHCHECK CMD curl --fail http://localhost:${INTERNAL_PORT}/liveness

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]

