# Build context is the workspace root: nowcast-runtime is a path dependency.
#   docker build -f models/inceptionv4/Dockerfile .
FROM python:3.12-slim-bookworm AS builder

# A build cache mount, not UV_NO_CACHE: any change under packages/ invalidates
# the sync layer, and re-downloading a 180 MiB wheel each time is both slow and
# a reliable way to hit a transient transfer failure. The cache lives in the
# builder stage only, so the published image is unaffected.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_CACHE_DIR=/opt/uv-cache
WORKDIR /build
RUN python -m pip install --no-cache-dir uv==0.11.32
COPY packages/nowcast-runtime ./packages/nowcast-runtime
COPY models/inceptionv4/pyproject.toml models/inceptionv4/uv.lock models/inceptionv4/README.md \
     ./models/inceptionv4/
COPY models/inceptionv4/src ./models/inceptionv4/src
RUN --mount=type=cache,target=/opt/uv-cache \
    uv sync --locked --no-dev --no-editable \
      --project models/inceptionv4 --python /usr/local/bin/python

FROM python:3.12-slim-bookworm

# LOGNAME: torch calls getpass.getuser() while importing, which otherwise falls
# back to an /etc/passwd lookup. This image runs under an arbitrary host UID, so
# no baked passwd entry can cover it; getpass reads LOGNAME first, UID agnostic.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp \
    LOGNAME=nowcast
RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY models/inceptionv4/config ./config
COPY models/inceptionv4/artifacts/v1/manifest.json models/inceptionv4/artifacts/v1/*.npy \
     ./artifacts/v1/
RUN mkdir -p outputs \
    && chmod -R a+rX /app \
    && chown -R 10001:10001 artifacts outputs
USER 10001:10001
CMD ["inceptionv4-serve"]
