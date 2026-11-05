# Build context is the workspace root, because nowcast-runtime is a path
# dependency:  docker build -f models/irene/Dockerfile .
FROM python:3.12-slim-bookworm AS builder

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_CACHE_DIR=/opt/uv-cache
WORKDIR /build
RUN python -m pip install --no-cache-dir uv==0.11.32
COPY packages/nowcast-runtime ./packages/nowcast-runtime
COPY models/irene/pyproject.toml models/irene/uv.lock models/irene/README.md ./models/irene/
COPY models/irene/src ./models/irene/src
RUN --mount=type=cache,target=/opt/uv-cache \
    uv sync --locked --no-dev --no-editable \
      --project models/irene --python /usr/local/bin/python

FROM python:3.12-slim-bookworm

# LOGNAME: some libraries call getpass.getuser() while importing, which falls
# back to an /etc/passwd lookup. This image runs under an arbitrary host UID, so
# no baked passwd entry can cover it; getpass reads LOGNAME first.
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
COPY models/irene/config ./config
COPY models/irene/artifacts/v1/manifest.json ./artifacts/v1/
RUN mkdir -p outputs \
    && chmod -R a+rX /app \
    && chown -R 10001:10001 artifacts outputs
USER 10001:10001
CMD ["irene-serve"]
