# Running in a container

A generated project ships a `Dockerfile` and a `compose.yaml` that work
without editing.

## What is in the image, and what is not

| In | Out |
| --- | --- |
| the runtime and your model, locked | the checkpoint |
| your configuration | credentials |
| the artifact **manifest** | published outputs |

```{important}
**Weights are not baked in.** The image carries the manifest; at startup the
service verifies the local bundle and downloads whatever is missing or
invalid. So the image is small, a new checkpoint does not mean a new image,
and the bundle is mounted from the host where it survives container
replacement.
```

## The build context is the workspace root

The runtime is a path dependency, so the build needs both directories.
**From the workspace root:**

```bash
docker build -f models/myunet/Dockerfile -t myunet-nowcast:local .
```

Note the trailing `.`: the context is the repository root, not the model
directory. `models/myunet/compose.yaml` already sets this:

```{code-block} yaml
:caption: models/myunet/compose.yaml

    build:
      context: ../..
      dockerfile: models/myunet/Dockerfile
```

```{tip}
Once the runtime is published to an index, replace the path source with a
version requirement and the context collapses back to the model directory
alone.
```

## Two details that are easy to get wrong

```{admonition} LOGNAME, and why it is set
:class: warning

Some libraries call `getpass.getuser()` **while importing**, which falls back
to an `/etc/passwd` lookup. The image runs under an arbitrary host UID, so no
baked passwd entry can cover it, and the import raises
`KeyError: getpwuid(): uid not found`.

Setting `LOGNAME` fixes it for any UID, because `getpass` reads it first. In
this repository that bug meant the container downloaded its artifacts
successfully, then crashed on the first poll and restarted forever — looking
healthy for the first few seconds of every cycle.
```

The Dockerfile uses a BuildKit cache mount rather than `UV_NO_CACHE`, because
any change under `packages/` invalidates the dependency layer and
re-downloading a large wheel each time is both slow and a reliable way to hit
a transient transfer failure.

## Running it

**From the workspace root:**

```bash
cd models/myunet
cp deploy/docker.env.example .env      # if your project has one
chmod 600 .env
mkdir -p outputs

docker compose config --quiet          # validate before building
docker compose up -d --build
docker compose logs -f
```

`.env` holds credentials and the poll interval. It is gitignored, and it goes
beside `compose.yaml`:

```{code-block} text
:caption: models/myunet/.env (new file, never committed)

MYUNET_FTP_USER=replace_me
MYUNET_FTP_PASSWORD=replace_me
NOWCAST_POLL_INTERVAL_SECONDS=300
```

```{note}
`NOWCAST_POLL_INTERVAL_SECONDS` has no model prefix because the polling loop
belongs to the shared runtime, not to your model.
```

## What the service does

1. Verify the artifact bundle. If the server is unreachable, wait one interval
   and try again. **Polling does not begin until the bundle is valid.**
2. Poll: list the latest observations, and if that issue time is already
   published and verified, do nothing.
3. Otherwise acquire the window, run, and publish atomically.
4. Wait one interval and repeat.

A failed poll logs a warning and retries on the next cycle; it does not bring
the container down.

## Mounts and permissions

```{code-block} yaml
:caption: models/myunet/compose.yaml

    volumes:
      - ./artifacts/v1:/app/artifacts/v1
      - ./outputs:/app/outputs
```

Verified weights and published outputs survive container replacement.

The image runs as UID 10001. To write host-mounted directories as yourself,
uncomment the `user:` line in `models/myunet/compose.yaml` and set the IDs:

```bash
id -u && id -g
```

## Checking an image before trusting it

```bash
docker run --rm --entrypoint sh myunet-nowcast:local -c '
  id
  ls /app /app/config /app/artifacts/v1
  myunet-nowcast --help | head -2
  python -c "import myunet_nowcast, nowcast_runtime; print(nowcast_runtime.__version__)"
'
```

That exercises the UID, the layout, the entry points and the imports without
starting the poll loop. A missing checkpoint should fail **cleanly**:

```text
ERROR Pipeline failed: [Errno 2] No such file or directory: '/app/artifacts/v1/myunet.pt'
```

## Kubernetes

The unit of work is `--scheduled`: one idempotent poll. That maps directly to
a `CronJob` running the one-shot command instead of the service loop.

```{warning}
The advisory lock is **host scoped**. It serialises polls on one machine and
does **not** coordinate replicas sharing a network volume. For several
replicas, use the orchestrator's concurrency policy —
`concurrencyPolicy: Forbid` — rather than relying on the file lock.
```
