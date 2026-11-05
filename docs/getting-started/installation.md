# Installation

## What you need

| | |
| --- | --- |
| Python | 3.11, 3.12 or 3.13 |
| [uv](https://docs.astral.sh/uv/) | 0.11 or newer; it manages the environments and the lock files |
| Git | to clone the repository |
| Docker | only if you intend to build container images |

The runtime itself depends on NumPy and rasterio and nothing else. In
particular it depends on **no deep learning framework**: a model brings its own,
or none at all.

## Clone and set up

```bash
git clone https://github.com/SInApSE-INPE/ainpp-pb-latam-nowcast-inceptionv4.git
cd ainpp-pb-latam-nowcast-inceptionv4
uv sync
```

That creates the workspace environment, which holds the shared runtime and its
development tools.

## How the workspace is laid out

```text
.
├── packages/nowcast-runtime/   the shared runtime
├── models/inceptionv4/         a real model that runs on it
├── docs/                       this documentation, and the decision records
└── pyproject.toml              the uv workspace, covering packages/ only
```

```{important}
**Models are deliberately not workspace members.** Each keeps its own
`uv.lock`, so a CPU-only deployment and a CUDA one can coexist without either
dictating the other's build of torch. The practical consequence is that the
runtime and each model are installed and checked **separately**, with two sets
of commands rather than one.

The reasoning is in [ADR 0002](../adr/0002-workspace-and-model-isolation.md).
```

## Run the checks

The shared runtime, from the workspace root:

```bash
uv run pytest
uv run ruff check packages
```

```text
........................................................................ [ 64%]
.......................................                                  [100%]
111 passed in 5.99s
```

A model, from its own project:

```bash
uv sync --locked --python 3.12 --project models/inceptionv4
uv run --directory models/inceptionv4 pytest
uv run --directory models/inceptionv4 ruff check src tests tools deploy
```

One ruff configuration lives at the workspace root and governs every package
and model, so lint rules cannot drift between them.

### Tests that skip, and why

Some tests skip rather than fail, because they cannot run everywhere and a
suite that always shows failures teaches people to ignore failures:

`reference/ assets are not in the repository`
: Two InceptionV4 tests compare against the supplied training reference, which
  is not redistributed. They run only where those files are present.

`symlinks unavailable on this platform`
: One cache-retention test creates a symbolic link. Windows grants that only
  with a privilege or with Developer Mode enabled.

## Building the documentation

```bash
uv sync --group docs
uv run --group docs sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html`. To regenerate the figures from a real
pipeline run:

```bash
uv run --group docs python docs/_scripts/make_figures.py
```

## Known environment problems

```{admonition} Windows: pytest cannot create its temporary directory
:class: warning

If every test errors with `PermissionError: [WinError 5]` pointing at
`%TEMP%\pytest-of-<you>`, a stale directory from an earlier run has broken
permissions. From an elevated terminal:

    takeown /F "$env:TEMP\pytest-of-$env:USERNAME" /R /D Y
    Remove-Item -Recurse -Force "$env:TEMP\pytest-of-$env:USERNAME"

As a workaround without elevation, pass `--basetemp` somewhere writable.
```

```{admonition} Container builds failing on TLS errors
:class: warning

A build that fails with `DECRYPTION_FAILED_OR_BAD_RECORD_MAC` or
`cannot decrypt peer's message` while downloading a large wheel is hitting a
network problem inside the Docker build network, not a problem with the
Dockerfile. The host reaching the same index proves nothing, because the build
uses its own network namespace. Retry; if it persists, look at MTU or VPN
settings in your Docker installation.
```

## Next

[Quickstart](quickstart.md) generates a working model and publishes a forecast
in about five minutes.
