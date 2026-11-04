# Nowcast platform

Operational infrastructure for publishing gridded nowcasts on a schedule, with
auditable provenance, and the models that run on it.

The scope is deliberately narrow — see
[ADR 0001](docs/adr/0001-pipeline-shape-scope.md):

> Gridded geospatial sequence-to-sequence forecasting, with scheduled batch
> publication and auditable provenance.

## Layout

```text
├── packages/nowcast-runtime/   # the shared runtime: acquisition, processing,
│                               # export, scheduling, atomic publication
├── models/inceptionv4/         # GSMaP NOW, direct six-lead InceptionV4
├── docs/adr/                   # why the structure is the way it is
└── pyproject.toml              # uv workspace covering packages/ only
```

`packages/*` form one uv workspace with a shared lock. **Models are not workspace
members**: each keeps its own `uv.lock`, so a CPU-only deployment and a CUDA one
can coexist without either dictating the other's torch build
([ADR 0002](docs/adr/0002-workspace-and-model-isolation.md)).

## Current state

The runtime has been extracted from the InceptionV4 package and is consumed as a
real dependency, so a fix lands in one place. **The abstraction work is not
done**: acquisition, decoding, normalization, and export are still written
against GSMaP binaries, JAXA FTP, a log1p/z-score transform, and single-band
GeoTIFF. Those become explicit ports in the phases that follow. Read the runtime
today as "shared code", not yet as "a stable plugin contract".

## Checks

The runtime and each model resolve separately, so they are checked separately.

```bash
# shared runtime, from the workspace root
uv sync
uv run pytest
uv run ruff check packages

# a model, from its own directory
uv sync --locked --python 3.12 --project models/inceptionv4
uv run --directory models/inceptionv4 pytest
uv run --directory models/inceptionv4 ruff check src tests tools deploy
```

One ruff configuration lives at the workspace root and applies to every package
and model, so lint rules cannot drift between them.

## Adding a model

Until the ports land, a new model supplies a `Backend` — a name, a version, a
function that loads a checkpoint, and a function that yields normalized leads —
plus its configuration and a SHA-256 artifact manifest. See
`models/inceptionv4/src/inceptionv4_nowcast/entrypoint.py`, which is the whole
wiring, and `packages/nowcast-runtime/src/nowcast_runtime/backend.py` for the
contract it satisfies.
