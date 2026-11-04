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
real dependency, so a fix lands in one place.

`nowcast_runtime.ports` now defines the contracts a deployment composes: five
ports, the value objects they exchange, and the plugin declaration a model
supplies. See [ADR 0003](docs/adr/0003-ports-and-their-granularity.md) for which
seams are justified and why there are not more of them.

**The pipeline has not been migrated onto those ports yet.** Acquisition,
decoding, normalization and export still read GSMaP binaries from JAXA FTP
through a log1p/z-score transform into single-band GeoTIFF, written directly
rather than through the contracts. Until that migration lands, read `ports.py`
as the target and the remaining modules as an implementation that does not yet
satisfy it.

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

What a model will supply is a `ModelPlugin`: its name and version, the artifact
roles its bundle contains, the runtime range it needs, and callables that build
its `Predictor` and `Transform`. That declaration lives in `ports.py`.

What a model supplies *today*, until the migration lands, is the narrower
`Backend` in `backend.py` — a name, a version, a checkpoint loader and a
function yielding normalized leads — plus its configuration and a SHA-256
artifact manifest. `models/inceptionv4/src/inceptionv4_nowcast/entrypoint.py` is
the whole wiring, and is worth reading first.
