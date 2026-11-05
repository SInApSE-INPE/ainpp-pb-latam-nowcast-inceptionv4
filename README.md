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

## How it fits together

`nowcast_runtime.ports` holds the contracts. Every run follows one spine, and
each bracketed stage may be empty:

```text
Source -> Decoder -> [Preprocess] -> Transform.forward -> Predictor
       -> Transform.inverse -> [Postprocess] -> Sink
```

The runtime owns the sequence and the operational guarantees that go with it:
nothing partial is ever published, a complete run is never overwritten, a
scheduled poll is idempotent and locked, and every run carries a manifest
recording the artifacts, stages, versions and checksums that produced it.

`nowcast_runtime.builtin` supplies adapters general enough to be parameterized
rather than rewritten. The GSMaP binary format this repository started with is
one configuration of `RawBinaryDecoder`, not a module of its own, which is the
standard the other built-ins are held to.

Composition lives in a model's own entrypoint, so the runtime never imports an
adapter and an adapter never imports the pipeline.

Read the decisions in order: [0001](docs/adr/0001-pipeline-shape-scope.md) fixes
what shape of forecast is in scope, [0003](docs/adr/0003-ports-and-their-granularity.md)
why there are these ports and not more, and
[0004](docs/adr/0004-optional-stages-and-built-in-adapters.md) why every stage is
optional.

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

Start from something that already runs:

```bash
uv run nowcast-new-model --name "MyUNet" --product my_product --units mm/h
uv sync --project models/myunet
uv run --directory models/myunet pytest
```

That generates a complete project with a persistence predictor and a placeholder
bundle, and the conformance suite is **green before you write a line**. Every
spot you must change carries a `REPLACE THIS` marker, and the generated
`README.md` lists them in the order to do them. Run the suite after each change:
a failure then means the thing you just touched.

What you are filling in is a `ModelPlugin`. Only six fields are required:

```python
PLUGIN = ModelPlugin(
    name="MyUNet",
    slug="myunet",
    version="0.1.0",
    bundle_roles=("checkpoint",),   # declared, not fixed: no mean/std needed
    requires_runtime=">=0.3,<0.4",
    load=load_model,                # rebuild the network, load the weights
)
```

`transform`, `preprocess` and `postprocess` are optional. A model that consumes
physical units and needs no stages declares none, and the runtime applies
identity: the absence of processing is the default path, not a special case.

Three things are always yours to supply, and the runtime will not invent them:

1. **The architecture code and its loader.** A checkpoint is weights; something
   has to rebuild the network. The runtime loads and runs, it never infers
   topology.
2. **The configuration** naming the domain, the source, the format and the stages.
3. **The artifact bundle**, with a SHA-256 manifest declaring the roles the
   plugin named.

Everything else should be composition. When your source, format and transform
are all covered by built-ins, the adapter code you write is none.

What the runtime checks is the contract: units, grid alignment, window spacing,
output shape, bundle checksums, and what a bundle declares against what a
deployment configured. What it cannot check is scientific validity — whether a
model trained on one product has skill on another stays the job of that model's
own validation record.

## Proving a model conforms

Subclass the shared suite and implement one hook:

```python
from nowcast_runtime.testing import ConformanceSuite

class TestMyModel(ConformanceSuite):
    product = "my_product"

    def build(self, tmp_path):
        return compose(args_for(tmp_path))
```

It runs your real composition root and checks what a type cannot: that the
transform round trips, that each stage leaves a frame on its own grid, that
valid times follow your declared step, that a completed run is recognized rather
than redone, that an incomplete window waits instead of narrowing, and that a
failed write leaves nothing behind. When your decoder is a built-in, the suite
synthesizes its own observations, so none of this needs real data.

See [ADR 0005](docs/adr/0005-conformance-as-the-contract.md) for why this is the
layer that actually enforces the contracts.

## Worked examples

`models/inceptionv4` is the real one; its `entrypoint.py` is the whole
composition and is worth reading first.

`packages/nowcast-runtime/tests/test_reference_deployment.py` is a deliberately
different deployment: no transform, no stages, a ten-minute step, a non-uniform
window, two members and two output variables. It is synthetic and does not
replace a second real model, but it is the shortest complete example of the
contracts, and writing it found two defects in the runtime.
