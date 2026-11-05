# How it works

## The spine

Every run follows one sequence. Stages in brackets may be empty, and usually
are:

```{image} ../_static/pipeline-spine.svg
:alt: Source, decoder, optional preprocessing and transform into the predictor, then inverse transform, optional postprocessing and sink
:width: 100%
```

```text
Source -> Decoder -> [Preprocess] -> Transform.forward -> Predictor
       -> Transform.inverse -> [Postprocess] -> Sink
```

The runtime owns that sequence and the operational guarantees that come with
it. A model fills in the steps.

## What the runtime guarantees

These hold for every model, which is the reason for inverting control at all:

Nothing partial is ever published
: Outputs are written to a staging directory and the whole run is moved into
  place only after every file **and** its manifest exist. A crash halfway
  leaves no directory at all, rather than a directory a downstream consumer
  will happily read.

A complete run is never overwritten
: Publishing into an existing issue time raises `FileExistsError`. Reprocessing
  is deliberate, never accidental.

A scheduled poll is idempotent
: Before doing any work, the scheduler checks whether that issue time is
  already complete, verifying each recorded output against its checksum. A
  malformed or partial existing run is an error, never a silent skip.

Concurrent polls do not collide
: An advisory lock, owned by the operating system, so it is released even if
  the process is killed. A leftover file never blocks a later poll.

An incomplete window waits
: A missing observation produces `DataNotReady` and the poll retries later. It
  never quietly falls back to an older window and publishes a forecast built
  from the wrong history.

Every run carries its provenance
: A manifest records the artifacts and their checksums, the stages that ran in
  order, the source, the grid, the versions of everything, and a checksum for
  each output.

## Ports and composition

Five contracts sit between the runtime and the pieces a deployment supplies:

| Port | Bound by | What it does |
| --- | --- | --- |
| `Source` | configuration | lists observations and fetches them locally |
| `Decoder` | configuration | turns one file into a `Frame` on the requested grid |
| `Preprocess` | both | steps on a decoded frame, in physical units |
| `Transform` | the plugin | model-space scaling, and its exact inverse |
| `Predictor` | the plugin | the model |
| `Postprocess` | both | steps on a prediction, in physical units |
| `Sink` | configuration | names and writes one lead |

They are `typing.Protocol`, so an adapter never imports the framework. Detail
and rationale in [The ports](ports.md).

### Where composition lives

```{important}
A model's own entrypoint is the composition root. It is the only place that
knows both the model and the concrete adapters.

The consequence is structural, not stylistic: **the runtime never imports an
adapter, and an adapter never imports the pipeline.** An earlier attempt put
adapter construction inside the runtime and produced an import cycle, which is
what forced this arrangement.
```

```python
def compose(args) -> tuple[Deployment, str]:
    config = Config.load(args.config)
    reader = decoder()
    grid = config.domain.to_grid(reader.native_grid)
    source = LocalFiles(args.input_dir, reader.glob)
    bundle, artifacts = verify_bundle(config.artifacts, PLUGIN.bundle_roles)
    return Deployment(
        plugin=PLUGIN, source=source, decoder=reader, sink=GeoTiffSink(),
        grid=grid, bundle=bundle, artifacts=artifacts,
    ), config.product
```

### Composition is two-phase

A scheduled poll must decide whether there is new work **before** reading a
checkpoint that may be hundreds of megabytes. `Deployment` holds the cheap
half; `Deployment.load()` produces the predictor, the transform and the
model-side stages.

```python
deployment, product = compose(args)      # cheap: adapters, grid, bundle check
loaded = deployment.load()               # expensive: reads the checkpoint
```

So a poll that finds nothing new never touches the weights.

## Built-in adapters are configuration

`nowcast_runtime.builtin` ships adapters general enough to be parameterized
rather than rewritten. The decisive example is the format this platform grew
out of: the GSMaP binary is a `RawBinaryDecoder` with a dtype, a grid, a column
roll and a name pattern. There is no GSMaP module.

When your source, format and transform are all covered by built-ins, the
adapter code you write is **none**.

## What stays yours

Three things the runtime will not invent:

1. **The architecture code and its loader.** A checkpoint is weights; something
   must rebuild the network.
2. **The configuration** naming the domain, the source, the format, the stages.
3. **The artifact bundle**, with a SHA-256 manifest declaring its roles.

And one thing it cannot do at all:

```{warning}
The runtime checks the **contract**: units, grid alignment, window spacing,
output shape, bundle checksums, declared against configured. It cannot check
**scientific validity**. Whether a model trained on one product has any skill
on another is not a question any amount of this machinery will answer, and it
remains the job of that model's own validation record.
```

## Why these seams and not others

Each port cost something to add and had to be justified by evidence rather
than anticipation:

- `Source` and `Decoder` stay separate because local directories and remote
  acquisition already feed the same decoding path. The split also removed a
  bug class: a source owns path syntax and reports bare names, a decoder owns
  name semantics. Name parsing in shared code with POSIX assumptions is what
  broke local acquisition on Windows.
- `Preprocess` and `Postprocess` became ports because three of them already
  existed as branches in shared code — a missing-data policy with three arms,
  and a latitude flip on input and again on output. Every new model would have
  added another arm to code all of them shared.
- `Grid` is deliberately **not** a port. There is one grid model, regular and
  axis-aligned. Irregular grids are out of scope.
- The scheduler is **not** a port either. `poll_once` is an idempotent unit of
  work that any scheduler can drive, so making it pluggable would abstract
  something that already composes.

The full reasoning, including the alternatives rejected, is in
[ADR 0003](../adr/0003-ports-and-their-granularity.md) and
[ADR 0004](../adr/0004-optional-stages-and-built-in-adapters.md).

## Reading on

- [The ports](ports.md) — each contract in detail
- [Grids and windows](grids-and-windows.md) — geometry and time
- [Processing stages](stages.md) — ordering, and where a stage belongs
- [Artifacts and manifests](artifacts-and-manifests.md) — provenance
