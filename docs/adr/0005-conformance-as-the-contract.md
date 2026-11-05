# 0005 — Conformance is where the contract is actually enforced

Status: Accepted
Date: 2026-11-05

## Context

[ADR 0003](0003-ports-and-their-granularity.md) promised three layers of
enforcement and delivered two: static checking inside this workspace, and a
structural check when an adapter is resolved. The third, a suite verifying what
a type cannot express, did not exist. Without it the ports were a convention,
and an adopter would discover a violated contract in production rather than in
their own test run.

There was a second gap. InceptionV4 exercises exactly one path through the
pipeline: hourly inputs, a log/z-score transform, three stages, one
deterministic output variable. Everything [ADR 0001](0001-pipeline-shape-scope.md)
committed the contracts to admitting — a missing transform, no stages, a
sub-hourly step, a non-uniform window, ensemble members, several output
variables — had never run. Signatures admitted them; nothing proved they worked.

## Decision

### A conformance suite a model subclasses

`nowcast_runtime.testing.ConformanceSuite` holds the checks every deployment
must pass. A model implements one hook and declares its product:

```python
class TestMyModel(ConformanceSuite):
    product = "my_product"

    def build(self, tmp_path):
        return compose(args_for(tmp_path))
```

It checks what types cannot: that a transform round trips, that each stage
leaves a frame on its own grid at its own time, that `describe()` is JSON
serializable so a manifest can hold it, that valid times follow the declared
step rather than an assumed hour, that a completed run is recognized instead of
redone, that a shorter horizon is honoured and a longer one refused, that an
incomplete window waits instead of silently narrowing, and that a failed write
leaves nothing at all behind.

It runs a model's **real composition root**, not a stand-in, so what passes is
the wiring the container uses.

### Decoders synthesize their own observations

`filename` and `synthesize` join the decoder surface as optional methods, and
`RawBinaryDecoder` implements both: `synthesize` is the exact inverse of `read`.
A deployment using a built-in decoder therefore gets the end-to-end half of
conformance without supplying any real data. A decoder without them still gets
every contract-level check, and the rest skip with a stated reason.

This also forced a correction. Matching and generating are different jobs: a
pattern with optional groups accepts more names than it can produce, so
`TimestampedName` now carries a `template` for generation and checks the
generated name against the pattern, which keeps the two from drifting.

### A synthetic reference deployment is kept permanently

`tests/test_reference_deployment.py` is a deployment deliberately unlike
InceptionV4 on every axis that matters, run through the same suite. It is not a
substitute for porting a second real model, and the file says so. What it does
is hold the admitted-but-unused paths under test before a real model depends on
them, and prove the suite passes for something that is not InceptionV4.

## Consequences

- Writing the reference deployment found two real defects before any model could
  hit them. The pipeline validated output width against the *input* variable
  count, which would have rejected a model that consumes one field and emits
  several; `Predictor.output_variables` is now declared. And the file name
  contract could not express a sub-hourly observation, which made every
  ten-minute product impossible.
- The round-trip check runs on values **after** the input stages, because that is
  what a transform actually receives. Checking the raw field would have been
  checking the wrong contract: a provider sentinel is still raw at that point,
  and no transform is asked to survive one.
- Conformance costs real time. The model suite went from about 20 to about 70
  seconds, because each test loads the checkpoint. That is acceptable for what it
  buys and is the price of running the real composition root rather than a fake.
- `Predictor` gained a member. That lands inside runtime 0.3.0, which exists only
  on this branch and was never published, so no version bump is owed and no
  consumer is broken. Once 0.3.0 is released, the same change would require one.

## What it still does not do

It checks the contract: units, grid alignment, window spacing, output shape,
naming, atomicity, idempotency. It cannot check scientific validity. Whether a
model trained on one product has skill on another remains the job of that
model's own validation record, and no amount of conformance will answer it.

## Alternatives rejected

- **A checklist in prose.** Costs nothing to write and nothing enforces it. The
  existing documentation already carried assumptions that silently went stale,
  which is the argument against adding more.
- **Fixtures each model writes itself.** Maximum flexibility, and every model
  would check a different subset, which is how uniform operational guarantees
  stop being uniform.
- **Conformance against a fake deployment only.** Fast and self-contained, but it
  would never exercise a real composition root, which is exactly where a model
  gets its wiring wrong.
- **Requiring a second real model before shipping the suite.** Correct in
  principle and it would have blocked indefinitely on data this repository does
  not have. The synthetic deployment is the honest compromise, labelled as one.
