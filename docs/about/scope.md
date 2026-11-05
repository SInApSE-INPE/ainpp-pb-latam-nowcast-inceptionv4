# Scope and limits

## What this is for

> Gridded geospatial sequence-to-sequence forecasting, with scheduled batch
> publication and auditable provenance.

That sentence is the scope, and anything outside it is refused rather than
accommodated. A platform that accepts everything guarantees nothing.

## What the contracts admit

| Capability | Contracts admit | Implemented |
| --- | --- | --- |
| A trained model of any architecture | yes | yes |
| Any source: local, FTP, and others you adapt | yes | local and FTP built in |
| Any raw binary format, as configuration | yes | yes |
| No preprocessing, no postprocessing | yes | yes, the default |
| Any time step, including sub-hourly | yes | yes |
| A non-uniform input window | yes | yes |
| Several output variables per lead | yes | yes |
| Ensemble members | yes | yes |
| Autoregressive or recurrent rollout | yes | the iterator supports it |
| Static auxiliary fields | yes | not yet exercised |
| Several input products, fused | yes | not yet implemented |

The last two are admitted by the signatures but have no working path yet.
Adding them does not require changing a contract, which was the point of
admitting them early.

## What is out of scope

These are refused, not merely unimplemented:

- **Irregular grids, point observations, graph data.** The grid is a value
  object, not a port, and admitting these would dissolve the export, masking
  and georeferencing guarantees that make the rest useful.
- **Models that are not history-in, leads-out.** A classifier, a diagnostic
  with no time axis, a data assimilation cycle.
- **Real-time streaming.** The unit of work is a scheduled batch.
- **Training.** This operationalizes an already-trained model.

## What the runtime will never do for you

Three things you always supply:

1. **The architecture code and its loader.** A checkpoint is weights.
2. **The configuration** naming domain, source, format and stages.
3. **The artifact bundle** with its SHA-256 manifest.

And one judgement it cannot make:

```{warning}
The runtime verifies the contract: units, grid alignment, window spacing,
output shape, bundle checksums, and what a bundle declares against what a
deployment configured.

It cannot verify **scientific validity**. It will happily publish a forecast
from a model trained on one product and fed another, with every checksum
correct and every shape right. Whether that forecast means anything is a
meteorological question, and it belongs in your model's own validation record.
```

## Honest status

The platform runs one real model, InceptionV4 on GSMaP. A second, deliberately
different deployment exists as a permanent test — no transform, no stages, a
ten-minute step, a non-uniform window, two members and two output variables —
and it is synthetic.

```{admonition} The two-model rule
:class: important

An abstraction with one real implementation is a guess. The claim that any
model can run here becomes a demonstrated fact when a **second real model**
runs without a line changing in the shared runtime, and not before. Writing
the synthetic deployment already exposed two defects that one model had hidden,
which is the argument for doing it with a real one too.
```
