# 0001 — Pipeline shape: what the contracts admit

Status: Accepted
Date: 2026-11-04

## Context

The runtime is being generalized so that a second and third forecaster can be
operationalized without editing shared code. Before any port is defined, the
*shape* of forecast this runtime commits to must be fixed, because the shape
determines every contract signature. The current pipeline assumes, implicitly:

> N consecutive frames of one variable, at a fixed time step, on one regular 2D
> grid, batch size 1, producing M direct leads on the same grid, published as
> files under a directory per issue time.

Retrofitting a capability that the signatures do not admit touches every port at
once. Admitting it up front costs only signature verbosity.

## Decision

Define the scope of this runtime as:

> Gridded geospatial sequence-to-sequence forecasting, with scheduled batch
> publication and auditable provenance.

Within that scope, the contracts **admit** the capabilities below from the start,
while only the paths the current model needs are **implemented**:

| Capability | Contract admits | Implemented now |
|---|---|---|
| Several input products (multi-sensor fusion) | Yes — sources and decoders take an ordered collection of products, temporally aligned | No — one product |
| Static auxiliary fields (orography, land mask) | Yes — extra bundle roles, concatenated as channels | No |
| Probabilistic / ensemble output | Yes — predictions carry a member/quantile axis | No — deterministic, one channel |
| Non-uniform lead offsets | Yes — the window carries explicit offsets | No — uniform step |
| Autoregressive rollout | Yes — leads are produced by an iterator | Not applicable: the model is direct |
| Irregular grids, point or graph data | **No — out of scope** | No |

Anything outside the scope statement is refused rather than accommodated. This
is not a general machine-learning framework.

## Consequences

Signatures carry dimensions and collections that today always have length one:

- A decoded frame carries a variable axis, not a bare 2D array.
- The window specification carries explicit lead and input offsets, which
  subsume "fixed step plus count" and admit non-uniform offsets for free.
- A prediction carries a member axis, with one member as the default.
- Output naming accounts for a member or quantile component, omitted when there
  is a single deterministic member.
- The bundle manifest declares file roles, so a model with no normalization
  statistics, or with auxiliary static fields, is expressible.

The cost is paid once, in verbosity, and read on every future traversal of the
code. The benefit is that fusion and ensemble work does not require a
simultaneous change to every port.

Unit-length axes must never be silently squeezed in shared code: a squeeze is
what makes the retrofit expensive later.

## Alternatives rejected

- **Only the current shape.** Lighter and more readable contracts, but
  multi-input or ensemble support would later require changing the source,
  decoder, transform, predictor, and sink contracts together, with a model
  already in production.
- **Admit irregular grids too.** Would force the grid out of the contracts
  entirely and dissolve the export, masking, and georeferencing guarantees that
  make this runtime operationally useful.
