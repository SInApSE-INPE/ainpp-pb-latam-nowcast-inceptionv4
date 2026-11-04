# 0004 — Optional stages, and adapters that are configuration

Status: Accepted
Date: 2026-11-04
Amends: [ADR 0003](0003-ports-and-their-granularity.md)

## Context

The requirement is sharper than "the ports exist": the effort of adopting this
runtime must be proportional to what is genuinely unique about the adopter's
model, and nothing else. Any domain, any source, any input format, and any kind
of pre or post processing — **including none at all**.

Two things in ADR 0003 do not meet that.

First, `Transform` was mandatory on the plugin. A model that consumes physical
units would have to supply an identity transform, which makes the absence of
preprocessing a special case instead of the default.

Second, pre and post processing were not ports at all. That is not a future
need: the current pipeline already carries three of them as branches inside
shared code.

| Special case | Location before this decision |
| --- | --- |
| `missing_policy`, three branches | `processing.py:62-69` |
| Flip latitude on input | `processing.py:76-77` |
| Flip latitude on output | `pipeline.py:199-200` |

Those are configurable stages wearing an `if`. Every new model would add
another branch to code shared by all of them, which is exactly the failure this
platform exists to prevent.

## Decision

### The spine, with every stage optional

    Source -> Decoder -> [Preprocess] -> Transform.forward -> Predictor
           -> Transform.inverse -> [Postprocess] -> Sink

`Preprocess` and `Postprocess` become ports: ordered, possibly empty sequences
of steps operating in physical units. `Transform` becomes optional, and the
runtime applies identity when a plugin declares none. A minimal `ModelPlugin` is
six fields: name, slug, version, bundle roles, runtime range, and a loader.

### Stage order is fixed, not composed

Configured stages run first on input and last on output; model stages run last
on input and first on output. A configured step therefore always sees physical
units as the provider published them, and a model step always sees them as its
checkpoint expects. Reversing the output restores the input ordering, so a
symmetric pair such as a latitude flip composes correctly with neither side
knowing about the other.

The split follows the rule from ADR 0003: latitude order is coupled to the
checkpoint, so the plugin owns it. Clipping a provider sentinel is coupled to
the source, so configuration owns it. Missing-data policy is operational, so
configuration owns it.

### A Prediction carries its own mask

The mask moves into `Prediction`, so a postprocess step and a sink each receive
everything they need from one object, rather than a field and a mask that a
caller has to keep paired.

### Built-in adapters are parameterized, so most adoption is configuration

The runtime ships generic adapters covering the common cases, parameterized
rather than subclassed. The decisive example is this model's own input: a
`RawBinaryDecoder` taking dtype, shape, origin, resolution, CRS, a column roll
and a file-name pattern expresses the GSMaP binary format **as configuration**.
A new raw-binary product becomes a config block, not a new module.

This also dissolves a blocker found while starting the migration. A separate
`nowcast-gsmap` package would have to import `nowcast_runtime.ports`, while the
runtime building that adapter from configuration would have to import the
package: a cycle, and a reintroduction of the coupling the ports exist to
remove. Generic adapters belong to the runtime; only genuinely bespoke ones need
to live outside it, and those are wired by the model that needs them.

### The irreducible floor

Three things cannot be made generic, and the documentation must say so rather
than imply otherwise:

1. **The architecture code and its loader.** A checkpoint is weights; something
   has to reconstruct the network. The runtime loads and runs, it never infers
   topology.
2. **The configuration** naming the domain, source, format and stages.
3. **The artifact bundle** with its SHA-256 manifest and declared roles.

Everything else should be composition. When an adopter's source, format and
transform are all covered by built-ins, the adapter code they write is zero.

### What the runtime still cannot check

Contracts can be made machine-checkable — units, grid, step, declared against
configured. Scientific validity cannot. The runtime will not tell anyone whether
a model trained on one product is skilful on another; that remains the job of a
model's own validation record.

## Consequences

- Seven ports rather than five. ADR 0003 argued against ports beyond the
  evidence, and this clears that bar: three branches already in shared code.
- `Preprocess` and `Postprocess` share a shape but keep distinct types, because
  one operates on a `Frame` and the other on a `Prediction`.
- `missing_policy` stops being a special-cased string and becomes the first
  built-in preprocess step.
- Built-in adapters live in the runtime, which therefore grows a surface that
  must stay generic. A provider-specific hack belongs in a model, never here.
- The adoption story becomes scaffolding plus a conformance suite: generate a
  plugin skeleton, fill in the loader, point the configuration at built-ins, and
  the suite says whether the result satisfies the contracts.

## Alternatives rejected

- **One unified `Stage` port applied at both ends.** Fewer concepts, but it
  would need a payload type loose enough to cover a Frame and a Prediction,
  which discards the invariants those types exist to enforce.
- **Preprocessing inside the decoder.** Convenient, and wrong: clipping a
  sentinel is source-coupled while flipping latitude is checkpoint-coupled, and
  folding both into the decoder would put model knowledge in a format adapter.
- **Keep `missing_policy` as a configured string with branches in the runtime.**
  Smaller change, but it keeps the pattern that every new model extends shared
  code, and the branches are already three.
- **Let configuration choose the transform.** Rejected again, for the reason in
  ADR 0001: its statistics are inseparable from the checkpoint, and a mismatch
  raises nothing while producing plausible output.
