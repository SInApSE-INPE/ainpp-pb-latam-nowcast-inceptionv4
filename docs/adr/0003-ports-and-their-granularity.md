# 0003 — The ports, and how many of them there are

Status: Accepted
Date: 2026-11-04

## Context

With the runtime extracted ([ADR 0002](0002-workspace-and-model-isolation.md)),
it is shared code but not yet a contract: acquisition, decoding, normalization
and export are written directly against GSMaP binaries, JAXA FTP, a
log1p/z-score transform and single-band GeoTIFF. A model on another source or
another output format cannot be operationalized without editing that shared
code, which is the problem this platform exists to solve.

Every port is worth something and costs something. It buys extensibility and a
test seam. It costs indirection on every traversal of the code, a contract that
must then be maintained, and version compatibility burden. The question is
therefore not whether to have ports, but which seams are justified by evidence
rather than by anticipation.

## Decision

Five ports, defined as `typing.Protocol` in `nowcast_runtime.ports`, plus the
value objects they exchange.

| Port | Bound by | Evidence for the seam |
| --- | --- | --- |
| `Source` | configuration | Already required: local directories and JAXA FTP both feed the same decoding path |
| `Decoder` | configuration | Anti-corruption layer; isolates provider quirks such as the GSMaP longitude roll |
| `Transform` | **the plugin** | Contract is valuable and testable by round trip, but the binding is not configurable |
| `Predictor` | the plugin | It is the model |
| `Sink` | configuration | One model may publish several formats or destinations |

### Source and Decoder stay separate

The split is justified by code that already exists, not by a future need:
`--source local` and `--source ftp` both produce files that one decoder reads.
It is also where a real bug lived. File name parsing sat in shared code that
derived names with POSIX path semantics, so on Windows the entire native path
was treated as the file name and no observation ever matched. The responsibility
split removes that class of bug by construction: **a source owns path syntax and
supplies bare names; a decoder owns name semantics and answers `observed_at`.**

### Transform is a port, but the plugin binds it

This reverses an earlier intention to select the transform through
configuration. The statistics live in the model artifact bundle and are
inseparable from the checkpoint: this model's inverse has a floor near
0.0453 mm/h purely because of its mean. A configurable transform would admit a
mismatch that raises nothing and produces plausible but meaningless forecasts.

The general rule this yields, applied to every future decision:

> What is scientifically coupled to the checkpoint belongs to the plugin. What
> is operationally variable belongs to the configuration.

### NumPy at the boundary, not tensors

Arrays cross the ports as NumPy. Tensor creation, device placement and the
inference loop live inside a model. The runtime keeps `torch` as a dependency
today, but no port mentions it, so the contracts do not need revising if a model
arrives on another framework.

### `Protocol`, with enforcement in three layers

Structural typing keeps adapters from importing the framework, which is real
dependency inversion. It is also checked only if someone runs a type checker, so
conformance is enforced in three layers: static checking in this workspace,
a structural check when an adapter is resolved, and a conformance suite that
verifies semantics a type cannot express, such as `inverse(forward(x)) ≈ x`.
Types do not capture contracts; tests do.

### Invariants live in the value objects

`Grid`, `WindowSpec`, `Frame` and `Prediction` validate on construction, so an
invalid one cannot be passed anywhere. `Grid.locate` replaces arithmetic that was
previously inlined in the GSMaP reader, and a regression test pins it to the
offsets that code produced for the operational domain.

`WindowSpec` carries explicit offsets rather than a step and a count, which
subsumes the uniform case through `WindowSpec.uniform` and admits a non-uniform
history at no cost. This removes the assumption that one hour is the only
possible spacing, which was previously written into time parsing, window
selection and valid-time arithmetic.

`bundle_roles` is declared by the plugin, so the artifact reader stops being
fixed to checkpoint, mean and std. This is what the second model needs: its
manifest names a single model file and carries no normalization statistics.

## What is deliberately not a port

- **`Grid`** is a value object. There is one grid model, regular and
  axis-aligned; admitting irregular grids is out of scope by ADR 0001.
- **The trigger.** `poll_once` remains the idempotent unit of work and any
  scheduler drives it. Making the scheduler pluggable would abstract something
  that already composes.
- **Missing-data policy** is a strategy selected by configuration, not a port: it
  is a small closed set, not an extension point for third parties.

## Consequences

- Signatures carry unit-length axes for variables and ensemble members, per
  ADR 0001. Shared code must never squeeze them.
- Five contracts must now be kept stable, and changing one is a runtime minor
  version bump that each model adopts by re-locking.
- The pipeline must be migrated onto these ports; defining them changes no
  behaviour by itself. Until that migration lands, the ports are the contract
  and the existing modules are the implementation that does not yet satisfy it.
- `Sink`, and the output naming that includes a member component, are specified
  ahead of any model that needs them. This is the ADR 0001 trade accepted
  knowingly: verbosity now against a retrofit across every port later.

## Alternatives rejected

- **Merge `Source` and `Decoder` into one provider.** Fewer pieces, but it would
  prevent reusing a decoder across local and remote acquisition, which this
  deployment already does.
- **Fold `Transform` into the `Predictor`.** Scientifically tidy, since both come
  from the same bundle, but it loses the round-trip test seam and the ability to
  log or inspect model-space inputs.
- **Abstract base classes.** Enforced at instantiation and able to carry shared
  behaviour, but they couple every adapter to the framework and to the evolution
  of a base class.
- **Fewer ports now, more later.** Cheaper to read today, but `Source` and
  `Decoder` are already justified by existing code, and deferring them would mean
  migrating the pipeline twice.
