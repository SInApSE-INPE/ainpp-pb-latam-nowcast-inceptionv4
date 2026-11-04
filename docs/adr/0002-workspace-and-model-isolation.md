# 0002 — A workspace for the runtime, isolated projects for models

Status: Accepted
Date: 2026-11-04

## Context

The operational runtime was vendored inside the InceptionV4 package as a private
`_runtime/` subpackage, deliberately, so that one model could be deployed without
any sibling directory. The cost is that a second model means copying the runtime,
after which every fix exists in N divergent copies. Evidence that this had
already begun: the shared artifact downloader carried a branch for a second
model's manifest format.

Two further forces apply:

- The ports will change while the second and third models are integrated. During
  that period, changing a contract and every adapter in one commit is worth more
  than independent release cadence.
- `torch` is installed from a CPU-only index for this model. A single shared lock
  cannot express a CPU deployment and a CUDA deployment at once without extras
  and environment markers.

## Decision

A uv workspace rooted at the repository, whose members are `packages/*` only.
Models live under `models/*` and are explicitly **excluded** from the workspace,
each with its own `pyproject.toml` and `uv.lock`, depending on the runtime
through a path source.

The runtime is consumed as a dependency, never copied. Operational independence
is preserved by the model's own lock file, which pins an exact runtime version,
rather than by duplicating source.

## Consequences

- A runtime fix lands once. Each model adopts it by re-locking, on its own
  schedule.
- Each model chooses its own torch build. The workspace lock covers only the
  runtime and its adapters.
- The runtime and each model are checked by separate commands, because they
  resolve into separate environments. This ergonomic cost is the direct price of
  per-model locks and is documented in the workspace README.
- A model's container build context is the workspace root, since the runtime is a
  path dependency. The root `pyproject.toml` must be part of that context: it
  declares the torch index that the runtime resolves through as a workspace
  member, so `uv sync --locked` is only reproducible when the file is present.
  Replacing the path source with a published version turns the context back into
  the model directory alone and removes this coupling.
- The runtime package is therefore not self-describing about its torch build. A
  consumer installing `nowcast-runtime` from an index chooses its own, which is
  the intended behaviour but worth stating, because it is invisible from inside
  the workspace.
- One ruff configuration at the root governs every package and model.

## Alternatives rejected

- **Separate repositories with published wheels from the start.** Maximum
  independence, but a port change would require coordinating several
  repositories during exactly the period when ports are least stable.
- **Models as workspace members.** One lock for everything, simpler commands,
  but it cannot express divergent torch builds, which is a requirement, not a
  preference.
- **Keep the runtime vendored.** No migration risk, but it does not address the
  problem: the second model still starts by copying the first.

## Reversal

This is intentionally the cheap direction. Workspace to separate repositories is
a move plus a change of dependency source. The opposite, with several models in
production, is not.
