# Configuration

One JSON file per deployment. It holds what changes between deployments of the
same model — the domain, the provider, the data policy — and nothing that is a
property of the checkpoint.

## A complete example

```json
{
  "artifacts": "../artifacts/v1",
  "domain": {
    "west": -120.0,
    "south": -55.0,
    "east": -23.0,
    "north": 33.0
  },
  "model_latitude_order": "north_to_south",
  "product": "gsmap_now",
  "remote_directory": "/now/latest",
  "missing_policy": "normalized_zero",
  "max_missing_fraction": 0.05,
  "ftp": {
    "timeout_seconds": 60,
    "attempts": 3,
    "retry_delay_seconds": 30,
    "max_retry_delay_seconds": 120
  }
}
```

## Fields

`artifacts` *(path)*
: The bundle directory, resolved **relative to the configuration file** so a
  project can be moved without editing it.

`domain` *(bounds)*
: `west`, `south`, `east`, `north`. **Bounds only** — no resolution, no CRS.
  Those come from the decoder, and the runtime reconciles the two. A domain
  that is not a cell-aligned subgrid of the provider's grid is refused with a
  specific reason. See [Grids and windows](../concepts/grids-and-windows.md).

`model_latitude_order` *(`north_to_south` | `south_to_north`)*
: The row order the checkpoint was trained on. When it is `south_to_north`,
  the composition root adds a `FlipLatitude` pair as model-side stages.

`product` *(string)*
: The input product identifier, used in file names and in the output path.

`remote_directory` *(string)*
: Where a remote source lists observations. A `%` makes it an strftime
  template for an archive, which requires an explicit `--run-time` and cannot
  be used with `--scheduled`.

`missing_policy` *(`error` | `zero_rain` | `normalized_zero`)*
: What to do about non-finite cells. See
  [Processing stages](../concepts/stages.md).

`max_missing_fraction` *(0 ≤ x < 1)*
: How much of a field may be missing before the run is refused.

`ftp` *(object)*
: Transfer policy: `timeout_seconds`, `attempts`, `retry_delay_seconds`,
  `max_retry_delay_seconds`. Retries back off exponentially, capped at the
  maximum.

## What is deliberately not here

```{important}
**Credentials.** Read from the environment only, never from a file that might
be committed, logged or copied into a manifest.

**The transform.** Its statistics are inseparable from the checkpoint. A
configurable transform would let a mismatched pair run cleanly and publish
plausible nonsense.

**Resolution and CRS.** They belong to the provider. Stating them in two
places invites them to disagree.

**The architecture.** The runtime loads and runs a network, it never infers
one.
```

## The rule for anything new

> What is scientifically coupled to the checkpoint belongs to the plugin.
> What is operationally variable belongs to the configuration.

Applied: latitude order is checkpoint-coupled, so the plugin owns the stage
even though an operator supplies the value. Clipping a provider sentinel is
source-coupled, so configuration owns it.

## Validating without running

Configuration errors surface during composition, before any data is touched:

```bash
uv run myunet-nowcast --config config/default.json --source local \
    --input-dir /nonexistent
```

A bad domain, a missing artifact role, a failed checksum or an unknown policy
all fail here with a specific message rather than part-way through a run.

## Several deployments of one model

Keep one file per domain or per provider:

```text
config/
├── latam.json
├── latam-archive.json      # remote_directory as an strftime template
└── test-small.json         # a 128 x 128 domain, for fast checks
```

They share the same code and the same bundle; only the file differs.

## Where this is going

Selecting a source or decoder is still Python in the composition root, not a
string in this file. A registry would make that declarative:

```json
{
  "source":  {"kind": "jaxa-ftp",     "options": {"directory": "/now/latest"}},
  "decoder": {"kind": "raw-binary",   "options": {"product": "gsmap_now"}}
}
```

That is deliberately not built yet. Without a second real adapter to validate
the dispatch, the design of the registry would be a guess, and a wrong
extension point is more expensive than a missing one.
