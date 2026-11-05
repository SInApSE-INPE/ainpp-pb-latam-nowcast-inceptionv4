# IRENE

Operational nowcasting on the shared `nowcast-runtime`.

## What is a placeholder

This project was generated and already passes its conformance suite, using a
persistence predictor and a synthesized bundle. Every spot you must change is
marked `REPLACE THIS`:

| File | Replace |
| --- | --- |
| `src/irene_nowcast/model.py` | the architecture and its checkpoint loader |
| `src/irene_nowcast/entrypoint.py` | the native grid, the input format, the source, the bundle roles |
| `config/default.json` | the domain, the product and the remote directory |
| `artifacts/v1/` | the real bundle, with a SHA-256 manifest |

Work in that order and run the suite after each step. It tells you when a
contract is broken, which is cheaper than finding out in production.

## Checks

```bash
uv sync
uv run pytest
uv run ruff check src tests
```

## Run it

The project makes its own sample observations, through the same decoder it
reads with, so they cannot drift from the real format:

```bash
uv run python tools/make_sample_inputs.py --into data/raw
uv run irene-nowcast --config config/default.json \
    --source local --input-dir data/raw --output-dir outputs
```

Each run writes one directory per issue time, holding the forecast files and
a `manifest.json` recording the artifacts, stages, versions and checksums
that produced them.

## What the runtime will not do for you

It loads and runs your network; it never infers its topology. It checks the
contract: units, grid alignment, window spacing, output shape, bundle checksums.
It cannot check whether your model has any skill on the data you point it at.
That stays your job, and is worth writing down beside this file.
