# Your first model

This walks through turning the generated placeholder into a real deployment:
your architecture, your data format, your domain, your artifacts.

Work in the order below and run the conformance suite after each step. That is
the whole method — you always have a working baseline, so a failure points at
the one thing you just changed.

```{contents} On this page
:local:
:depth: 1
```

## Before you start

You need a **trained checkpoint** and the **code that rebuilds its
architecture**. The runtime loads and runs a network; it never infers topology
from weights. If your architecture lives in a training repository, you will be
copying or importing that code.

Generate the project if you have not already:

```bash
uv run nowcast-new-model --name "MyUNet" --product my_product --units "mm/h"
uv sync --project models/myunet
uv run --directory models/myunet pytest    # 19 passed
```

Every spot you must change is marked `REPLACE THIS`. To find them all:

```bash
grep -rn "REPLACE THIS" models/myunet
```

## Step 1: declare what your bundle contains

Open `src/myunet_nowcast/entrypoint.py`. The plugin declares which files your
artifact bundle holds:

```python
PLUGIN = ModelPlugin(
    name=NAME,
    slug=SLUG,
    version=__version__,
    bundle_roles=("checkpoint",),      # <- declare every file you need
    requires_runtime=">=0.3,<0.4",
    load=load_model,
    library_versions=library_versions,
)
```

Roles are **declared, not fixed**. A model with normalization statistics lists
them; a model without simply does not:

```python
    bundle_roles=("checkpoint", "mean", "std"),
```

Each role must appear in your bundle's `manifest.json`, and the runtime
verifies every one against its SHA-256 before loading anything.

## Step 2: put your real artifacts in place

Replace `artifacts/v1/` with your checkpoint and a manifest naming it:

```json
{
  "schema_version": 1,
  "version": "v1",
  "files": {
    "checkpoint": {
      "name": "myunet.pt",
      "sha256": "33d03addba1d70459b65a832396a81dabe11e4c144c32dd136e9c55feec30a17"
    }
  }
}
```

Compute the digest with `sha256sum myunet.pt`, or on Windows
`Get-FileHash myunet.pt -Algorithm SHA256`.

```{tip}
Keep large weights out of Git. Publish the bundle to an HTTP(S) location and
record it as `download_base_url` in the manifest; the generated
`<slug>-download-artifacts` command then fetches and verifies each file
against the **local** manifest. Remote metadata is never trusted.
See [Artifacts and manifests](../concepts/artifacts-and-manifests.md).
```

## Step 3: replace the predictor

Open `src/myunet_nowcast/model.py`. The placeholder is persistence; your job is
to keep the contract and change the inside.

What the contract requires:

| Attribute | Meaning |
| --- | --- |
| `window` | the history you consume, as offsets ending at the run time |
| `step` | spacing between consecutive output leads |
| `output_steps` | your native horizon in leads |
| `output_variables` | how many fields each lead carries |
| `grid` | the grid this instance was loaded for |
| `leads(inputs, steps)` | yields `steps` arrays, in chronological order |

```python
from datetime import timedelta

import numpy as np
import torch
from nowcast_runtime.ports import Grid, WindowSpec

STEP = timedelta(hours=1)


class MyUNetPredictor:
    def __init__(self, module, grid: Grid, device: torch.device):
        self.module = module
        self.grid = grid
        self.device = device
        self.step = STEP
        self.window = WindowSpec.uniform(STEP, module.input_steps)
        self.output_steps = module.output_steps
        self.output_variables = 1

    def leads(self, inputs: np.ndarray, steps: int):
        # inputs is (time, variables, rows, columns), already in model space.
        tensor = torch.from_numpy(np.ascontiguousarray(inputs[:, 0])).unsqueeze(0)
        self.module.eval()
        with torch.inference_mode():
            prediction = self.module(tensor.to(self.device))
            if not torch.isfinite(prediction).all():
                raise ValueError("Model produced non-finite predictions")
            fields = prediction[0].cpu().numpy()
        for lead in range(steps):
            # (variables, rows, columns) gains a single-member leading axis.
            yield fields[lead][np.newaxis]
```

```{important}
**Arrays cross the boundary as NumPy.** Tensor creation and device placement
live in your model, not in the runtime. No contract names a deep learning
framework, which is what lets a TensorFlow or JAX model use this platform
without any new port.
```

```{warning}
**Return your module in evaluation mode.** A loader that hands back a module in
training mode makes BatchNorm use batch statistics, and the output changes
completely while nothing raises. This exact mistake was caught here only by a
numerical test — so write one for your model too.
```

Then the loader:

```python
def load_model(artifacts: Mapping[str, Path], device: str, grid: Grid):
    resolved = resolve_device(device)
    module = build_my_network(grid.shape)
    state = torch.load(artifacts["checkpoint"], map_location="cpu", weights_only=True)
    module.load_state_dict(state, strict=True)
    return MyUNetPredictor(module.eval().to(resolved), grid, resolved)
```

```{tip}
Load with `weights_only=True`. A checkpoint is data from somewhere else;
unpickling arbitrary objects from it is a code-execution path.
```

Finally, declare what you depend on, which goes into every run manifest:

```python
def library_versions() -> dict[str, str]:
    return {"torch": torch.__version__, "numpy": np.__version__}
```

And add the dependency to `pyproject.toml`, choosing your own build:

```toml
dependencies = [
    "nowcast-runtime>=0.3,<0.4",
    "numpy>=1.26,<3",
    "rasterio>=1.4,<2",
    "torch==2.10.0",
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
```

Re-lock and run the suite:

```bash
uv lock --project models/myunet
uv sync --project models/myunet
uv run --directory models/myunet pytest
```

## Step 4: describe your input format

Back in `entrypoint.py`, the native grid and the decoder:

```python
# The grid your provider publishes, which requested domains must fit inside.
NATIVE_GRID = Grid("EPSG:4326", -180.0, 60.0, 0.1, 0.1, 3600, 1200)


def decoder() -> RawBinaryDecoder:
    return RawBinaryDecoder(
        native_grid=NATIVE_GRID,
        naming=TimestampedName(
            pattern=r"my_product\.(?P<stamp>\d{8}\.\d{4})\.dat\.gz",
            stamp_format="%Y%m%d.%H%M",
            template="my_product.%Y%m%d.%H%M.dat.gz",
        ),
        units="mm/h",
        dtype="<f4",
        roll_columns=0,
        gzipped=True,
        suffix=".dat.gz",
    )
```

A headerless binary field is **configuration, not code**. Everything that
varies is a parameter: byte order and width through `dtype`, the grid it
covers, any column roll needed to reach that grid origin, the units, and the
file name contract.

```{admonition} Sub-hourly data works too
:class: tip

`stamp_format` reads the whole timestamp, so a ten-minute product is just a
different format string:

    pattern=r"radar\.(?P<stamp>\d{12})\.bin",
    stamp_format="%Y%m%d%H%M",
    template="radar.%Y%m%d%H%M.bin",
```

If your format is not a raw binary block, see
[Adding a new data source](02-a-new-data-source.md).

## Step 5: set your domain

`config/default.json` holds what changes per deployment:

```json
{
  "artifacts": "../artifacts/v1",
  "domain": { "west": -120.0, "south": -55.0, "east": -23.0, "north": 33.0 },
  "model_latitude_order": "north_to_south",
  "product": "my_product",
  "remote_directory": "/now/latest",
  "missing_policy": "zero_rain",
  "max_missing_fraction": 0.05,
  "ftp": { "timeout_seconds": 60, "attempts": 3,
           "retry_delay_seconds": 30, "max_retry_delay_seconds": 120 }
}
```

The domain is **bounds only**. Resolution and CRS come from the decoder, and
the runtime negotiates the two: a request that is not a cell-aligned subgrid of
what the provider publishes is refused with a specific reason rather than
quietly resampled. Every field is documented in
[Configuration](../guides/configuration.md).

## Step 6: add stages, if you need any

The default is no preprocessing and no postprocessing. Add only what your model
genuinely requires:

```python
    deployment = Deployment(
        ...
        preprocess=(
            ClipNegative(),                       # a -99 sentinel becomes zero rain
            MissingPolicy(config.missing_policy, config.max_missing_fraction),
        ),
    )
```

And a transform, if your checkpoint expects scaled inputs:

```python
PLUGIN = ModelPlugin(
    ...
    transform=lambda artifacts: LogZScore.from_arrays(artifacts["mean"], artifacts["std"]),
)
```

```{important}
**A transform is bound by the plugin, never by configuration.** Its statistics
are inseparable from the checkpoint. A configurable transform would let a
mismatched pair raise nothing while producing plausible, meaningless forecasts.

The rule this follows, which also decides where a new stage belongs: *what is
scientifically coupled to the checkpoint belongs to the plugin; what is
operationally variable belongs to the configuration.*
```

[Processing stages](../concepts/stages.md) explains ordering, which is load
bearing and fixed rather than composed.

## Step 7: run it for real

```bash
uv run --directory models/myunet python tools/make_sample_inputs.py --into data/raw
uv run --directory models/myunet myunet-nowcast \
    --config config/default.json --source local \
    --input-dir data/raw --output-dir outputs
```

Inspect the manifest, confirm the stages listed are the ones you intended, and
check a GeoTIFF opens where you expect it on the map.

## Step 8: write the tests only you can write

Conformance checks the **contract**. It cannot check whether your model is
correct, and it will never tell you whether it has any skill. Two tests are
worth writing yourself:

**A numerical equivalence test.** Run the pipeline and compare against the same
computation done directly, so a refactor cannot silently change the numbers:

```python
def test_output_matches_a_direct_computation(case, module):
    target = run(args_for(case))
    manifest = json.loads((target / "manifest.json").read_text())
    expected = direct_computation(case["inputs"], module)
    with rasterio.open(target / manifest["outputs"][0]["name"]) as dataset:
        np.testing.assert_allclose(dataset.read(1), expected[0], atol=1e-6)
```

**A record of what you have not verified.** Write down the assumptions you are
running on — that latitude order, that this product resembles what you trained
on, that the domain matches. Prose is the right tool here, and the honesty is
worth more than the tidiness.

## Checklist

- [ ] `bundle_roles` names every file your bundle holds
- [ ] `artifacts/v1/manifest.json` lists each with a real SHA-256
- [ ] the predictor declares `window`, `step`, `output_steps`, `output_variables`
- [ ] the loader returns a module in **evaluation** mode
- [ ] the decoder describes your real format and file names
- [ ] the configured domain fits inside the native grid
- [ ] stages, if any, are in the right place: plugin or configuration
- [ ] conformance passes
- [ ] you have a numerical test of your own
- [ ] you have written down what you have **not** verified

## Next

- [Adding a new data source](02-a-new-data-source.md)
- [Ensembles and several variables](03-ensembles-and-variables.md)
- [Running in a container](04-running-in-a-container.md)
