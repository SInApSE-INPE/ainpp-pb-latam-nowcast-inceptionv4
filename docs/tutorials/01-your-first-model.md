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

Generate the project if you have not already, **from the workspace root**:

```bash
uv run nowcast-new-model --name "MyUNet" --product my_product --units "mm/h"
uv sync --project models/myunet
uv run --directory models/myunet pytest    # 19 passed
```

Everything below happens inside `models/myunet/`. Here is the whole project,
with the files you will touch marked:

```text
models/myunet/
├── pyproject.toml                        ← step 3: dependencies
├── README.md
├── Dockerfile
├── compose.yaml
├── config/
│   └── default.json                      ← step 5: domain and product
├── artifacts/
│   └── v1/
│       ├── manifest.json                 ← step 2: your checksums
│       └── placeholder.npy               ← step 2: delete, add your weights
├── src/
│   └── myunet_nowcast/
│       ├── __init__.py
│       ├── __main__.py
│       ├── entrypoint.py                 ← steps 1, 4, 6: composition
│       └── model.py                      ← step 3: your architecture
├── tests/
│   └── test_conformance.py               ← step 8: your own tests go beside it
└── tools/
    └── make_sample_inputs.py
```

Every spot you must change is marked `REPLACE THIS`. To find them all, **from
the workspace root**:

```bash
grep -rn "REPLACE THIS" models/myunet
```

## Step 1: declare what your bundle contains

**File: `models/myunet/src/myunet_nowcast/entrypoint.py`**

Find the `PLUGIN` declaration and change `bundle_roles` to name every file your
artifact bundle holds:

```{code-block} python
:caption: models/myunet/src/myunet_nowcast/entrypoint.py
:emphasize-lines: 5

PLUGIN = ModelPlugin(
    name=NAME,
    slug=SLUG,
    version=__version__,
    bundle_roles=("checkpoint", "mean", "std"),   # name every file you need
    requires_runtime=">=0.3,<0.4",
    load=load_model,
    library_versions=library_versions,
)
```

Roles are **declared, not fixed**. A model with normalization statistics lists
them; a model without simply writes `("checkpoint",)`.

Each role must appear in your bundle's `manifest.json`, and the runtime
verifies every one against its SHA-256 before loading anything.

## Step 2: put your real artifacts in place

**Directory: `models/myunet/artifacts/v1/`**

Delete the placeholder and copy in your weights:

```bash
cd models/myunet
rm artifacts/v1/placeholder.npy
cp /path/to/your/myunet.pt artifacts/v1/
```

Compute the digest of each file:

```bash
sha256sum artifacts/v1/myunet.pt
# Windows PowerShell:  Get-FileHash artifacts/v1/myunet.pt -Algorithm SHA256
```

Then write them into the manifest, one entry per role you declared in step 1:

```{code-block} json
:caption: models/myunet/artifacts/v1/manifest.json

{
  "schema_version": 1,
  "version": "v1",
  "download_base_url": "https://example.org/checkpoints/myunet/v1/",
  "files": {
    "checkpoint": {
      "name": "myunet.pt",
      "sha256": "33d03addba1d70459b65a832396a81dabe11e4c144c32dd136e9c55feec30a17"
    },
    "mean": { "name": "mean.npy", "sha256": "fc60bf59e715a6ffeb3c92d784395d5a4ab3d0527acb66f0881f2011f5d77697" },
    "std":  { "name": "std.npy",  "sha256": "8e09d7e62fbfaf6bcd6177abe9f1f1dfef455119053fc2e6af43a796d3aa5786" }
  }
}
```

```{tip}
Keep large weights out of Git — the generated `.gitignore` already excludes
`*.pt` and `*.pth`. Publish the bundle to an HTTP(S) location, record it as
`download_base_url`, and `myunet-download-artifacts` fetches and verifies each
file against the **local** manifest. Remote metadata is never trusted.
```

## Step 3: replace the predictor

**File: `models/myunet/src/myunet_nowcast/model.py`**

Replace the whole file. The placeholder is persistence; keep the contract and
change the inside.

What the contract requires of your predictor class:

| Attribute | Meaning |
| --- | --- |
| `window` | the history you consume, as offsets ending at the run time |
| `step` | spacing between consecutive output leads |
| `output_steps` | your native horizon in leads |
| `output_variables` | how many fields each lead carries |
| `grid` | the grid this instance was loaded for |
| `leads(inputs, steps)` | yields `steps` arrays, in chronological order |

```{code-block} python
:caption: models/myunet/src/myunet_nowcast/model.py

from collections.abc import Iterator, Mapping
from datetime import timedelta
from pathlib import Path

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

    def leads(self, inputs: np.ndarray, steps: int) -> Iterator[np.ndarray]:
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


def load_model(artifacts: Mapping[str, Path], device: str, grid: Grid):
    resolved = resolve_device(device)
    module = build_my_network(grid.shape)
    state = torch.load(artifacts["checkpoint"], map_location="cpu", weights_only=True)
    module.load_state_dict(state, strict=True)
    return MyUNetPredictor(module.eval().to(resolved), grid, resolved)


def library_versions() -> dict[str, str]:
    return {"torch": torch.__version__, "numpy": np.__version__}
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
completely while nothing raises. This exact mistake was caught in this
repository only by a numerical test — so write one for your model too.
```

```{tip}
Load with `weights_only=True`. A checkpoint is data from somewhere else;
unpickling arbitrary objects from it is a code-execution path. If your
checkpoint will not load that way, convert it once offline — see
[Porting a published model](06-porting-a-published-model.md), step 3.
```

### Declare the dependency

**File: `models/myunet/pyproject.toml`**

```{code-block} toml
:caption: models/myunet/pyproject.toml

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

Then, **from the workspace root**:

```bash
uv lock --project models/myunet
uv sync --project models/myunet
uv run --directory models/myunet pytest
```

## Step 4: describe your input format

**File: `models/myunet/src/myunet_nowcast/entrypoint.py`**

Two things to change: the native grid constant near the top, and the `decoder`
function below it.

```{code-block} python
:caption: models/myunet/src/myunet_nowcast/entrypoint.py

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

**File: `models/myunet/config/default.json`**

```{code-block} json
:caption: models/myunet/config/default.json

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
quietly resampled.

```{note}
Your model may need its own settings — an ensemble size, a seed, a threshold.
Add them as extra keys here; the runtime carries any key it does not own in
`config.extra`, and your composition root reads them from there. Every field
the runtime itself validates is listed in
[Configuration](../guides/configuration.md).
```

## Step 6: add stages, if you need any

**File: `models/myunet/src/myunet_nowcast/entrypoint.py`**

The default is no preprocessing and no postprocessing. Add only what your model
genuinely requires, inside the `Deployment(...)` call in `compose`:

```{code-block} python
:caption: models/myunet/src/myunet_nowcast/entrypoint.py

    deployment = Deployment(
        plugin=PLUGIN,
        source=source,
        decoder=reader,
        sink=GeoTiffSink(variable_names=("my_product",)),
        grid=grid,
        bundle=bundle,
        artifacts=artifacts,
        device=args.device,
        preprocess=(
            ClipNegative(),                       # a -99 sentinel becomes zero rain
            MissingPolicy(config.missing_policy, config.max_missing_fraction),
        ),
    )
```

And a transform, if your checkpoint expects scaled inputs — in the `PLUGIN`
declaration at the top of the same file:

```{code-block} python
:caption: models/myunet/src/myunet_nowcast/entrypoint.py

PLUGIN = ModelPlugin(
    ...,
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

**From the workspace root:**

```bash
uv run --directory models/myunet python tools/make_sample_inputs.py --into data/raw
uv run --directory models/myunet myunet-nowcast \
    --config config/default.json --source local \
    --input-dir data/raw --output-dir outputs
```

Outputs land in `models/myunet/outputs/myunet/my_product/YYYY/MM/DD/HHMMZ/`.
Inspect the manifest beside them, confirm the stages listed are the ones you
intended, and check a GeoTIFF opens where you expect it on the map.

## Step 8: write the tests only you can write

**File: `models/myunet/tests/test_numerics.py`** (new, beside the generated
`test_conformance.py`)

Conformance checks the **contract**. It cannot check whether your model is
correct, and it will never tell you whether it has any skill. Two things are
worth writing yourself.

**A numerical equivalence test.** Run the pipeline and compare against the same
computation done directly, so a refactor cannot silently change the numbers:

```{code-block} python
:caption: models/myunet/tests/test_numerics.py

def test_output_matches_a_direct_computation(case, module):
    target = run(args_for(case))
    manifest = json.loads((target / "manifest.json").read_text())
    expected = direct_computation(case["inputs"], module)
    with rasterio.open(target / manifest["outputs"][0]["name"]) as dataset:
        np.testing.assert_allclose(dataset.read(1), expected[0], atol=1e-6)
```

**File: `models/myunet/VALIDATION.md`** (new)

A record of what you have **not** verified. Write down the assumptions you are
running on — that latitude order, that this product resembles what you trained
on, that the domain matches. Prose is the right tool here, and the honesty is
worth more than the tidiness.

## Checklist

- [ ] `entrypoint.py` — `bundle_roles` names every file your bundle holds
- [ ] `artifacts/v1/manifest.json` — lists each with a real SHA-256
- [ ] `model.py` — the predictor declares `window`, `step`, `output_steps`, `output_variables`
- [ ] `model.py` — the loader returns a module in **evaluation** mode
- [ ] `entrypoint.py` — the decoder describes your real format and file names
- [ ] `config/default.json` — the domain fits inside the native grid
- [ ] `entrypoint.py` — stages, if any, are in the right place: plugin or configuration
- [ ] `pyproject.toml` — your framework is declared, and the lock is updated
- [ ] conformance passes
- [ ] `tests/test_numerics.py` — you have a numerical test of your own
- [ ] `VALIDATION.md` — you have written down what you have **not** verified

## Next

- [Adding a new data source](02-a-new-data-source.md)
- [Ensembles and several variables](03-ensembles-and-variables.md)
- [Porting a published model](06-porting-a-published-model.md) — the same
  journey done for a real third-party model, with everything that went wrong
