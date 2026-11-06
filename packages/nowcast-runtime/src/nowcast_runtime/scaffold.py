"""Generate a model project that already passes conformance.

The skeleton ships a persistence predictor and a synthesized bundle, so the
generated tests are green before the adopter writes a line. They then replace one
function with their own architecture and watch what breaks. Starting from
something that works, rather than from something that must be assembled before
it can be run, is the whole point.

Templates use $-substitution so that generated Python can contain braces freely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from string import Template

# This file lives at <runtime>/src/nowcast_runtime/scaffold.py, so the package
# root is two levels up. Installed into site-packages there is no pyproject
# alongside it, which is exactly how a path source is ruled out.
RUNTIME_PACKAGE = Path(__file__).resolve().parents[2]
PLACEHOLDER = "REPLACE THIS"


@dataclass(frozen=True)
class Plan:
    name: str
    slug: str
    package: str
    product: str
    units: str
    target: Path
    runtime_source: str

    @property
    def substitutions(self) -> dict[str, str]:
        return {
            "name": self.name,
            "slug": self.slug,
            "package": self.package,
            "product": self.product,
            "units": self.units,
            "runtime_source": self.runtime_source,
            "placeholder": PLACEHOLDER,
        }


PYPROJECT = Template('''[build-system]
requires = ["hatchling>=1.27,<2"]
build-backend = "hatchling.build"

[project]
name = "$slug-nowcast"
version = "0.1.0"
description = "$name operational nowcasting"
readme = "README.md"
requires-python = ">=3.11,<3.14"
dependencies = [
    "nowcast-runtime>=0.3,<0.4",
    "numpy>=1.26,<3",
    "rasterio>=1.4,<2",
]

[project.scripts]
$slug-nowcast = "$package.entrypoint:main"
$slug-serve = "$package.entrypoint:serve"
$slug-download-artifacts = "nowcast_runtime.download_artifacts:main"

[dependency-groups]
dev = ["pytest>=8,<10", "ruff>=0.11,<1"]

[tool.uv.sources]
$runtime_source

[tool.hatch.build.targets.wheel]
packages = ["src/$package"]

[tool.pytest.ini_options]
testpaths = ["tests"]
''')

CONFIG = Template('''{
  "artifacts": "../artifacts/v1",
  "domain": {
    "west": -10.0,
    "south": 0.0,
    "east": 0.0,
    "north": 5.0
  },
  "model_latitude_order": "north_to_south",
  "product": "$product",
  "remote_directory": "/latest",
  "missing_policy": "zero_rain",
  "max_missing_fraction": 0.05,
  "ftp": {
    "timeout_seconds": 60,
    "attempts": 3,
    "retry_delay_seconds": 30,
    "max_retry_delay_seconds": 120
  }
}
''')

INIT = Template('''"""$name operational nowcasting."""

__version__ = "0.1.0"
''')

MAIN = Template('''from .entrypoint import main

raise SystemExit(main())
''')

MODEL = Template('''"""The $name architecture and how to load it.

$placeholder: everything below is a persistence placeholder so that the
generated project runs and passes conformance from the start. Replace
`load_model` with your own checkpoint loader and `Persistence` with your own
predictor. Keep the shape of the contract:

- `window` is the history you consume, `step` the spacing between leads
- `output_steps` is your native horizon, `output_variables` how many fields a
  lead carries
- `leads` yields bare arrays shaped (members, variables, rows, columns); lead
  numbering, valid times and the inherited mask are the runtime's job

Inputs arrive as (time, variables, rows, columns) in model space. Create tensors
and place them on a device here, not in the runtime: no contract mentions a deep
learning framework, which is what keeps another one from needing a new port.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from datetime import timedelta
from pathlib import Path

import numpy as np
from nowcast_runtime.ports import Grid, WindowSpec

STEP = timedelta(hours=1)
INPUT_STEPS = 3
OUTPUT_STEPS = 3


class Persistence:
    """$placeholder: repeats the most recent observation at every lead."""

    def __init__(self, grid: Grid):
        self.grid = grid
        self.step = STEP
        self.window = WindowSpec.uniform(STEP, INPUT_STEPS)
        self.output_steps = OUTPUT_STEPS
        self.output_variables = 1

    def leads(self, inputs: np.ndarray, steps: int) -> Iterator[np.ndarray]:
        if not 1 <= steps <= self.output_steps:
            raise ValueError(f"steps must be between 1 and {self.output_steps}")
        latest = inputs[-1]
        for _ in range(steps):
            yield latest[np.newaxis].astype(np.float32)


def load_model(artifacts: Mapping[str, Path], device: str, grid: Grid) -> Persistence:
    """$placeholder: rebuild your network and load its weights.

    `artifacts` holds the verified path of every role the plugin declared, so
    read your checkpoint from `artifacts["checkpoint"]` once you declare it.
    """
    return Persistence(grid)


def library_versions() -> dict[str, str]:
    """Recorded in every run manifest. Add the libraries your model relies on."""
    return {"numpy": np.__version__}
''')

ENTRYPOINT = Template('''"""The composition root: the only file knowing both model and adapters.

Because composition lives here, the runtime never imports an adapter and an
adapter never imports the pipeline. Changing the source, the input format or the
output format is a change to this file and to the configuration, not to any
shared code.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from nowcast_runtime import pipeline, service
from nowcast_runtime.artifacts import verify_bundle
from nowcast_runtime.builtin.decoders import RawBinaryDecoder, TimestampedName
from nowcast_runtime.builtin.sinks import GeoTiffSink
from nowcast_runtime.builtin.sources import Credentials, FtpSource, LocalFiles
from nowcast_runtime.builtin.stages import MissingPolicy
from nowcast_runtime.composition import Deployment
from nowcast_runtime.config import Config
from nowcast_runtime.ports import Grid, ModelPlugin

from . import __version__
from .model import library_versions, load_model

NAME = "$name"
SLUG = "$slug"
PRODUCT = "$product"
UNITS = "$units"
CONFIG = "config/default.json"
POLL_ARGS = ["--config", CONFIG, "--source", "ftp", "--scheduled", "--output-dir", "outputs"]

# $placeholder: the grid your provider publishes, which requested domains must
# fit inside. A decoder owns this, so the runtime never guesses it.
NATIVE_GRID = Grid("EPSG:4326", -180.0, 90.0, 0.5, 0.5, 720, 360)

PLUGIN = ModelPlugin(
    name=NAME,
    slug=SLUG,
    version=__version__,
    # $placeholder: name every file your bundle contains. Declared, not fixed:
    # a model with no normalization statistics simply does not list them.
    bundle_roles=("checkpoint",),
    requires_runtime=">=0.3,<0.4",
    load=load_model,
    library_versions=library_versions,
    # transform, preprocess and postprocess are optional. Leaving them out means
    # the runtime applies identity and runs no stages.
)


def decoder() -> RawBinaryDecoder:
    """$placeholder: describe your input format.

    A raw binary field is configuration, not code: byte order and width through
    dtype, the grid it covers, any column roll needed to reach that grid origin,
    the units, and the file name contract. For another format, swap this for a
    different decoder; nothing else changes.
    """
    return RawBinaryDecoder(
        native_grid=NATIVE_GRID,
        naming=TimestampedName(
            pattern=rf"{re.escape(PRODUCT)}\\.(?P<stamp>\\d{{12}})\\.bin",
            stamp_format="%Y%m%d%H%M",
            template=f"{PRODUCT}.%Y%m%d%H%M.bin",
        ),
        units=UNITS,
        dtype="<f4",
        gzipped=False,
        suffix=".bin",
    )


def compose(args: argparse.Namespace) -> tuple[Deployment, str]:
    """Build one deployment from the configuration and the command line."""
    config = Config.load(args.config)
    reader = decoder()
    grid = config.domain.to_grid(reader.native_grid)

    if args.source == "local":
        if args.input_dir is None:
            raise ValueError("--source local requires --input-dir")
        source = LocalFiles(args.input_dir, reader.glob)
    else:
        # $placeholder: point this at your provider, or swap in another source.
        source = FtpSource(
            directory=config.remote_directory,
            cache=args.cache_dir,
            credentials=Credentials.from_env(
                "example.invalid",
                "${slug_upper}_FTP_HOST",
                "${slug_upper}_FTP_USER",
                "${slug_upper}_FTP_PASSWORD",
            ),
            retry=config.ftp,
            # A transfer counts as complete only once it decodes on this grid.
            verify=lambda path: reader.read(path, grid),
        )

    bundle, artifacts = verify_bundle(config.artifacts, PLUGIN.bundle_roles)
    deployment = Deployment(
        plugin=PLUGIN,
        source=source,
        decoder=reader,
        sink=GeoTiffSink(variable_names=("$product",)),
        grid=grid,
        bundle=bundle,
        artifacts=artifacts,
        device=args.device,
        preprocess=(MissingPolicy(config.missing_policy, config.max_missing_fraction),),
    )
    return deployment, config.product


def parser():
    return pipeline.parser(SLUG, NAME)


def run(args):
    deployment, product = compose(args)
    return pipeline.run(args, deployment, product)


def main(argv=None):
    return pipeline.main(compose, SLUG, NAME, argv)


def serve():
    """Container entrypoint: verify the bundle, then poll on one cadence."""
    from nowcast_runtime import download_artifacts

    return service.serve(
        NAME,
        download=lambda: download_artifacts.main(["--bundle", str(Path("artifacts/v1"))]),
        poll=lambda: main(POLL_ARGS),
    )
''')

CONFORMANCE = Template('''"""$name through the shared conformance suite.

This runs the real composition root, so what passes here is the wiring the
container uses. The suite synthesizes its own observations through the
deployment's decoder, so it needs no real data.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from nowcast_runtime.testing import ConformanceSuite

from $package.entrypoint import compose, parser

PROJECT = Path(__file__).resolve().parents[1]


class $class_name(ConformanceSuite):
    product = "$product"
    run_time = datetime(2000, 1, 2, 12, 0, tzinfo=UTC)

    def build(self, tmp_path):
        payload = json.loads((PROJECT / "config/default.json").read_text(encoding="utf-8"))
        payload["artifacts"] = str(PROJECT / "artifacts/v1")
        config = tmp_path / "config.json"
        config.write_text(json.dumps(payload), encoding="utf-8")
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        args = parser().parse_args(
            ["--config", str(config), "--source", "local", "--input-dir", str(inputs)]
        )
        deployment, product = compose(args)
        assert product == self.product
        return deployment
''')

SAMPLE_INPUTS = Template("""\"\"\"Write sample observations this project can run on.

Uses the deployment's own decoder, so the files are exactly what it expects
and nothing here can drift from the real format.
\"\"\"

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from $package.entrypoint import compose, decoder
from $package.entrypoint import parser as cli_parser


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--into", type=Path, default=Path("data/raw"))
    cli.add_argument("--config", type=Path, default=Path("config/default.json"))
    cli.add_argument(
        "--run-time",
        help="Timestamp of the LAST observation, e.g. 2000-01-02T12:00:00Z",
    )
    args = cli.parse_args()
    args.into.mkdir(parents=True, exist_ok=True)

    probe = cli_parser().parse_args(
        ["--config", str(args.config), "--source", "local", "--input-dir", str(args.into)]
    )
    deployment, _ = compose(probe)
    grid = deployment.grid
    reader = decoder()
    window = deployment.load().predictor.window

    if args.run_time is None:
        latest = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    else:
        latest = datetime.fromisoformat(args.run_time.replace("Z", "+00:00"))

    rows = np.linspace(0.0, 1.0, grid.height, dtype=np.float32)[:, None]
    columns = np.linspace(0.0, 1.0, grid.width, dtype=np.float32)[None, :]
    written = []
    for index, observed in enumerate(window.times(latest)):
        values = (rows + columns + index * 0.25).astype(np.float32)
        path = reader.synthesize(args.into / reader.filename(observed), values, grid)
        written.append(path)

    print(f"Wrote {len(written)} observations to {args.into}")
    for path in written:
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""")


README = Template('''# $name

Operational nowcasting on the shared `nowcast-runtime`.

## What is a placeholder

This project was generated and already passes its conformance suite, using a
persistence predictor and a synthesized bundle. Every spot you must change is
marked `$placeholder`:

| File | Replace |
| --- | --- |
| `src/$package/model.py` | the architecture and its checkpoint loader |
| `src/$package/entrypoint.py` | the native grid, the input format, the source, the bundle roles |
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
uv run $slug-nowcast --config config/default.json \\
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
''')

DOCKERFILE = Template('''# Build context is the workspace root, because nowcast-runtime is a path
# dependency:  docker build -f models/$slug/Dockerfile .
FROM python:3.12-slim-bookworm AS builder

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \\
    UV_CACHE_DIR=/opt/uv-cache
WORKDIR /build
RUN python -m pip install --no-cache-dir uv==0.11.32
COPY packages/nowcast-runtime ./packages/nowcast-runtime
COPY models/$slug/pyproject.toml models/$slug/uv.lock models/$slug/README.md ./models/$slug/
COPY models/$slug/src ./models/$slug/src
RUN --mount=type=cache,target=/opt/uv-cache \\
    uv sync --locked --no-dev --no-editable \\
      --project models/$slug --python /usr/local/bin/python

FROM python:3.12-slim-bookworm

# LOGNAME: some libraries call getpass.getuser() while importing, which falls
# back to an /etc/passwd lookup. This image runs under an arbitrary host UID, so
# no baked passwd entry can cover it; getpass reads LOGNAME first.
ENV PATH=/opt/venv/bin:$$PATH \\
    PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    HOME=/tmp \\
    LOGNAME=nowcast
RUN apt-get update \\
    && apt-get install -y --no-install-recommends libexpat1 libgomp1 \\
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY models/$slug/config ./config
COPY models/$slug/artifacts/v1/manifest.json ./artifacts/v1/
RUN mkdir -p outputs \\
    && chmod -R a+rX /app \\
    && chown -R 10001:10001 artifacts outputs
USER 10001:10001
CMD ["$slug-serve"]
''')

COMPOSE = Template('''services:
  $slug:
    build:
      # The runtime lives outside this directory, so the context is the
      # workspace root and the Dockerfile path is relative to it.
      context: ../..
      dockerfile: models/$slug/Dockerfile
    image: $slug-nowcast:local
    restart: unless-stopped
    stop_grace_period: 20m
    env_file: .env
    volumes:
      - ./artifacts/v1:/app/artifacts/v1
      - ./outputs:/app/outputs
''')

GITIGNORE = """.venv/
__pycache__/
# Trailing glob: a run written to outputs-real/ or data/real/ is still output.
outputs*/
data*/
*.pth
*.pt
*.ckpt
*.safetensors
.env
.env.*
deploy/generated/
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_source(target: Path) -> str:
    """Point at a checked-out runtime, or leave resolution to an index."""
    if not (RUNTIME_PACKAGE / "pyproject.toml").is_file():
        return "# nowcast-runtime resolves from an index; no path source needed"
    relative = Path(os.path.relpath(RUNTIME_PACKAGE, target)).as_posix()
    return f'nowcast-runtime = {{ path = "{relative}", editable = true }}'


def render(plan: Plan) -> dict[str, str]:
    """Every generated text file, keyed by its path inside the project."""
    values = plan.substitutions
    values["slug_upper"] = plan.slug.upper().replace("-", "_")
    # A display name may contain anything; a class name may not.
    safe = "".join(part for part in plan.name.title() if part.isalnum())
    values["class_name"] = f"Test{safe or plan.package.title().replace('_', '')}"
    return {
        "pyproject.toml": PYPROJECT.substitute(values),
        "README.md": README.substitute(values),
        "Dockerfile": DOCKERFILE.substitute(values),
        "compose.yaml": COMPOSE.substitute(values),
        ".gitignore": GITIGNORE,
        "config/default.json": CONFIG.substitute(values),
        f"src/{plan.package}/__init__.py": INIT.substitute(values),
        f"src/{plan.package}/__main__.py": MAIN.substitute(values),
        f"src/{plan.package}/model.py": MODEL.substitute(values),
        f"src/{plan.package}/entrypoint.py": ENTRYPOINT.substitute(values),
        "tests/test_conformance.py": CONFORMANCE.substitute(values),
        "tools/make_sample_inputs.py": SAMPLE_INPUTS.substitute(values),
    }


def write_bundle(bundle: Path) -> list[Path]:
    """A placeholder bundle with a real manifest, so conformance runs at once."""
    import numpy as np

    bundle.mkdir(parents=True, exist_ok=True)
    checkpoint = bundle / "placeholder.npy"
    np.save(checkpoint, np.zeros(1, dtype=np.float32))
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "version": "v1",
                "placeholder": True,
                "files": {
                    "checkpoint": {"name": checkpoint.name, "sha256": sha256(checkpoint)}
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return [checkpoint, bundle / "manifest.json"]


def plan_from(args: argparse.Namespace) -> Plan:
    slug = args.slug or args.name.lower().replace(" ", "-")
    if not slug.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Slug must be a path and filename safe token")
    target = (args.into or Path("models") / slug).resolve()
    package = args.package or f"{slug.replace('-', '_')}_nowcast"
    if not package.isidentifier():
        raise ValueError(f"Derived package name is not importable: {package}")
    return Plan(
        name=args.name,
        slug=slug,
        package=package,
        product=args.product or f"{slug}_product",
        units=args.units,
        target=target,
        runtime_source=runtime_source(target),
    )


def generate(plan: Plan, force: bool = False) -> list[Path]:
    if plan.target.exists() and any(plan.target.iterdir()) and not force:
        raise FileExistsError(f"{plan.target} is not empty; use --force to write into it")
    written = []
    for relative, content in render(plan).items():
        path = plan.target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        written.append(path)
    written.extend(write_bundle(plan.target / "artifacts" / "v1"))
    return written


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        description="Generate a model project that already passes conformance"
    )
    cli.add_argument("--name", required=True, help="Display name, e.g. MyUNet")
    cli.add_argument("--slug", help="Path and filename safe token; defaults from the name")
    cli.add_argument("--package", help="Python package name; defaults to <slug>_nowcast")
    cli.add_argument("--product", help="Input product identifier used in file names")
    cli.add_argument("--units", default="mm/h", help="Physical units of the data")
    cli.add_argument("--into", type=Path, help="Target directory; defaults to models/<slug>")
    cli.add_argument("--force", action="store_true", help="Write into a non-empty directory")
    return cli


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        plan = plan_from(args)
        written = generate(plan, force=args.force)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Could not generate the project: {exc}", file=sys.stderr)
        return 1
    print(f"Created {plan.target} with {len(written)} files.")
    print("\nNext:")
    print(f"  uv sync --project {plan.target}")
    print(f"  uv run --directory {plan.target} pytest")
    print(f"\nThe suite should pass now. Then replace every {PLACEHOLDER} marker,")
    print("running it again after each change. README.md lists them in order.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
