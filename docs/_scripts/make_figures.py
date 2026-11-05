"""Render documentation figures from a real pipeline run.

Nothing here is a mock-up: the forecast panels are read from GeoTIFFs this
repository actually produced, so a figure cannot drift from what the code does.
Re-run after changing the pipeline:

    uv run --group docs python docs/_scripts/make_figures.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent.parent / "_static"
PRODUCT = "rain_obs"
RUN_TIME = "2000-01-02T12:00:00Z"


def generate_run(workspace: Path) -> Path:
    """Scaffold a project, give it sample inputs, and publish one run."""
    sys.path.insert(0, str(ROOT / "packages" / "nowcast-runtime" / "src"))
    from nowcast_runtime import scaffold

    target = workspace / "figure-model"
    plan = scaffold.plan_from(
        scaffold.parser().parse_args(
            ["--name", "RainNet", "--product", PRODUCT, "--into", str(target)]
        )
    )
    scaffold.generate(plan)

    sys.path.insert(0, str(target / "src"))
    import numpy as np
    from rainnet_nowcast.entrypoint import compose, decoder, parser  # type: ignore

    from nowcast_runtime import pipeline

    inputs = target / "data" / "raw"
    inputs.mkdir(parents=True)
    outputs = target / "outputs"
    args = parser().parse_args(
        [
            "--config",
            str(target / "config" / "default.json"),
            "--source",
            "local",
            "--input-dir",
            str(inputs),
            "--output-dir",
            str(outputs),
        ]
    )
    deployment, product = compose(args)
    reader = decoder()
    window = deployment.load().predictor.window

    from datetime import UTC, datetime

    latest = datetime.fromisoformat(RUN_TIME.replace("Z", "+00:00")).astimezone(UTC)
    rows = np.linspace(0.0, 1.0, deployment.grid.height, dtype=np.float32)[:, None]
    columns = np.linspace(0.0, 1.0, deployment.grid.width, dtype=np.float32)[None, :]
    for index, observed in enumerate(window.times(latest)):
        field = (rows + columns + index * 0.25).astype(np.float32)
        reader.synthesize(inputs / reader.filename(observed), field, deployment.grid)

    args.run_time = latest
    return pipeline.run(args, deployment, product)


def plot_run(run: Path, destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio

    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    outputs = sorted(manifest["outputs"], key=lambda item: item["lead_hours"])
    figure, axes = plt.subplots(1, len(outputs), figsize=(4 * len(outputs), 3.4), dpi=160)
    axes = axes if len(outputs) > 1 else [axes]

    images = []
    for axis, item in zip(axes, outputs, strict=True):
        with rasterio.open(run / item["name"]) as dataset:
            field = dataset.read(1, masked=True)
            bounds = dataset.bounds
        image = axis.imshow(
            field,
            origin="upper",
            extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
            cmap="viridis",
        )
        images.append(image)
        axis.set_title(f"+{item['lead_hours']} h\n{item['valid_time'][:16]}Z", fontsize=9)
        axis.set_xlabel("longitude", fontsize=8)
        axis.tick_params(labelsize=7)
    axes[0].set_ylabel("latitude", fontsize=8)

    figure.suptitle(
        f"{manifest['model']} forecast issued {manifest['run_time'][:16]}Z "
        f"({manifest['units']})",
        fontsize=10,
    )
    figure.colorbar(images[-1], ax=axes, shrink=0.85, label=manifest["units"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Wrote {destination}")


def main() -> int:
    if shutil.which("python") is None:  # pragma: no cover
        print("python is not on PATH", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory(prefix="nowcast-figures-") as temporary:
        run = generate_run(Path(temporary))
        plot_run(run, STATIC / "forecast-panels.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
