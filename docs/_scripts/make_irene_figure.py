"""Render the IRENE figure from a real published run, beside what was observed.

Cannot be regenerated in CI: it needs the IRENE artifact bundle, which is
257 MB and not committed, and the published DPC sample. Run it locally:

    uv run --group docs python docs/_scripts/make_irene_figure.py \
        --run models/irene/outputs-real/irene/dpc_sri/2025/08/29/0040Z \
        --truth %TEMP%/irene/sample_data.nc --truth-index 33

The top row is what the radar actually recorded; the bottom row is the
ensemble mean forecast for the same moments. One case is an illustration, not
a verification, and the figure caption says so.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

STATIC = Path(__file__).resolve().parent.parent / "_static"
# The deployment domain inside the native DPC mosaic, from Grid.locate.
ROW0, COLUMN0, SIZE = 394, 400, 256


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", type=Path, required=True, help="A published run directory")
    cli.add_argument("--truth", type=Path, help="The DPC datacube, for the observed panels")
    cli.add_argument("--truth-index", type=int, help="Datacube index of the run time")
    cli.add_argument("--leads", type=int, nargs="+", default=[3, 6, 12])
    args = cli.parse_args()

    manifest_path = args.run / "manifest.json"
    if not manifest_path.is_file():
        print(f"No manifest in {args.run}", file=sys.stderr)
        return 1

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import rasterio

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = manifest["outputs"]
    members = sorted({item["member"] for item in outputs})

    def forecast(lead: int, member: int):
        item = next(
            entry for entry in outputs
            if entry["lead_hours"] == lead and entry["member"] == member
        )
        with rasterio.open(args.run / item["name"]) as dataset:
            return dataset.read(1, masked=True).filled(np.nan), dataset.bounds

    def observed(lead: int):
        import h5py

        with h5py.File(args.truth, "r") as handle:
            block = handle["RR"][
                args.truth_index + lead,
                ROW0 : ROW0 + SIZE,
                COLUMN0 : COLUMN0 + SIZE,
            ]
        return np.asarray(block, dtype=np.float32)

    _, bounds = forecast(args.leads[0], 0)
    extent = (
        bounds.left / 1000, bounds.right / 1000,
        bounds.bottom / 1000, bounds.top / 1000,
    )
    has_truth = args.truth is not None and args.truth_index is not None
    rows = 2 if has_truth else 1
    figure, axes = plt.subplots(
        rows, len(args.leads), figsize=(4.6 * len(args.leads), 4.4 * rows), dpi=150
    )
    axes = np.atleast_2d(axes)

    # One scale for every panel, so forecast and observation are comparable.
    peak = 20.0
    step = int(manifest["step_seconds"] // 60)

    def draw(axis, field, title):
        image = axis.imshow(
            field, origin="upper", extent=extent, cmap="turbo", vmin=0.0, vmax=peak
        )
        axis.set_title(title, fontsize=10)
        axis.tick_params(labelsize=7)
        return image

    if has_truth:
        for axis, lead in zip(axes[0], args.leads, strict=True):
            draw(axis, observed(lead), f"observed, +{lead * step} min")
        axes[0][0].set_ylabel("km north", fontsize=8)

    row = axes[-1]
    for axis, lead in zip(row, args.leads, strict=True):
        stack = np.stack([forecast(lead, member)[0] for member in members])
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(stack, axis=0)
        image = draw(axis, mean, f"forecast mean, +{lead * step} min")
        axis.set_xlabel("km east", fontsize=8)
    row[0].set_ylabel("km north", fontsize=8)

    figure.suptitle(
        f"{manifest['model']} on Italian DPC radar, issued "
        f"{manifest['run_time'][:16]}Z — {len(members)} members "
        f"({manifest['units']})",
        fontsize=12,
    )
    figure.colorbar(image, ax=axes, shrink=0.75, label=manifest["units"])
    destination = STATIC / "irene-ensemble.png"
    figure.savefig(destination, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
