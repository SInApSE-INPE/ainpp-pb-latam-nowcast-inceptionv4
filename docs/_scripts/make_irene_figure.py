"""Render the IRENE ensemble figure from a real published run.

Unlike the other figure, this one cannot be regenerated in CI: it needs the
IRENE artifact bundle, which is 257 MB and not committed. Point it at a run
directory produced locally.

    uv run --group docs python docs/_scripts/make_irene_figure.py \
        --run models/irene/outputs/irene/dpc_sri/2025/08/28/2200Z
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

STATIC = Path(__file__).resolve().parent.parent / "_static"


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", type=Path, required=True, help="A published run directory")
    cli.add_argument("--lead", type=int, default=3, help="Lead to show members for")
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
    leads = sorted({item["lead_hours"] for item in outputs})

    def read(lead: int, member: int):
        item = next(
            entry
            for entry in outputs
            if entry["lead_hours"] == lead and entry["member"] == member
        )
        with rasterio.open(args.run / item["name"]) as dataset:
            return dataset.read(1, masked=True), dataset.bounds, item["valid_time"]

    shown_leads = [leads[0], leads[len(leads) // 4], leads[len(leads) // 2], leads[-1]]
    figure, axes = plt.subplots(2, 4, figsize=(15, 7.4), dpi=150)

    # The colour scale is shared, so panels are comparable rather than each
    # being stretched to its own maximum.
    peak = max(
        float(read(args.lead, member)[0].max() or 0.0) for member in members[:4]
    )
    peak = max(peak, 0.5)

    for axis, member in zip(axes[0], members[:4], strict=False):
        field, bounds, valid = read(args.lead, member)
        image = axis.imshow(
            field,
            origin="upper",
            extent=(bounds.left / 1000, bounds.right / 1000,
                    bounds.bottom / 1000, bounds.top / 1000),
            cmap="turbo",
            vmin=0.0,
            vmax=peak,
        )
        axis.set_title(f"member {member}", fontsize=9)
        axis.tick_params(labelsize=7)
    axes[0][0].set_ylabel("km north", fontsize=8)

    for axis, lead in zip(axes[1], shown_leads, strict=False):
        stack = np.stack([read(lead, member)[0].filled(np.nan) for member in members])
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(stack, axis=0)
        image = axis.imshow(
            mean,
            origin="upper",
            extent=(bounds.left / 1000, bounds.right / 1000,
                    bounds.bottom / 1000, bounds.top / 1000),
            cmap="turbo",
            vmin=0.0,
            vmax=peak,
        )
        axis.set_title(f"ensemble mean, +{lead * 5} min", fontsize=9)
        axis.set_xlabel("km east", fontsize=8)
        axis.tick_params(labelsize=7)
    axes[1][0].set_ylabel("km north", fontsize=8)

    figure.suptitle(
        f"{manifest['model']} issued {manifest['run_time'][:16]}Z — "
        f"{len(members)} members, {len(leads)} leads at "
        f"{int(manifest['step_seconds'] // 60)} min ({manifest['units']})",
        fontsize=11,
    )
    figure.colorbar(image, ax=axes, shrink=0.8, label=manifest["units"])
    destination = STATIC / "irene-ensemble.png"
    figure.savefig(destination, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
