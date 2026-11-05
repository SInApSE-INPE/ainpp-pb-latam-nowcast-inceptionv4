"""Write sample radar scans this project can run on.

Uses the deployment's own decoder, so the files are exactly what it expects.
A moving, decaying cell over an otherwise dry domain, with the top rows left
as gaps so the coverage mask is exercised end to end.
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from irene_nowcast.entrypoint import compose, decoder
from irene_nowcast.entrypoint import parser as cli_parser


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--into", type=Path, default=Path("data/raw"))
    cli.add_argument("--config", type=Path, default=Path("config/default.json"))
    cli.add_argument("--run-time", help="Timestamp of the LAST scan, e.g. 2025-08-28T22:00:00Z")
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
        latest = datetime.now(UTC).replace(second=0, microsecond=0)
        latest = latest.replace(minute=latest.minute - latest.minute % 5)
    else:
        latest = datetime.fromisoformat(args.run_time.replace("Z", "+00:00"))

    rows = np.arange(grid.height, dtype=np.float32)[:, None]
    columns = np.arange(grid.width, dtype=np.float32)[None, :]
    written = []
    for index, observed in enumerate(window.times(latest)):
        centre_row, centre_column = 90 + 6 * index, 110 + 8 * index
        distance = np.sqrt((rows - centre_row) ** 2 + (columns - centre_column) ** 2)
        field = np.where(distance < 40, 14.0 * np.exp(-distance / 18.0), 0.0).astype(np.float32)
        field[:12, :] = np.nan  # outside radar coverage
        written.append(reader.synthesize(args.into / reader.filename(observed), field, grid))

    print(f"Wrote {len(written)} scans to {args.into}")
    for path in written:
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
