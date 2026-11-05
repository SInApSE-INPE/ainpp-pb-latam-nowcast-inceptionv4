"""IRENE through the shared conformance suite.

This runs the real composition root, so what passes here is the wiring an
operational deployment would use. The suite synthesizes its own radar scans
through the deployment's decoder, so it needs no DPC data.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from irene_nowcast.entrypoint import compose, parser

from nowcast_runtime.testing import ConformanceSuite

PROJECT = Path(__file__).resolve().parents[1]


class TestIRENE(ConformanceSuite):
    product = "dpc_sri"
    run_time = datetime(2025, 8, 28, 22, 0, tzinfo=UTC)

    def build(self, tmp_path):
        payload = json.loads((PROJECT / "config/default.json").read_text(encoding="utf-8"))
        payload["artifacts"] = str(PROJECT / "artifacts/v1")
        config = tmp_path / "config.json"
        config.write_text(json.dumps(payload), encoding="utf-8")
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        args = parser().parse_args(
            [
                "--config", str(config),
                "--source", "local",
                "--input-dir", str(inputs),
                "--device", "cpu",
            ]
        )
        deployment, product = compose(args)
        assert product == self.product
        return deployment

    def field(self, index, shape):
        """Rain rates that look like radar: mostly dry, with a moving cell.

        Values are kept inside what the reflectivity scale can represent, apart
        from the dry background, which is the physically correct zero.
        """
        rows = np.arange(shape[0], dtype=np.float32)[:, None]
        columns = np.arange(shape[1], dtype=np.float32)[None, :]
        centre_row, centre_column = 90 + 6 * index, 110 + 8 * index
        distance = np.sqrt((rows - centre_row) ** 2 + (columns - centre_column) ** 2)
        field = np.where(distance < 40, 14.0 * np.exp(-distance / 18.0), 0.0)
        # Radar coverage does not fill the rectangle; a real scan is about half
        # missing, and the mask must survive the whole pipeline.
        field[:12, :] = np.nan
        return field.astype(np.float32)
