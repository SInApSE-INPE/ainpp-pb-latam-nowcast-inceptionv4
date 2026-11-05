"""InceptionV4 through the shared conformance suite.

This runs the real composition root, not a stand-in, so what passes here is the
same wiring the container uses. The domain is reduced to the architecture's
128-pixel minimum: conformance checks the contracts, not throughput.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import torch

from inceptionv4_nowcast.entrypoint import compose, parser
from nowcast_runtime.testing import ConformanceSuite

PROJECT = Path(__file__).resolve().parents[1]
SMALL_DOMAIN = {"west": -120.0, "south": 20.2, "east": -107.2, "north": 33.0}


class TestInceptionV4(ConformanceSuite):
    product = "gsmap_now"
    run_time = datetime(2000, 1, 2, 0, 0, tzinfo=UTC)

    def build(self, tmp_path):
        torch.set_num_threads(2)
        payload = json.loads((PROJECT / "config/latam.json").read_text(encoding="utf-8"))
        payload["artifacts"] = str(PROJECT / "artifacts/v1")
        payload["domain"] = SMALL_DOMAIN
        config = tmp_path / "config.json"
        config.write_text(json.dumps(payload), encoding="utf-8")
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        args = parser().parse_args(
            [
                "--config",
                str(config),
                "--source",
                "local",
                "--input-dir",
                str(inputs),
                "--device",
                "cpu",
                "--threads",
                "2",
            ]
        )
        deployment, product = compose(args)
        assert product == self.product
        return deployment

    def field(self, index, shape):
        """Plausible rain rates, including the provider's negative sentinel.

        The sentinel exercises ClipNegative on a real deployment rather than
        only in a unit test, and it must not become NoData.
        """
        values = super().field(index, shape)
        values[0, 0] = -99.0
        return values
