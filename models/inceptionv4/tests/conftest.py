import gzip
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch

from inceptionv4_nowcast.model import load_module

PROJECT = Path(__file__).resolve().parents[1]
# Indices of the test patch inside the stored GSMaP array, which the decoder
# rolls by half the globe: stored column 2400 becomes grid column 600, the
# offset of west=-120 on the native grid.
ROW0, COLUMN0 = 270, 2400
SIZE = 128


@pytest.fixture(scope="session")
def module():
    torch.set_num_threads(2)
    return load_module(PROJECT / "artifacts/v1/inceptionv4.pth", (SIZE, SIZE))


@pytest.fixture
def case(tmp_path):
    payload = json.loads((PROJECT / "config/latam.json").read_text())
    payload["artifacts"] = str(PROJECT / "artifacts/v1")
    payload["domain"] = {"west": -120, "south": 20.2, "east": -107.2, "north": 33}
    payload["model_latitude_order"] = "south_to_north"
    config = tmp_path / "config.json"
    config.write_text(json.dumps(payload))
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    end = datetime(2000, 1, 2, 0, tzinfo=UTC)
    frames = []
    for index in range(12):
        stamp = end - timedelta(hours=11 - index)
        rain = np.zeros((1200, 3600), dtype="<f4")
        patch = np.arange(SIZE, dtype=np.float32)[:, None] / SIZE + index / 20
        rain[ROW0 : ROW0 + SIZE, COLUMN0 : COLUMN0 + SIZE] = patch
        rain[ROW0, COLUMN0] = -99  # The negative-to-zero rule, never NoData.
        frames.append(
            np.maximum(rain[ROW0 : ROW0 + SIZE, COLUMN0 : COLUMN0 + SIZE], 0).copy()
        )
        with gzip.open(
            inputs / f"gsmap_now.{stamp:%Y%m%d.%H00}.dat.gz", "wb", compresslevel=1
        ) as handle:
            handle.write(rain.tobytes())
    return {
        "config": config,
        "inputs": inputs,
        "outputs": tmp_path / "outputs",
        "rain": np.stack(frames),
        "run_time": end,
    }
