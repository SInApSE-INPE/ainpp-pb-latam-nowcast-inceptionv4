"""IRENE: a ConvGRU ensemble nowcaster, wired to the runtime contracts.

The architecture and the reflectivity transform are vendored verbatim from
`it4lia/irene` under `_vendor/`, BSD 2-Clause, Fondazione Bruno Kessler. Only
the torch-only `EncoderDecoder` is used, which avoids PyTorch Lightning and the
twenty-odd training and notebook dependencies the published package declares.

The published `RadarLightningModel.predict` does the reflectivity conversion
internally. Here that conversion is declared as the plugin's `Transform`
instead, so the runtime applies it on the way in and its inverse on the way
out, and it appears in every run manifest.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch

from nowcast_runtime.ports import Grid, WindowSpec

from ._vendor.model import EncoderDecoder
from ._vendor.utils import normalized_to_rainrate, rainrate_to_normalized

STEP = timedelta(minutes=5)
INPUT_STEPS = 6
OUTPUT_STEPS = 12
INPUT_CHANNELS = 1
NUM_BLOCKS = 5
# PixelUnshuffle halves the grid once per block, so both dimensions must be a
# multiple of 2**num_blocks. The published predict() pads to this silently; we
# do the same and crop back, rather than refusing an honest grid.
DIVISOR = 2**NUM_BLOCKS
# Both ends of the reflectivity scale saturate, so these are derived from the
# transform rather than written down. A rounded literal was off by less than a
# float32 ulp and the floor threshold silently missed every pixel, leaving a
# uniform drizzle across the domain that looked like a forecast.
OUTPUT_FLOOR = float(normalized_to_rainrate(np.array([-1.0], dtype=np.float32))[0])
OUTPUT_CEILING = float(normalized_to_rainrate(np.array([1.0], dtype=np.float32))[0])


class ReflectivityScale:
    """Marshall-Palmer dBZ scaling, the transform this checkpoint was trained on.

    Rain rate to reflectivity with ``Z = 200 R**1.6``, clipped to 0-60 dBZ, then
    mapped onto [-1, 1]. It is deliberately **lossy**: everything at or below
    0.0365 mm/h collapses to the same value, so the inverse is a left inverse
    only above that floor. `representable_minimum` states that, rather than
    leaving a caller to discover it from surprising output.
    """

    representable_minimum = OUTPUT_FLOOR
    representable_maximum = OUTPUT_CEILING

    def forward(self, values: np.ndarray) -> np.ndarray:
        if np.isnan(values).any():
            raise ValueError("Reflectivity scaling needs gaps filled before it runs")
        scaled = rainrate_to_normalized(np.maximum(values, 0.0).astype(np.float32))
        return np.asarray(scaled, dtype=np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        # The published inference path omits the clamp used in training, so an
        # out-of-range member would grow exponentially: -1.2 becomes 486 mm/h.
        bounded = np.clip(values.astype(np.float32), -1.0, 1.0)
        return np.asarray(normalized_to_rainrate(bounded), dtype=np.float32)

    def neutral(self) -> float:
        """The physical rain rate that maps to zero in model space, about 2.7 mm/h.

        For this model that is moderate rain, not an absence of it, which is why
        a deployment should fill gaps with `zero_rain` rather than
        `normalized_zero`.
        """
        return float(normalized_to_rainrate(np.zeros(1, dtype=np.float32))[0])

    def describe(self) -> Mapping[str, object]:
        return {
            "kind": "marshall-palmer-dbz",
            "relation": "Z = 200 * R**1.6",
            "decibel_range": [0, 60],
            "representable_mm_h": [OUTPUT_FLOOR, OUTPUT_CEILING],
            "clamped_before_inverse": True,
        }


class IrenePredictor:
    """The runtime's Predictor port over the vendored ConvGRU encoder-decoder."""

    def __init__(
        self,
        module: EncoderDecoder,
        grid: Grid,
        device: torch.device,
        members: int,
        seed: int | None,
    ):
        if members < 1:
            raise ValueError("At least one ensemble member is required")
        self.module = module
        self.grid = grid
        self.device = device
        self.members = members
        self.seed = seed
        self.step = STEP
        self.window = WindowSpec.uniform(STEP, INPUT_STEPS)
        self.output_steps = OUTPUT_STEPS
        self.output_variables = 1

    def leads(self, inputs: np.ndarray, steps: int) -> Iterator[np.ndarray]:
        """One encode, then unroll; yields (members, variables, rows, columns)."""
        if not 1 <= steps <= self.output_steps:
            raise ValueError(f"steps must be between 1 and the horizon {self.output_steps}")
        if inputs.ndim != 4 or inputs.shape[1] != INPUT_CHANNELS:
            raise ValueError("IRENE consumes one variable per timestep")
        height, width = inputs.shape[-2:]
        pad_rows = (DIVISOR - height % DIVISOR) % DIVISOR
        pad_columns = (DIVISOR - width % DIVISOR) % DIVISOR
        padded = np.pad(
            inputs, ((0, 0), (0, 0), (0, pad_rows), (0, pad_columns)), constant_values=-1.0
        )

        # Ensemble members come from fresh noise per decoder pass, so a run is
        # only reproducible if the generator is seeded. The seed is recorded.
        if self.seed is not None:
            torch.manual_seed(self.seed)
        tensor = torch.from_numpy(np.ascontiguousarray(padded)).unsqueeze(0)
        self.module.eval()
        with torch.inference_mode():
            output = self.module(
                tensor.to(self.device),
                steps,
                noisy_decoder=True,
                ensemble_size=self.members,
            )
            expected = (1, steps, self.members, *padded.shape[-2:])
            if tuple(output.shape) != expected:
                raise ValueError(f"IRENE returned {tuple(output.shape)}, expected {expected}")
            if not torch.isfinite(output).all():
                raise ValueError("Model produced non-finite predictions")
            fields = output[0].cpu().numpy()
        for lead in range(steps):
            # (members, rows, columns), cropped back, gains a variable axis.
            member_fields = fields[lead][:, :height, :width]
            yield member_fields[:, np.newaxis].astype(np.float32)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def load_model(
    artifacts: Mapping[str, Path],
    device: str,
    grid: Grid,
    members: int = 10,
    seed: int | None = None,
) -> IrenePredictor:
    """Build the predictor, reading tensors only.

    The bundle holds weights extracted by ``tools/prepare_artifacts.py``, so
    this never unpickles anything but tensors. The published 770 MB Lightning
    checkpoint cannot be read that way, which is the reason the extraction step
    exists.
    """
    resolved = resolve_device(device)
    weights = torch.load(artifacts["weights"], map_location="cpu", weights_only=True)
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError("Bundle weights must be a non-empty state dictionary")
    module = EncoderDecoder(INPUT_CHANNELS, NUM_BLOCKS)
    module.load_state_dict(weights, strict=True)
    # Inference only: training mode would make any normalization layer use batch
    # statistics instead of the learned ones.
    return IrenePredictor(module.eval().to(resolved), grid, resolved, members, seed)


def transform(artifacts: Mapping[str, Path]) -> ReflectivityScale:
    """The scaling is fixed by the architecture, so the bundle is not consulted."""
    return ReflectivityScale()


def library_versions() -> dict[str, str]:
    return {"torch": torch.__version__, "numpy": np.__version__}
