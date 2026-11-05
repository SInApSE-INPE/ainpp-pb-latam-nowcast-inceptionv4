"""Checkpoint-compatible InceptionV4 encoder and U-Net decoder; no pretrained downloads."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from datetime import timedelta
from pathlib import Path

import numpy as np
import timm
import torch
from torch import nn
from torch.nn import functional as F

from nowcast_runtime.ports import Grid, WindowSpec

STEP = timedelta(hours=1)
MIN_PIXELS = 128


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y, diff_x = x2.shape[-2] - x1.shape[-2], x2.shape[-1] - x1.shape[-1]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class PrecipitationForecaster(nn.Module):
    def __init__(self, input_timesteps: int, output_timesteps: int, img_size: tuple[int, int]):
        super().__init__()
        if min(input_timesteps, output_timesteps) < 1 or min(img_size) < MIN_PIXELS:
            raise ValueError("Positive sequence lengths and image dimensions >= 128 are required")
        self.input_steps = input_timesteps
        self.output_steps = output_timesteps
        self.img_size = tuple(img_size)
        self.encoder = timm.create_model(
            "inception_v4",
            pretrained=False,
            in_chans=input_timesteps,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )
        c0, c1, c2, c3, c4 = self.encoder.feature_info.channels()
        self.up1 = Up(c4, c3, c3)
        self.up2 = Up(c3, c2, c2)
        self.up3 = Up(c2, c1, c1)
        self.up4 = Up(c1, 64, c0)
        self.up_final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.outc = nn.Conv2d(32, output_timesteps, 1)

    def forward(self, x):
        if x.ndim == 5:
            if x.shape[2] != 1:
                raise ValueError("Expected one precipitation channel per timestep")
            x = x.squeeze(2)
        if x.ndim != 4 or tuple(x.shape[1:]) != (self.input_steps, *self.img_size):
            raise ValueError(f"Expected [batch, {self.input_steps}, {self.img_size}]")
        x0, x1, x2, x3, x4 = self.encoder(x)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        x = self.up_final(x)
        if x.shape[-2:] != self.img_size:
            x = F.interpolate(x, size=self.img_size, mode="bilinear", align_corners=True)
        # Preserve the training reference: ReLU is applied in normalized space.
        return torch.relu(self.outc(x)).unsqueeze(2)


class InceptionV4Predictor:
    """The runtime's Predictor port over the checkpoint architecture.

    Tensor creation and device placement live here rather than in the runtime,
    so no contract mentions torch and another framework needs no new port.
    """

    def __init__(self, module: PrecipitationForecaster, grid: Grid, device: torch.device):
        self.module = module
        self.grid = grid
        self.device = device
        self.step = STEP
        self.window = WindowSpec.uniform(STEP, module.input_steps)
        self.output_steps = module.output_steps

    def leads(self, inputs: np.ndarray, steps: int) -> Iterator[np.ndarray]:
        """One forward pass, then yield each lead as (members, variables, rows, columns)."""
        if not 1 <= steps <= self.output_steps:
            raise ValueError(f"steps must be between 1 and the horizon {self.output_steps}")
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValueError("InceptionV4 consumes one variable per timestep")
        tensor = torch.from_numpy(np.ascontiguousarray(inputs[:, 0])).unsqueeze(0)
        self.module.eval()
        with torch.inference_mode():
            prediction = self.module(tensor.to(self.device))
            expected = (1, self.output_steps, 1, *self.module.img_size)
            if tuple(prediction.shape) != expected:
                raise ValueError("Invalid InceptionV4 output shape")
            if not torch.isfinite(prediction).all():
                raise ValueError("Model produced non-finite predictions")
            fields = prediction[0].cpu().numpy()
        for lead in range(steps):
            # (variables, rows, columns) gains a single-member leading axis.
            yield fields[lead][np.newaxis]


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def load_module(path: Path, shape: tuple[int, int]) -> PrecipitationForecaster:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must contain a state dictionary")
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Empty or invalid state dictionary")
    while all(k.startswith("module.") for k in state) or all(
        k.startswith("_orig_mod.") for k in state
    ):
        state = {k.split(".", 1)[1]: v for k, v in state.items()}
    try:
        input_steps = state["encoder.features_0.conv.weight"].shape[1]
        output_steps = state["outc.weight"].shape[0]
    except (KeyError, AttributeError, IndexError) as exc:
        raise ValueError("Not a supported InceptionV4 checkpoint") from exc
    module = PrecipitationForecaster(input_steps, output_steps, shape)
    module.load_state_dict(state, strict=True)
    # Inference only: training mode would make BatchNorm use batch statistics.
    return module.eval()


def load_model(artifacts: Mapping[str, Path], device: str, grid: Grid) -> InceptionV4Predictor:
    """Build the predictor the runtime asked for, on the grid it will run."""
    resolved = resolve_device(device)
    module = load_module(artifacts["checkpoint"], grid.shape)
    return InceptionV4Predictor(module.to(resolved), grid, resolved)


def library_versions() -> dict[str, str]:
    import torchvision

    return {
        "timm": timm.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
    }
