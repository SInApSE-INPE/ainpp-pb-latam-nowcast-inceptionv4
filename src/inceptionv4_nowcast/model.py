"""Checkpoint-compatible InceptionV4 encoder and U-Net decoder; no pretrained downloads."""

from collections.abc import Mapping
from pathlib import Path

import timm
import torch
from torch import nn
from torch.nn import functional as F


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
        if min(input_timesteps, output_timesteps) < 1 or min(img_size) < 128:
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


def load_model(path: Path, device: torch.device, expected_shape: tuple[int, int]):
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
    model = PrecipitationForecaster(input_steps, output_steps, expected_shape)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def forecast_steps(model: PrecipitationForecaster, inputs: torch.Tensor, steps: int):
    """One forward pass, then yield individual normalized leads in chronological order."""
    if not 1 <= steps <= model.output_steps:
        raise ValueError(f"steps must be between 1 and native horizon {model.output_steps}")
    model.eval()
    with torch.inference_mode():
        prediction = model(inputs)
        if prediction.shape != (inputs.shape[0], model.output_steps, 1, *model.img_size):
            raise ValueError("Invalid InceptionV4 output shape")
        if not torch.isfinite(prediction).all():
            raise ValueError("Model produced non-finite predictions")
    for lead in range(steps):
        yield prediction[:, lead]
