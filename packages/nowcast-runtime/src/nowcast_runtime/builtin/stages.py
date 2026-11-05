"""Built-in processing steps, each one previously a branch in shared code."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np

from ..ports import Frame

POLICIES = ("error", "zero_rain", "normalized_zero")


@dataclass(frozen=True)
class ClipNegative:
    """Clamp finite negatives to a floor, leaving non-finite cells untouched.

    A provider sentinel such as -99 is a source concern, not a format one, so it
    is clipped here rather than inside a decoder. Non-finite cells stay
    non-finite and remain the mask's business.
    """

    floor: float = 0.0

    def apply(self, frame: Frame) -> Frame:
        values = np.maximum(frame.values, self.floor)
        return replace(frame, values=values.astype(np.float32, copy=False))

    def describe(self) -> Mapping[str, object]:
        return {"kind": "clip-negative", "floor": self.floor}


@dataclass(frozen=True)
class MissingPolicy:
    """Bound how much may be missing, and decide what fills the gaps.

    The fill is an explicit physical value, so this step is independent of any
    model transform. A deployment that wants the value a model reads as neutral
    passes it in; the composition root resolves that from the transform, because
    only the plugin knows what neutral means for its checkpoint.
    """

    policy: str
    max_fraction: float
    fill: float = 0.0

    def __post_init__(self):
        if self.policy not in POLICIES:
            raise ValueError(f"Unknown missing policy: {self.policy}")
        if not 0 <= self.max_fraction < 1:
            raise ValueError("max_fraction must be in [0, 1)")
        if not np.isfinite(self.fill):
            raise ValueError("Missing-data fill must be finite")

    def apply(self, frame: Frame) -> Frame:
        fraction = float(frame.missing.mean())
        if fraction > self.max_fraction:
            raise ValueError(
                f"Missing fraction {fraction:.3%} exceeds the limit at "
                f"{frame.time.isoformat()}"
            )
        if fraction and self.policy == "error":
            raise ValueError(
                f"Missing values at {frame.time.isoformat()}; choose an explicit policy"
            )
        values = frame.values.copy()
        values[frame.missing] = self.fill
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values survived the fill at {frame.time.isoformat()}")
        return replace(frame, values=values)

    def describe(self) -> Mapping[str, object]:
        return {
            "kind": "missing-policy",
            "policy": self.policy,
            "max_fraction": self.max_fraction,
            "fill": self.fill,
        }


@dataclass(frozen=True)
class FlipLatitude:
    """Reverse row order, for a model trained south to north.

    The same step serves as a preprocess and a postprocess, so a deployment
    declares it once at each end and the output is restored to the orientation
    the grid describes.
    """

    def apply(self, payload):
        return replace(
            payload,
            values=np.ascontiguousarray(payload.values[..., ::-1, :]),
            missing=np.ascontiguousarray(payload.missing[..., ::-1, :]),
        )

    def describe(self) -> Mapping[str, object]:
        return {"kind": "flip-latitude"}
