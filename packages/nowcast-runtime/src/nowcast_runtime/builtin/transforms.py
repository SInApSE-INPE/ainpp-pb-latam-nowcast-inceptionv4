"""Built-in transforms, including the identity a model without one receives."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Identity:
    """What the runtime applies when a plugin declares no transform.

    A model that consumes physical units therefore needs no transform code at
    all, which is the point: the absence of preprocessing is the default path.
    """

    def forward(self, values: np.ndarray) -> np.ndarray:
        return values

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values

    def neutral(self) -> float:
        """The physical value a model reads as neutral: zero, for identity."""
        return 0.0

    def describe(self) -> Mapping[str, object]:
        return {"kind": "identity"}


@dataclass(frozen=True)
class LogZScore:
    """log1p followed by a scalar z-score, and its exact inverse.

    The statistics belong to one checkpoint, so a plugin binds this and
    configuration cannot replace it.
    """

    mean: float
    std: float

    def __post_init__(self):
        if not np.isfinite(self.mean) or not np.isfinite(self.std) or self.std <= 0:
            raise ValueError("LogZScore requires a finite mean and a positive finite std")

    @classmethod
    def from_arrays(cls, mean: Path, std: Path) -> LogZScore:
        return cls(
            float(np.load(mean, allow_pickle=False).reshape(())),
            float(np.load(std, allow_pickle=False).reshape(())),
        )

    def forward(self, values: np.ndarray) -> np.ndarray:
        normalized = (np.log1p(values.astype(np.float32)) - self.mean) / self.std
        if not np.isfinite(normalized).all():
            raise ValueError("Non-finite values after normalization")
        return normalized.astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            physical = np.expm1(values.astype(np.float32) * self.std + self.mean)
        if not np.isfinite(physical).all():
            raise ValueError("Non-finite values after inverse normalization")
        return np.maximum(physical, 0).astype(np.float32)

    def neutral(self) -> float:
        """The physical value that maps to zero in model space.

        log1p(expm1(mean)) is mean, so a cell filled with this reads as the
        training mean. It is not bit-identical to writing zero into model space;
        the difference is at float32 resolution and applies only to cells the
        mask already marks as missing, whose outputs are published as NoData.
        """
        return float(np.expm1(np.float32(self.mean)))

    def describe(self) -> Mapping[str, object]:
        return {"kind": "log-zscore", "mean": self.mean, "std": self.std}
