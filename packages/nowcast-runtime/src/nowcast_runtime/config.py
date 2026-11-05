"""Validated configuration; all paths are relative to the configuration file.

The schema is still close to one deployment's shape rather than fully
declarative: selecting a source or decoder by name belongs with the registry,
and until then a model's composition root reads these fields. What has gone is
the hardcoded product whitelist, because a decoder now validates its own
product.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .builtin.sources import RetryPolicy
from .builtin.stages import POLICIES
from .ports import Grid

LATITUDE_ORDERS = ("north_to_south", "south_to_north")


@dataclass(frozen=True)
class Bounds:
    """A requested extent, in the CRS of whichever decoder supplies the data.

    Bounds alone cannot make a Grid: resolution and CRS belong to the provider.
    `to_grid` is therefore the negotiation between what an operator asked for
    and what a decoder publishes, and it fails loudly when they disagree.
    """

    west: float
    south: float
    east: float
    north: float

    def __post_init__(self):
        if not all(math.isfinite(value) for value in vars(self).values()):
            raise ValueError("Domain values must be finite")
        if not (self.west < self.east and self.south < self.north):
            raise ValueError("Domain must have positive extent")

    def to_grid(self, native: Grid) -> Grid:
        width = (self.east - self.west) / native.resolution_x
        height = (self.north - self.south) / native.resolution_y
        if not (math.isclose(width, round(width)) and math.isclose(height, round(height))):
            raise ValueError("Domain extent must be a whole number of native cells")
        grid = Grid(
            native.crs,
            self.west,
            self.north,
            native.resolution_x,
            native.resolution_y,
            round(width),
            round(height),
        )
        # Raises unless the request is a cell-aligned subgrid of the native grid.
        native.locate(grid)
        return grid


@dataclass(frozen=True)
class Config:
    artifacts: Path
    domain: Bounds
    model_latitude_order: str
    product: str
    remote_directory: str
    missing_policy: str
    max_missing_fraction: float
    ftp: RetryPolicy

    def __post_init__(self):
        if self.model_latitude_order not in LATITUDE_ORDERS:
            raise ValueError(f"model_latitude_order must be one of {LATITUDE_ORDERS}")
        if self.missing_policy not in POLICIES:
            raise ValueError(f"missing_policy must be one of {POLICIES}")
        if not 0 <= self.max_missing_fraction < 1:
            raise ValueError("max_missing_fraction must be in [0, 1)")
        if not self.product or not self.remote_directory:
            raise ValueError("A product and a remote directory are required")

    @classmethod
    def load(cls, path: Path) -> Config:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["artifacts"] = (path.resolve().parent / payload["artifacts"]).resolve()
        payload["domain"] = Bounds(**payload["domain"])
        payload["ftp"] = RetryPolicy(**payload["ftp"])
        return cls(**payload)
