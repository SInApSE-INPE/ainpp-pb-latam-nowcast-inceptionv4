"""The DPC radar NetCDF format, isolated so its quirks never reach shared code.

A decoder of this shape is not a built-in because the format is not a raw block:
it is NetCDF4, which is HDF5 underneath, so h5py reads it without the netCDF C
library. Everything else about it follows the ordinary `Decoder` contract.

Two things about the published sample are worth stating rather than silently
working around, because both are the kind of mismatch that produces plausible
output and no error:

- The ``RR`` variable declares ``units = kg m-2 s-1``. Its values do not agree:
  a maximum of 61.6 with a mean of 0.61 and 84% zeros is a rain rate in mm/h,
  while the same figure in kg m-2 s-1 would be 220,000 mm/h. The model card
  says mm/h. This decoder reports mm/h and ignores the attribute.
- The variable carries ``grid_mapping = "crs"``, and the sample contains no
  such variable. The coordinate reference system is therefore **not
  determinable from the data** and must be asserted by configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from nowcast_runtime.builtin.decoders import TimestampedName
from nowcast_runtime.ports import Frame, Grid


@dataclass(frozen=True)
class RadarNetcdf:
    """One rain-rate field per file, on a projected, north-up metre grid."""

    native_grid: Grid
    naming: TimestampedName
    variable: str = "RR"
    units: str = "mm/h"
    suffix: str = ".nc"

    def __post_init__(self):
        if not self.units:
            raise ValueError("A decoder must declare physical units")

    @property
    def glob(self) -> str:
        return "*" + self.suffix

    def observed_at(self, name: str) -> datetime | None:
        return self.naming.observed_at(name)

    def filename(self, observed: datetime) -> str:
        return self.naming.filename(observed)

    def read(self, path: Path, grid: Grid) -> Frame:
        observed = self.observed_at(path.name)
        if observed is None:
            raise ValueError(f"Not an observation this decoder recognizes: {path.name}")
        row, column = self.native_grid.locate(grid)
        with h5py.File(path, "r") as handle:
            if self.variable not in handle:
                raise ValueError(f"{path.name} has no {self.variable} variable")
            field = handle[self.variable]
            if field.ndim != 2:
                raise ValueError(
                    f"{path.name} holds a {field.ndim}-dimensional {self.variable}; "
                    "this decoder reads one observation per file"
                )
            if field.shape != (self.native_grid.height, self.native_grid.width):
                raise ValueError(
                    f"{path.name} is {field.shape}, not the declared native grid "
                    f"{(self.native_grid.height, self.native_grid.width)}"
                )
            block = field[row : row + grid.height, column : column + grid.width]
        values = np.asarray(block, dtype=np.float32)
        # Radar coverage does not fill the rectangle, so a large NaN fraction is
        # ordinary here rather than a fault. The mask carries it onward.
        missing = ~np.isfinite(values)
        return Frame(values[np.newaxis], missing[np.newaxis], self.units, grid, observed)

    def synthesize(self, path: Path, values: np.ndarray, grid: Grid) -> Path:
        """Write a file this decoder reads back as values on grid.

        The inverse of `read`, so a conformance suite can exercise the whole
        deployment without any real radar data.
        """
        if values.shape != grid.shape:
            raise ValueError("Synthetic values must match the requested grid")
        row, column = self.native_grid.locate(grid)
        field = np.full(
            (self.native_grid.height, self.native_grid.width), np.nan, dtype=np.float32
        )
        field[row : row + grid.height, column : column + grid.width] = values
        with h5py.File(path, "w") as handle:
            dataset = handle.create_dataset(self.variable, data=field, dtype="float32")
            dataset.attrs["units"] = self.units
            dataset.attrs["long_name"] = "Total precipitation rate"
        return path
