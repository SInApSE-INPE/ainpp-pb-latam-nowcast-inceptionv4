"""Built-in output formats."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from ..ports import Grid, Prediction

NODATA = -9999.0
STAMP = "%Y%m%dT%H%MZ"


@dataclass(frozen=True)
class GeoTiffSink:
    """North-up, pixel-area GeoTIFF with one band per variable.

    A deterministic run omits any member component from the file name, so the
    single-member case reads exactly as it did before members were admitted.
    """

    nodata: float = NODATA
    compress: str = "deflate"
    predictor: int = 3
    blocksize: int = 256
    variable_names: Sequence[str] = field(default_factory=lambda: ("precipitation_rate",))

    @property
    def extension(self) -> str:
        return "tif"

    def filename(self, slug: str, run_time: datetime, prediction: Prediction, member: int) -> str:
        part = "" if prediction.members == 1 else f"_m{member:03d}"
        return (
            f"{slug}_{run_time:{STAMP}}_f{prediction.lead:03d}"
            f"{part}_{prediction.valid_time:{STAMP}}.{self.extension}"
        )

    def write(
        self,
        path: Path,
        prediction: Prediction,
        member: int,
        grid: Grid,
        units: str,
        tags: Mapping[str, object],
    ) -> None:
        values = prediction.values[member]
        missing = prediction.missing[member]
        if values.shape[1:] != grid.shape:
            raise ValueError("Output dimensions differ from the configured grid")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Output values must be finite and nonnegative")
        data = np.where(missing, self.nodata, values).astype(np.float32)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=grid.height,
            width=grid.width,
            count=values.shape[0],
            dtype="float32",
            crs=grid.crs,
            transform=from_origin(grid.west, grid.north, grid.resolution_x, grid.resolution_y),
            nodata=self.nodata,
            tiled=True,
            blockxsize=self.blocksize,
            blockysize=self.blocksize,
            compress=self.compress,
            predictor=self.predictor,
            BIGTIFF="IF_SAFER",
        ) as dataset:
            dataset.write(data)
            for index in range(values.shape[0]):
                name = (
                    self.variable_names[index]
                    if index < len(self.variable_names)
                    else f"variable_{index}"
                )
                dataset.set_band_description(index + 1, name)
                dataset.set_band_unit(index + 1, units)
            dataset.update_tags(AREA_OR_POINT="Area", units=units, **tags)
