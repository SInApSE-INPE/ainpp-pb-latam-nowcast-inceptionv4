"""Decoders general enough that a new product is configuration, not code."""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ..ports import Frame, Grid


@dataclass(frozen=True)
class TimestampedName:
    """A file name contract: a regex naming its own date and hour groups.

    Optional groups make this cover a provider that appends an interval, such as
    the GSMaP NOW product. When interval_hours is set and the pattern matched an
    end group, the interval is verified rather than trusted.
    """

    pattern: str
    interval_hours: int | None = None
    date_format: str = "%Y%m%d%H"

    def __post_init__(self):
        compiled = re.compile(self.pattern)
        required = {"date", "hour"}
        if not required <= set(compiled.groupindex):
            raise ValueError("Name pattern must define date and hour groups")
        if self.interval_hours is not None and "end" not in compiled.groupindex:
            raise ValueError("interval_hours requires an end group in the pattern")
        object.__setattr__(self, "_compiled", compiled)

    @property
    def compiled(self) -> re.Pattern:
        return self._compiled  # type: ignore[attr-defined]

    def observed_at(self, name: str) -> datetime | None:
        match = self.compiled.fullmatch(name)
        if not match:
            return None
        observed = datetime.strptime(
            match["date"] + match["hour"], self.date_format
        ).replace(tzinfo=UTC)
        end = match.groupdict().get("end")
        if self.interval_hours is not None and end is not None:
            if int(end) != (observed.hour + self.interval_hours) % 24:
                raise ValueError(f"Unexpected interval in observation name: {name}")
        return observed


@dataclass(frozen=True)
class RawBinaryDecoder:
    """A headerless, fixed-shape binary field on a regular global grid.

    Everything that varies between such products is a parameter: byte order and
    width through dtype, the array shape, the grid the array covers, how far the
    columns must be rolled to reach that grid origin, the units, and the file
    name contract. The GSMaP binary format is one configuration of this class,
    which is the point: another raw product needs no new module.
    """

    native_grid: Grid
    naming: TimestampedName
    units: str
    dtype: str = "<f4"
    roll_columns: int = 0
    gzipped: bool = True
    suffix: str = ".dat.gz"

    def __post_init__(self):
        if not self.units:
            raise ValueError("A decoder must declare physical units")
        shape = (self.native_grid.height, self.native_grid.width)
        itemsize = np.dtype(self.dtype).itemsize
        object.__setattr__(self, "_shape", shape)
        object.__setattr__(self, "_expected_bytes", shape[0] * shape[1] * itemsize)

    @property
    def glob(self) -> str:
        """Pattern a local source may use to narrow its listing cheaply."""
        return "*" + self.suffix

    def observed_at(self, name: str) -> datetime | None:
        return self.naming.observed_at(name)

    def read(self, path: Path, grid: Grid) -> Frame:
        """Decode and crop, raising on truncation or a misaligned request.

        Values are returned exactly as stored, in physical units. Non-finite
        cells are reported through the mask; a provider sentinel such as a
        negative fill is a preprocess concern, not this decoder's, because it is
        coupled to the source rather than to the format.
        """
        observed = self.observed_at(path.name)
        if observed is None:
            raise ValueError(f"Not an observation this decoder recognizes: {path.name}")
        row, column = self.native_grid.locate(grid)
        expected = self._expected_bytes  # type: ignore[attr-defined]
        opener = gzip.open if self.gzipped else open
        # Reading one byte past the expected size also detects truncation, and
        # for a gzip member it forces the CRC and length check to run.
        with opener(path, "rb") as stream:
            raw = stream.read(expected + 1)
        if len(raw) != expected:
            raise ValueError(f"Invalid field size in {path.name}: {len(raw)} bytes")
        data = np.frombuffer(raw, dtype=np.dtype(self.dtype))
        data = data.reshape(self._shape)  # type: ignore[attr-defined]
        if self.roll_columns:
            data = np.roll(data, self.roll_columns, axis=1)
        values = data[row : row + grid.height, column : column + grid.width].astype(np.float32)
        missing = ~np.isfinite(values)
        return Frame(values[np.newaxis], missing[np.newaxis], self.units, grid, observed)


def gsmap(product: str, native_grid: Grid) -> RawBinaryDecoder:
    """The GSMaP hourly rain rate binary, expressed as configuration.

    JAXA stores the global field starting half a cell east of the antimeridian,
    so the columns are rolled by half the width to reach a -180 origin. Rows are
    already north first. The NOW product may append a one-hour interval.
    """
    if product not in {"gsmap_now", "gsmap_nrt"}:
        raise ValueError(f"Unknown GSMaP product: {product}")
    interval = r"(?:_(?P<end>\d{2})00)?" if product == "gsmap_now" else ""
    return RawBinaryDecoder(
        native_grid=native_grid,
        naming=TimestampedName(
            pattern=rf"{re.escape(product)}\.(?P<date>\d{{8}})\.(?P<hour>\d{{2}})00"
            + interval
            + r"\.dat\.gz",
            interval_hours=1 if product == "gsmap_now" else None,
        ),
        units="mm/h",
        dtype="<f4",
        roll_columns=native_grid.width // 2,
    )


GSMAP_GRID = Grid("EPSG:4326", -180.0, 60.0, 0.1, 0.1, 3600, 1200)
