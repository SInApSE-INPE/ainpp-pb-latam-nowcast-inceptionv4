"""Decoders general enough that a new product is configuration, not code."""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from ..ports import Frame, Grid


@dataclass(frozen=True)
class TimestampedName:
    """A file name contract: a regex with a `stamp` group, and how to read it.

    One group holding the whole timestamp, rather than separate date and hour
    groups, is what admits a sub-hourly product: a ten-minute observation is a
    different `stamp_format`, not a different class.

    `template` generates names and `pattern` matches them, because a pattern with
    optional groups accepts more names than it can produce.

    `interval` and `end_format` together verify a provider that appends the end
    of its accumulation window, such as the GSMaP NOW product, instead of
    trusting it.
    """

    pattern: str
    stamp_format: str
    template: str | None = None
    interval: timedelta | None = None
    end_format: str | None = None

    def __post_init__(self):
        compiled = re.compile(self.pattern)
        if "stamp" not in compiled.groupindex:
            raise ValueError("Name pattern must define a stamp group")
        if (self.interval is None) != (self.end_format is None):
            raise ValueError("interval and end_format must be declared together")
        if self.end_format is not None and "end" not in compiled.groupindex:
            raise ValueError("An interval check requires an end group in the pattern")
        object.__setattr__(self, "_compiled", compiled)

    @property
    def compiled(self) -> re.Pattern:
        return self._compiled  # type: ignore[attr-defined]

    def filename(self, observed: datetime) -> str:
        """Build a name for an observation time, checked against the pattern."""
        if self.template is None:
            raise ValueError("This name contract cannot generate names; set a template")
        name = observed.strftime(self.template)
        if self.compiled.fullmatch(name) is None:
            raise ValueError(f"Template produced {name!r}, which the pattern rejects")
        return name

    def observed_at(self, name: str) -> datetime | None:
        match = self.compiled.fullmatch(name)
        if not match:
            return None
        observed = datetime.strptime(match["stamp"], self.stamp_format).replace(tzinfo=UTC)
        end = match.groupdict().get("end")
        if end is not None and self.interval is not None:
            expected = (observed + self.interval).strftime(self.end_format)
            if end != expected:
                raise ValueError(f"Unexpected interval in observation name: {name}")
        return observed


@dataclass(frozen=True)
class RawBinaryDecoder:
    """A headerless, fixed-shape binary field on a regular grid.

    Everything that varies between such products is a parameter: byte order and
    width through dtype, the grid the array covers, how far the columns must be
    rolled to reach that grid origin, the units, and the file name contract. The
    GSMaP binary format is one configuration of this class, which is the point:
    another raw product needs no new module.
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

    def filename(self, observed: datetime) -> str:
        """A name this decoder recognizes, for generating synthetic observations."""
        return self.naming.filename(observed)

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

    def synthesize(self, path: Path, values: np.ndarray, grid: Grid) -> Path:
        """Write a file this decoder reads back as values on grid.

        The exact inverse of `read`: embed the subgrid in a native-sized array,
        undo the column roll, and write the raw block. A conformance suite can
        therefore exercise a whole deployment without any real observation, for
        any model whose decoder is one of these.
        """
        if values.shape != grid.shape:
            raise ValueError("Synthetic values must match the requested grid")
        row, column = self.native_grid.locate(grid)
        field = np.zeros(self._shape, dtype=np.dtype(self.dtype))  # type: ignore[attr-defined]
        field[row : row + grid.height, column : column + grid.width] = values
        if self.roll_columns:
            field = np.roll(field, -self.roll_columns, axis=1)
        payload = field.tobytes()
        if self.gzipped:
            with gzip.open(path, "wb", compresslevel=1) as stream:
                stream.write(payload)
        else:
            path.write_bytes(payload)
        return path


GSMAP_GRID = Grid("EPSG:4326", -180.0, 60.0, 0.1, 0.1, 3600, 1200)
GSMAP_PRODUCTS = ("gsmap_now", "gsmap_nrt")


def gsmap(product: str, native_grid: Grid = GSMAP_GRID) -> RawBinaryDecoder:
    """The GSMaP hourly rain rate binary, expressed as configuration.

    JAXA stores the global field starting half a cell east of the antimeridian,
    so the columns are rolled by half the width to reach a -180 origin. Rows are
    already north first. The NOW product may append the end of its hour.
    """
    if product not in GSMAP_PRODUCTS:
        raise ValueError(f"Unknown GSMaP product: {product}")
    now = product == "gsmap_now"
    end_group = r"(?:_(?P<end>\d{4}))?" if now else ""
    return RawBinaryDecoder(
        native_grid=native_grid,
        naming=TimestampedName(
            pattern=(
                rf"{re.escape(product)}\.(?P<stamp>\d{{8}}\.\d{{4}})"
                + end_group
                + r"\.dat\.gz"
            ),
            stamp_format="%Y%m%d.%H%M",
            template=f"{product}.%Y%m%d.%H%M.dat.gz",
            interval=timedelta(hours=1) if now else None,
            end_format="%H%M" if now else None,
        ),
        units="mm/h",
        dtype="<f4",
        roll_columns=native_grid.width // 2,
    )
