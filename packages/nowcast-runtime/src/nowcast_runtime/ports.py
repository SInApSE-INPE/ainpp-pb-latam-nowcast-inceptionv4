"""Contracts between the runtime and the pieces a deployment composes.

Several shapes carry axes that are length one in every deployment today. That is
deliberate: ADR 0001 commits the contracts to admitting several input products,
auxiliary variables and ensemble members, and a unit-length axis is never
squeezed in shared code, because the squeeze is what makes the retrofit
expensive later.

Arrays cross these boundaries as NumPy, never as the tensor type of a particular
deep learning framework. Tensor handling belongs inside a model.

The spine every run follows, where bracketed stages may be empty:

    Source -> Decoder -> [Preprocess] -> Transform.forward -> Predictor
           -> Transform.inverse -> [Postprocess] -> Sink

Stage order is load bearing, so it is fixed rather than left to a composition:
configured stages run first on input and last on output, model stages run last
on input and first on output. A configured step therefore always sees physical
units as the provider published them, and a model step always sees them as its
checkpoint expects. Reversing the output restores the input ordering, so a pair
such as flipping latitude composes correctly without either side knowing about
the other.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

ZERO = timedelta(0)


@dataclass(frozen=True)
class Grid:
    """A regular, axis-aligned, north-up grid of cell areas in a single CRS.

    The origin is the outer corner of the north-west cell, matching the GeoTIFF
    convention, so a grid is fully described without a separate transform.
    """

    crs: str
    west: float
    north: float
    resolution_x: float
    resolution_y: float
    width: int
    height: int

    def __post_init__(self):
        if not self.crs:
            raise ValueError("Grid requires a CRS")
        origin = (self.west, self.north, self.resolution_x, self.resolution_y)
        if not all(math.isfinite(value) for value in origin):
            raise ValueError("Grid origin and resolution must be finite")
        if self.resolution_x <= 0 or self.resolution_y <= 0:
            raise ValueError("Grid resolution must be positive")
        if type(self.width) is not int or type(self.height) is not int:
            raise ValueError("Grid width and height must be integers")
        if self.width < 1 or self.height < 1:
            raise ValueError("Grid width and height must be positive")

    @property
    def shape(self) -> tuple[int, int]:
        """Rows then columns, matching NumPy and rasterio."""
        return (self.height, self.width)

    @property
    def east(self) -> float:
        return self.west + self.width * self.resolution_x

    @property
    def south(self) -> float:
        return self.north - self.height * self.resolution_y

    def locate(self, other: Grid) -> tuple[int, int]:
        """Row and column of another grid origin inside this one.

        Raises when the other grid is not a cell-aligned subgrid. This is the
        check every decoder owes its caller before slicing a native array.
        """
        if other.crs != self.crs:
            raise ValueError(f"Grid CRS differs: {other.crs} is not {self.crs}")
        if not (
            math.isclose(other.resolution_x, self.resolution_x)
            and math.isclose(other.resolution_y, self.resolution_y)
        ):
            raise ValueError("Grid resolution differs from the native grid")
        column = (other.west - self.west) / self.resolution_x
        row = (self.north - other.north) / self.resolution_y
        if not (math.isclose(column, round(column)) and math.isclose(row, round(row))):
            raise ValueError("Grid edges must align with the native grid cells")
        column, row = round(column), round(row)
        if row < 0 or column < 0:
            raise ValueError("Grid starts outside the native grid")
        if row + other.height > self.height or column + other.width > self.width:
            raise ValueError("Grid extends beyond the native grid")
        return row, column

    def contains(self, other: Grid) -> bool:
        try:
            self.locate(other)
        except ValueError:
            return False
        return True


@dataclass(frozen=True)
class WindowSpec:
    """Which observations one run consumes, as offsets from its run time.

    Offsets are non-positive, unique and increasing, so the run time is the last
    entry and no run depends on a future observation. Explicit offsets subsume a
    fixed step with a count, and admit a non-uniform history without changing
    this signature.
    """

    offsets: tuple[timedelta, ...]

    def __post_init__(self):
        if not self.offsets:
            raise ValueError("A window needs at least one offset")
        if any(offset > ZERO for offset in self.offsets):
            raise ValueError("Input offsets must not follow the run time")
        if list(self.offsets) != sorted(set(self.offsets)):
            raise ValueError("Input offsets must be unique and increasing")
        if self.offsets[-1] != ZERO:
            raise ValueError("The last input offset must be the run time itself")

    @classmethod
    def uniform(cls, step: timedelta, count: int) -> WindowSpec:
        """The common case: count observations spaced step apart, ending now."""
        if step <= ZERO:
            raise ValueError("Window step must be positive")
        if type(count) is not int or count < 1:
            raise ValueError("Window count must be a positive integer")
        return cls(tuple(-step * index for index in reversed(range(count))))

    def __len__(self) -> int:
        return len(self.offsets)

    def times(self, run_time: datetime) -> tuple[datetime, ...]:
        if run_time.tzinfo is None:
            raise ValueError("Run time must be timezone aware")
        return tuple(run_time + offset for offset in self.offsets)


@dataclass(frozen=True)
class Frame:
    """One decoded observation in physical units, with its missing-data mask.

    Both arrays have shape (variables, height, width). The variable axis is
    length one for a single-product deployment and is not squeezed.
    """

    values: np.ndarray
    missing: np.ndarray
    units: str
    grid: Grid
    time: datetime

    def __post_init__(self):
        if self.values.ndim != 3:
            raise ValueError("Frame values must be (variables, height, width)")
        if self.values.shape[1:] != self.grid.shape:
            raise ValueError("Frame values do not match the grid")
        if self.missing.shape != self.values.shape:
            raise ValueError("Frame mask must match its values")
        if self.missing.dtype != np.bool_:
            raise ValueError("Frame mask must be boolean")
        if not self.units:
            raise ValueError("Frame requires explicit physical units")

    @property
    def variables(self) -> int:
        return self.values.shape[0]


@dataclass(frozen=True)
class Prediction:
    """One lead, with the mask it inherited from its inputs.

    values has shape (members, variables, height, width). Both leading axes are
    length one for a deterministic, single-variable model and are not squeezed.
    Carrying the mask here keeps a Prediction self-contained, so a postprocess
    step and a sink both receive everything they need from one object.
    """

    values: np.ndarray
    missing: np.ndarray
    lead: int
    valid_time: datetime

    def __post_init__(self):
        if self.values.ndim != 4:
            raise ValueError("Prediction values must be (members, variables, height, width)")
        if self.missing.shape != self.values.shape:
            raise ValueError("Prediction mask must match its values")
        if self.missing.dtype != np.bool_:
            raise ValueError("Prediction mask must be boolean")
        if self.lead < 1:
            raise ValueError("Lead index starts at one")

    @property
    def members(self) -> int:
        return self.values.shape[0]


@dataclass(frozen=True)
class Observation:
    """One remote or local object that a decoder can turn into a Frame."""

    time: datetime
    name: str
    location: str


@runtime_checkable
class Source(Protocol):
    """Where observation objects are listed, and how they reach local storage.

    A source owns path syntax: it supplies each candidate bare name, so shared
    code never has to guess whether a location is POSIX or native.
    """

    @property
    def identity(self) -> str:
        """Stable description for the run manifest. Must not carry credentials."""

    def candidates(
        self, window: WindowSpec, run_time: datetime | None
    ) -> Sequence[tuple[str, str]]:
        """Name and location of every object that might belong to the window."""

    def fetch(self, observation: Observation) -> Path:
        """Return a local, fully transferred and structurally valid file."""

    def prune(self, ttl: timedelta) -> int:
        """Delete expired managed downloads. Supplied inputs are never removed."""


@runtime_checkable
class Decoder(Protocol):
    """How a named object becomes a Frame on the requested grid.

    The decoder owns the file name contract and the native grid, which is why a
    new input product requires no change to acquisition or scheduling.
    """

    @property
    def native_grid(self) -> Grid:
        """The full grid the provider publishes, which requests must fit inside."""

    @property
    def units(self) -> str:
        """Physical units of the decoded values."""

    def observed_at(self, name: str) -> datetime | None:
        """The observation time this name encodes, or None when it is not ours."""

    def read(self, path: Path, grid: Grid) -> Frame:
        """Decode and crop, raising on truncation or a misaligned grid."""


@runtime_checkable
class Preprocess(Protocol):
    """One step applied to a decoded Frame, in physical units.

    Runs before the transform. A deployment may declare none, which is the
    default: the absence of preprocessing is an ordinary configuration, not a
    special case.
    """

    def apply(self, frame: Frame) -> Frame:
        """Return a Frame on the same grid, at the same time."""

    def describe(self) -> Mapping[str, object]:
        """Parameters to record in the run manifest."""


@runtime_checkable
class Postprocess(Protocol):
    """One step applied to a Prediction, in physical units.

    Runs after the inverse transform, so a step sees real units rather than
    model space. A deployment may declare none.
    """

    def apply(self, prediction: Prediction) -> Prediction:
        """Return a Prediction for the same lead and valid time."""

    def describe(self) -> Mapping[str, object]:
        """Parameters to record in the run manifest."""


@runtime_checkable
class Transform(Protocol):
    """A model input scaling, inseparable from its checkpoint.

    Optional: a model that consumes physical units declares no transform and the
    runtime applies identity. When one exists, the plugin binds it and
    configuration cannot override it, because the statistics live in the artifact
    bundle and a mismatch would raise nothing while producing plausible but
    meaningless output (ADR 0001, ADR 0004).
    """

    def forward(self, values: np.ndarray) -> np.ndarray:
        """Physical units to model space."""

    def inverse(self, values: np.ndarray) -> np.ndarray:
        """Model space back to physical units."""

    def describe(self) -> Mapping[str, object]:
        """Parameters to record in the run manifest."""


@runtime_checkable
class Predictor(Protocol):
    """A loaded model, ready to infer on one fixed grid."""

    @property
    def window(self) -> WindowSpec:
        """The history this model consumes, including its spacing."""

    @property
    def step(self) -> timedelta:
        """Spacing between consecutive output leads."""

    @property
    def output_steps(self) -> int:
        """Native horizon in leads. A request may be shorter, never longer."""

    @property
    def grid(self) -> Grid:
        """The grid this instance was loaded for."""

    def leads(self, inputs: np.ndarray, steps: int, run_time: datetime) -> Iterator[Prediction]:
        """Yield leads in chronological order.

        An iterator, so a direct model can run a single forward pass while a
        recurrent or autoregressive model advances its own state per lead.
        """


@runtime_checkable
class Sink(Protocol):
    """How one lead is named and written."""

    @property
    def extension(self) -> str:
        """File extension, without the leading dot."""

    def filename(self, slug: str, run_time: datetime, prediction: Prediction, member: int) -> str:
        """Name one output. A deterministic run omits any member component."""

    def write(
        self,
        path: Path,
        prediction: Prediction,
        member: int,
        grid: Grid,
        units: str,
        tags: Mapping[str, object],
    ) -> None:
        """Write one member of one lead into a staging directory."""


@dataclass(frozen=True)
class ModelPlugin:
    """Everything the runtime needs from a model, and nothing more.

    Only the first six fields are required. A model that consumes physical units
    and needs no model-side stages declares nothing further, because the absence
    of pre and post processing is the default path rather than a special case
    (ADR 0004).

    bundle_roles is declared rather than fixed, so a model with no normalization
    statistics, or one with auxiliary static fields, is expressible without
    changing the manifest reader.

    The optional factories receive the verified artifact paths, so a model-side
    stage may be parameterized by its own bundle.
    """

    name: str
    slug: str
    version: str
    bundle_roles: tuple[str, ...]
    requires_runtime: str
    load: Callable[[Mapping[str, Path], str, Grid], Predictor]
    transform: Callable[[Mapping[str, Path]], Transform] | None = None
    preprocess: Callable[[Mapping[str, Path]], Sequence[Preprocess]] | None = None
    postprocess: Callable[[Mapping[str, Path]], Sequence[Postprocess]] | None = None
    library_versions: Callable[[], Mapping[str, str]] = dict

    def __post_init__(self):
        if not self.name or not self.version or not self.requires_runtime:
            raise ValueError("A plugin must declare a name, version and runtime range")
        if not self.slug or not self.slug.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Plugin slug must be a path and filename safe token")
        if not self.bundle_roles or len(set(self.bundle_roles)) != len(self.bundle_roles):
            raise ValueError("Bundle roles must be non-empty and unique")
