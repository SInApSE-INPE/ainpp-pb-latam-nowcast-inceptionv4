"""Where concrete pieces are chosen, and the only place that knows all of them.

A model's entrypoint builds a Deployment and hands it to the pipeline, so the
runtime never imports an adapter and an adapter never imports the pipeline.

Composition is two phase on purpose. Everything a scheduled poll needs to decide
whether there is new work is cheap; loading a checkpoint is not. `Deployment`
holds the cheap half and `Deployment.load` produces the rest, so a poll that
finds nothing new never reads a 300 MB file.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .builtin.transforms import Identity
from .ports import (
    Decoder,
    Grid,
    ModelPlugin,
    Postprocess,
    Predictor,
    Preprocess,
    Sink,
    Source,
    Transform,
)


@dataclass(frozen=True)
class Loaded:
    """The expensive half: a predictor, its transform and its own stages."""

    predictor: Predictor
    transform: Transform
    preprocess: tuple[Preprocess, ...]
    postprocess: tuple[Postprocess, ...]


@dataclass(frozen=True)
class Deployment:
    """One composed deployment, before the checkpoint is read."""

    plugin: ModelPlugin
    source: Source
    decoder: Decoder
    sink: Sink
    grid: Grid
    bundle: Mapping[str, object]
    artifacts: Mapping[str, Path]
    device: str = "cpu"
    preprocess: tuple[Preprocess, ...] = ()
    postprocess: tuple[Postprocess, ...] = ()
    units: str = ""
    extra_versions: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        missing = [role for role in self.plugin.bundle_roles if role not in self.artifacts]
        if missing:
            raise ValueError(f"Bundle is missing declared roles: {', '.join(missing)}")
        if self.decoder.units != (self.units or self.decoder.units):
            raise ValueError("Configured units differ from the decoder's")
        object.__setattr__(self, "units", self.units or self.decoder.units)

    @property
    def slug(self) -> str:
        return self.plugin.slug

    def load(self) -> Loaded:
        """Read the checkpoint and build the model side of the pipeline."""
        predictor = self.plugin.load(self.artifacts, self.device, self.grid)
        transform: Transform = (
            Identity() if self.plugin.transform is None else self.plugin.transform(self.artifacts)
        )
        return Loaded(
            predictor,
            transform,
            tuple(_stages(self.plugin.preprocess, self.artifacts)),
            tuple(_stages(self.plugin.postprocess, self.artifacts)),
        )

    def input_stages(self, model_side: Sequence[Preprocess]) -> tuple[Preprocess, ...]:
        """Configured steps first, so they see units as the provider published them."""
        return (*self.preprocess, *model_side)

    def output_stages(self, model_side: Sequence[Postprocess]) -> tuple[Postprocess, ...]:
        """Model steps first, mirroring the input order so a symmetric pair undoes itself."""
        return (*model_side, *self.postprocess)


def _stages(factory, artifacts: Mapping[str, Path]):
    return () if factory is None else factory(artifacts)
