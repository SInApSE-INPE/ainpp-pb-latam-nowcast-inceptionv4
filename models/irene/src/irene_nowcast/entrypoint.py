"""The composition root: the only file knowing both the model and the adapters."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from nowcast_runtime import pipeline, service
from nowcast_runtime.artifacts import verify_bundle
from nowcast_runtime.builtin.decoders import TimestampedName
from nowcast_runtime.builtin.sinks import GeoTiffSink
from nowcast_runtime.builtin.sources import LocalFiles
from nowcast_runtime.builtin.stages import ClipNegative, MissingPolicy
from nowcast_runtime.composition import Deployment
from nowcast_runtime.config import Config
from nowcast_runtime.ports import Grid, ModelPlugin

from . import __version__
from .decoder import RadarNetcdf
from .model import OUTPUT_CEILING, OUTPUT_FLOOR, library_versions, load_model, transform
from .stages import Threshold

NAME = "IRENE"
SLUG = "irene"
CONFIG = "config/default.json"
POLL_ARGS = ["--config", CONFIG, "--source", "ftp", "--scheduled", "--output-dir", "outputs"]

# The DPC mosaic as published: 1 km cells on a projected metre grid, rows north
# first. Coordinates in the sample are cell centres, so the outer corner is half
# a cell beyond: x starts at -599500, the grid origin at -600000.
#
# The CRS is an ASSERTION, not a fact. The sample carries grid_mapping = "crs"
# and contains no such variable, so the projection is not determinable from the
# data. It is declared here so it reaches the manifest and can be contradicted,
# rather than being left implicit.
ASSUMED_CRS = "EPSG:3003"
NATIVE_GRID = Grid(ASSUMED_CRS, -600000.0, 650000.0, 1000.0, 1000.0, 1200, 1400)

PLUGIN = ModelPlugin(
    name=NAME,
    slug=SLUG,
    version=__version__,
    bundle_roles=("weights",),
    requires_runtime=">=0.3,<0.4",
    load=load_model,
    transform=transform,
    # Both ends of the reflectivity scale saturate, and the published inference
    # path applies neither bound. Zeroing the floor keeps dry pixels dry instead
    # of a uniform 0.0365 mm/h drizzle over the whole domain.
    postprocess=lambda artifacts: (
        Threshold(floor=OUTPUT_FLOOR, ceiling=OUTPUT_CEILING),
    ),
    library_versions=library_versions,
)


def decoder() -> RadarNetcdf:
    """One NetCDF per five-minute scan, variable RR, rain rate in mm/h."""
    return RadarNetcdf(
        native_grid=NATIVE_GRID,
        naming=TimestampedName(
            pattern=r"dpc_sri\.(?P<stamp>\d{12})\.nc",
            stamp_format="%Y%m%d%H%M",
            template="dpc_sri.%Y%m%d%H%M.nc",
        ),
        variable="RR",
        units="mm/h",
    )


def compose(args: argparse.Namespace) -> tuple[Deployment, str]:
    config = Config.load(args.config)
    reader = decoder()
    grid = config.domain.to_grid(reader.native_grid)

    if args.source == "local":
        if args.input_dir is None:
            raise ValueError("--source local requires --input-dir")
        source = LocalFiles(args.input_dir, reader.glob)
    else:
        raise ValueError(
            "IRENE has no remote source configured; the DPC feed is not public. "
            "Supply scans with --source local --input-dir."
        )

    bundle, artifacts = verify_bundle(config.artifacts, PLUGIN.bundle_roles)

    # Ensemble size and the noise seed are operationally variable, so they live
    # in configuration. The runtime carries keys it does not own in `extra`.
    members = int(config.extra.get("ensemble_members", 10))
    seed = config.extra.get("seed")
    seed = None if seed is None else int(seed)
    plugin = replace(
        PLUGIN,
        load=lambda a, device, g: load_model(a, device, g, members=members, seed=seed),
        library_versions=lambda: {
            **library_versions(),
            "irene.ensemble_members": str(members),
            "irene.seed": "unseeded" if seed is None else str(seed),
        },
    )

    deployment = Deployment(
        plugin=plugin,
        source=source,
        decoder=reader,
        sink=GeoTiffSink(variable_names=("precipitation_rate",)),
        grid=grid,
        bundle=bundle,
        artifacts=artifacts,
        device=args.device,
        preprocess=(
            # A negative rain rate would become NaN under R**1.6.
            ClipNegative(),
            # Radar coverage leaves most of the rectangle empty, so the limit
            # here is far looser than a satellite product would need. The mask
            # carries the gaps to the output, which the published inference path
            # does not do.
            MissingPolicy(config.missing_policy, config.max_missing_fraction),
        ),
    )
    return deployment, config.product


def parser():
    return pipeline.parser(SLUG, NAME)


def run(args):
    deployment, product = compose(args)
    return pipeline.run(args, deployment, product)


def main(argv=None):
    return pipeline.main(compose, SLUG, NAME, argv)


def serve():
    from nowcast_runtime import download_artifacts

    return service.serve(
        NAME,
        download=lambda: download_artifacts.main(["--bundle", str(Path("artifacts/v1"))]),
        poll=lambda: main(POLL_ARGS),
    )


__all__ = ["NAME", "SLUG", "PLUGIN", "compose", "decoder", "parser", "run", "main", "serve"]

# Spacing between scans, exported so a sample generator can use it.
SCAN_INTERVAL = timedelta(minutes=5)
