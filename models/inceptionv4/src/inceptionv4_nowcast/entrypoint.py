"""The composition root: the only file that knows both the model and the adapters.

Because composition lives here, the runtime never imports an adapter and an
adapter never imports the pipeline. Swapping the source, the input format or the
output format is a change to this file and to the configuration, not to any
shared code.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from nowcast_runtime import pipeline, service
from nowcast_runtime.artifacts import verify_bundle
from nowcast_runtime.builtin.decoders import GSMAP_GRID, gsmap
from nowcast_runtime.builtin.sinks import GeoTiffSink
from nowcast_runtime.builtin.sources import Credentials, FtpSource, LocalFiles
from nowcast_runtime.builtin.stages import ClipNegative, FlipLatitude, MissingPolicy
from nowcast_runtime.builtin.transforms import LogZScore
from nowcast_runtime.composition import Deployment
from nowcast_runtime.config import Config
from nowcast_runtime.ports import ModelPlugin

from . import __version__
from .model import library_versions, load_model

NAME = "InceptionV4"
SLUG = "inceptionv4"
CONFIG = "config/latam.json"
POLL_ARGS = ["--config", CONFIG, "--source", "ftp", "--scheduled", "--output-dir", "outputs"]

PLUGIN = ModelPlugin(
    name=NAME,
    slug=SLUG,
    version=__version__,
    bundle_roles=("checkpoint", "mean", "std"),
    requires_runtime=">=0.3,<0.4",
    load=load_model,
    transform=lambda artifacts: LogZScore.from_arrays(artifacts["mean"], artifacts["std"]),
    library_versions=library_versions,
)


def compose(args: argparse.Namespace) -> tuple[Deployment, str]:
    """Build one deployment from the configuration and the command line."""
    config = Config.load(args.config)
    torch.set_num_threads(args.threads)
    decoder = gsmap(config.product, GSMAP_GRID)
    grid = config.domain.to_grid(decoder.native_grid)

    if args.source == "local":
        if args.input_dir is None:
            raise ValueError("--source local requires --input-dir")
        source = LocalFiles(args.input_dir, decoder.glob)
    else:
        source = FtpSource(
            directory=config.remote_directory,
            cache=args.cache_dir,
            credentials=Credentials.from_env(
                "hokusai.eorc.jaxa.jp", "FTP_JAXA_HOST", "FTP_JAXA_USER", "FTP_JAXA_PASSWORD"
            ),
            retry=config.ftp,
            # A transfer counts as complete only once it decodes on this grid.
            verify=lambda path: decoder.read(path, grid),
        )

    bundle, artifacts = verify_bundle(config.artifacts, PLUGIN.bundle_roles)
    check_bundle(bundle, config)
    transform = LogZScore.from_arrays(artifacts["mean"], artifacts["std"])
    # normalized_zero means the value this checkpoint reads as neutral, which only
    # the transform can supply, so the composition resolves it to physical units.
    fill = transform.neutral() if config.missing_policy == "normalized_zero" else 0.0
    flip = config.model_latitude_order == "south_to_north"

    plugin = PLUGIN
    if flip:
        # Latitude order is coupled to the checkpoint, so it is a model-side stage
        # and therefore runs last on input and first on output.
        plugin = replace(
            plugin,
            preprocess=lambda artifacts: (FlipLatitude(),),
            postprocess=lambda artifacts: (FlipLatitude(),),
        )

    deployment = Deployment(
        plugin=plugin,
        source=source,
        decoder=decoder,
        sink=GeoTiffSink(),
        grid=grid,
        bundle=bundle,
        artifacts=artifacts,
        device=args.device,
        preprocess=(
            ClipNegative(),
            MissingPolicy(config.missing_policy, config.max_missing_fraction, fill),
        ),
    )
    return deployment, config.product


def check_bundle(bundle: dict, config: Config) -> None:
    """Compare what the bundle declares with what the deployment configured."""
    reference = bundle.get("reference_input_product")
    if reference and config.product != reference:
        pipeline.LOG.warning(
            "%s inputs differ from %s in the supplied inference reference",
            config.product,
            reference,
        )
    declared = bundle.get("training_latitude_order")
    if declared is None:
        pipeline.LOG.warning(
            "Training latitude order is unverified; using configured %s",
            config.model_latitude_order,
        )
    elif declared != config.model_latitude_order:
        raise ValueError("Configured latitude order differs from the model bundle")


def parser():
    return pipeline.parser(SLUG, NAME)


def run(args):
    deployment, product = compose(args)
    return pipeline.run(args, deployment, product)


def main(argv=None):
    return pipeline.main(compose, SLUG, NAME, argv)


def serve():
    """Container entrypoint: verify the bundle, then poll on one cadence."""
    from nowcast_runtime import download_artifacts

    return service.serve(
        NAME,
        download=lambda: download_artifacts.main(["--bundle", str(Path("artifacts/v1"))]),
        poll=lambda: main(POLL_ARGS),
    )
