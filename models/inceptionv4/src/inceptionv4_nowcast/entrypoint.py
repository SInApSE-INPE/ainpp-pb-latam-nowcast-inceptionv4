"""InceptionV4 wiring: the only file that knows both the model and the runtime."""

import timm
import torchvision

from nowcast_runtime import pipeline, service
from nowcast_runtime.backend import Backend

from . import __version__
from .model import forecast_steps, load_model

BACKEND = Backend(
    "InceptionV4",
    __version__,
    load_model,
    forecast_steps,
    library_versions={"timm": timm.__version__, "torchvision": torchvision.__version__},
)
CONFIG = "config/latam.json"
POLL_ARGS = ["--config", CONFIG, "--source", "ftp", "--scheduled", "--output-dir", "outputs"]


def parser():
    return pipeline.parser(BACKEND)


def run(args):
    return pipeline.run(args, BACKEND)


def main(argv=None):
    return pipeline.main(BACKEND, argv)


def serve():
    """Container entrypoint: verify the bundle, then poll FTP on one cadence."""
    from nowcast_runtime import download_artifacts

    return service.serve(
        BACKEND.name,
        download=lambda: download_artifacts.main([]),
        poll=lambda: main(POLL_ARGS),
    )
