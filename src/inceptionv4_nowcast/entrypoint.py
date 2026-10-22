"""InceptionV4 command-line entrypoint using the bundled operational pipeline."""

import timm
import torchvision

from . import __version__
from ._runtime import pipeline
from ._runtime.backend import Backend
from .model import forecast_steps, load_model

BACKEND = Backend(
    "InceptionV4",
    __version__,
    load_model,
    forecast_steps,
    library_versions={"timm": timm.__version__, "torchvision": torchvision.__version__},
)


def parser():
    return pipeline.parser(BACKEND)


def run(args):
    return pipeline.run(args, BACKEND)


def main(argv=None):
    return pipeline.main(BACKEND, argv)
