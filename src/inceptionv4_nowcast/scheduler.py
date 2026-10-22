"""InceptionV4 adapter for bundled scheduled polling."""

from ._runtime import scheduler
from .entrypoint import BACKEND


def poll_once(args):
    return scheduler.poll_once(args, BACKEND)
