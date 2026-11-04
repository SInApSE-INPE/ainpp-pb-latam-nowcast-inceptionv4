"""Long-lived container driver: prepare the bundle, then poll on one cadence.

The runtime owns the loop and its signal handling; a model supplies only the two
callables that fetch its bundle and run one scheduled poll.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
from collections.abc import Callable, Mapping

LOG = logging.getLogger(__name__)
POLL_INTERVAL_ENV = "NOWCAST_POLL_INTERVAL_SECONDS"
DEFAULT_POLL_INTERVAL_SECONDS = 300


def poll_interval_seconds(environ: Mapping[str, str] | None = None) -> int:
    source = os.environ if environ is None else environ
    raw = source.get(POLL_INTERVAL_ENV, str(DEFAULT_POLL_INTERVAL_SECONDS))
    try:
        interval = int(raw)
    except ValueError as exc:
        raise ValueError(f"{POLL_INTERVAL_ENV} must be a positive integer") from exc
    if interval < 1:
        raise ValueError(f"{POLL_INTERVAL_ENV} must be a positive integer")
    return interval


def run_loop(
    stop: threading.Event,
    interval: int,
    download: Callable[[], int],
    poll: Callable[[], int],
) -> None:
    """Block until the bundle is valid, then poll forever. Both phases honour stop."""
    while not stop.is_set():
        if download() == 0:
            break
        LOG.warning("Model bundle is unavailable; retrying in %s seconds", interval)
        stop.wait(interval)

    while not stop.is_set():
        if poll() != 0:
            LOG.warning("Scheduled poll failed; retrying in %s seconds", interval)
        stop.wait(interval)


def serve(name: str, download: Callable[[], int], poll: Callable[[], int]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        interval = poll_interval_seconds()
    except ValueError as exc:
        LOG.error("Container configuration failed: %s", exc)
        return 2
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    LOG.info("Starting %s with a %s-second poll interval", name, interval)
    run_loop(stop, interval, download, poll)
    LOG.info("%s container stopped", name)
    return 0
