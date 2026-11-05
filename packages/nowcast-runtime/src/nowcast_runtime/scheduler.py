"""One idempotent polling cycle. Any scheduler may drive it."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from .acquisition import DataNotReady, catalog, latest_time
from .artifacts import sha256
from .locking import exclusive_lock
from .pipeline import forecast_directory, prune, require_fresh, run, validate_runtime
from .ports import WindowSpec

LOG = logging.getLogger(__name__)
# One observation is enough to learn whether anything new was published.
PROBE = WindowSpec((timedelta(0),))


def poll_lock(output_dir: Path, product: str, slug: str):
    """Advisory lock shared by scheduled invocations using the same output root."""
    return exclusive_lock(output_dir / f".{slug}-{product}.lock")


def completed_run(
    target: Path, timestamp: datetime, product: str, steps: int | None, model: str
) -> bool:
    if not target.exists():
        return False
    # A malformed or partial existing target is an error, never a successful skip.
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    steps = manifest.get("native_output_steps") if steps is None else steps
    if type(steps) is not int or steps < 1:
        raise ValueError(f"Existing run has no valid native horizon: {target}")
    leads = {item["lead_hours"] for item in manifest["outputs"]}
    if (
        manifest["model"] != model
        or manifest["status"] != "complete"
        or manifest["run_time"] != timestamp.isoformat()
        or manifest["config_product"] != product
        or manifest["output_steps"] != steps
        or leads != set(range(1, steps + 1))
    ):
        raise ValueError(f"Existing run is incomplete or incompatible: {target}")
    for item in manifest["outputs"]:
        path = (target / item["name"]).resolve()
        if path.parent != target.resolve() or sha256(path) != item["sha256"]:
            raise ValueError(f"Existing run failed output integrity checks: {target}")
    return True


def poll_once(args: argparse.Namespace, compose) -> Path | None:
    validate_runtime(args)
    if args.source != "ftp" or args.run_time is not None:
        raise ValueError("--scheduled requires --source ftp and no --run-time")
    deployment, product = compose(args)
    if getattr(deployment.source, "templated", False):
        raise ValueError("--scheduled requires a latest directory, not an archive template")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with poll_lock(args.output_dir, product, deployment.slug) as acquired:
        if not acquired:
            LOG.info("BUSY: another scheduled invocation is running")
            return None
        try:
            prune(deployment, args)
            # A cheap listing precedes loading the checkpoint. Pin the discovered
            # time so a newly published file cannot change this cycle's target.
            latest = latest_time(
                catalog(deployment.source, deployment.decoder, PROBE, None)
            )
            require_fresh(latest, args.max_age_hours)
            target = forecast_directory(args.output_dir, product, latest, deployment.slug)
            if completed_run(target, latest, product, args.steps, deployment.plugin.name):
                LOG.info("UP_TO_DATE: %s already completed", latest.isoformat())
                return None
            LOG.info("NEW_DATA: processing %s", latest.isoformat())
            invocation = argparse.Namespace(**vars(args))
            invocation.run_time = latest
            invocation.scheduled = False
            return run(invocation, deployment, product)
        except DataNotReady as exc:
            LOG.info("WAITING: %s; retry on the next cycle", exc)
            return None
