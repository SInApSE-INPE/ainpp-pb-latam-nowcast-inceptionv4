"""Orchestration: acquire, decode, stage, infer, and atomically publish.

The invariant sequence lives here and a deployment fills the steps. That is what
makes the operational guarantees uniform across models: nothing partial is ever
published, a complete run is never overwritten, and every run carries a manifest
recording exactly what produced it.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from . import __version__
from .acquisition import DataNotReady, catalog, parse_time, select_window
from .artifacts import sha256
from .composition import Deployment
from .ports import Prediction

LOG = logging.getLogger(__name__)


def default_cache_dir(slug: str) -> Path:
    """Isolate per user: a POSIX /tmp is shared, the Windows temp dir already is not."""
    user = f"-{os.getuid()}" if hasattr(os, "getuid") else ""
    return Path(tempfile.gettempdir()) / f"nowcasting{user}" / slug


def parser(slug: str, name: str) -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=f"{name} gridded forecasts to files (UTC)")
    cli.add_argument("--config", type=Path, required=True)
    cli.add_argument("--source", choices=("local", "ftp"), default="local")
    cli.add_argument("--input-dir", type=Path, help="Directory of supplied inputs (recursive)")
    cli.add_argument(
        "--cache-dir",
        type=Path,
        default=default_cache_dir(slug),
        help="Temporary directory for transfers; supplied inputs are never removed",
    )
    cli.add_argument(
        "--cache-ttl-hours",
        type=float,
        default=24.0,
        help="Delete expired transfers on each run or scheduled poll (default: 24 hours)",
    )
    cli.add_argument("--output-dir", type=Path, default=Path("outputs"))
    cli.add_argument(
        "--run-time", type=parse_time, help="Timestamp of LAST input, timezone required"
    )
    cli.add_argument(
        "--steps", type=int, help="Forecast leads; defaults to the native model horizon"
    )
    cli.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    cli.add_argument("--threads", type=int, default=4, help="CPU threads for inference")
    cli.add_argument("--max-age-hours", type=float, default=12, help="Latest window age limit")
    cli.add_argument("--verbose", action="store_true")
    cli.add_argument(
        "--scheduled",
        action="store_true",
        help="One live poll: skip completed runs, lock concurrent polls, wait for missing data",
    )
    return cli


def validate_runtime(args: argparse.Namespace) -> None:
    if (args.steps is not None and args.steps < 1) or args.threads < 1:
        raise ValueError("steps and threads must be positive")
    if not math.isfinite(args.max_age_hours) or args.max_age_hours <= 0:
        raise ValueError("max-age-hours must be finite and positive")
    if not math.isfinite(args.cache_ttl_hours) or args.cache_ttl_hours <= 0:
        raise ValueError("cache-ttl-hours must be finite and positive")


def forecast_directory(root: Path, product: str, run_time: datetime, slug: str) -> Path:
    return root / slug / product / run_time.strftime("%Y/%m/%d/%H%MZ")


def require_fresh(run_time: datetime, max_age_hours: float) -> None:
    age = (datetime.now(UTC) - run_time).total_seconds() / 3600
    if age < 0 or age > max_age_hours:
        raise DataNotReady(f"Latest observation is stale or in the future (age={age:.1f}h)")


def prune(deployment: Deployment, args: argparse.Namespace) -> None:
    removed = deployment.source.prune(timedelta(hours=args.cache_ttl_hours))
    if removed:
        LOG.info("Removed %d expired input files from %s", removed, args.cache_dir)


def apply_stages(payload, stages):
    for stage in stages:
        payload = stage.apply(payload)
    return payload


def run(args: argparse.Namespace, deployment: Deployment, product: str) -> Path:
    validate_runtime(args)
    prune(deployment, args)
    loaded = deployment.load()
    predictor, transform = loaded.predictor, loaded.transform
    window = predictor.window
    steps = args.steps if args.steps is not None else predictor.output_steps
    if steps > predictor.output_steps:
        raise ValueError(f"Requested {steps} leads exceeds native horizon {predictor.output_steps}")
    if predictor.grid != deployment.grid:
        raise ValueError("Model was loaded for a grid other than the deployment grid")
    LOG.info("Model ready on %s; %s input steps", deployment.device, len(window))

    observations = select_window(
        catalog(deployment.source, deployment.decoder, window, args.run_time),
        window,
        args.run_time,
    )
    run_time = observations[-1].time
    if args.run_time is None and deployment.source.live:
        require_fresh(run_time, args.max_age_hours)
    target = forecast_directory(args.output_dir, product, run_time, deployment.slug)
    if target.exists():
        raise FileExistsError(f"Run already exists: {target}; use a different output directory")

    paths = [deployment.source.fetch(item) for item in observations]
    LOG.info("Acquired %s consecutive inputs ending at %s", len(paths), run_time.isoformat())
    frames = [deployment.decoder.read(path, deployment.grid) for path in paths]
    fractions = [float(frame.missing.mean()) for frame in frames]
    staged = [apply_stages(frame, deployment.input_stages(loaded.preprocess)) for frame in frames]
    # A cell missing anywhere in the history is masked in every output.
    missing = np.any([frame.missing for frame in staged], axis=0)
    inputs = transform.forward(np.stack([frame.values for frame in staged]))

    metadata = manifest(args, deployment, loaded, observations, paths, fractions, missing, steps)
    metadata["run_time"] = run_time.isoformat()
    metadata["config_product"] = product
    target.parent.mkdir(parents=True, exist_ok=True)
    # Publish the whole run only once every file and its manifest exist.
    with tempfile.TemporaryDirectory(prefix=f".{deployment.slug}-", dir=target.parent) as temporary:
        staging = Path(temporary) / "run"
        staging.mkdir()
        for lead, field in enumerate(predictor.leads(inputs, steps), start=1):
            if lead > steps:
                raise ValueError("Model yielded more leads than requested")
            prediction = Prediction(
                transform.inverse(field),
                np.broadcast_to(missing, field.shape).copy(),
                lead,
                run_time + lead * predictor.step,
            )
            prediction = apply_stages(prediction, deployment.output_stages(loaded.postprocess))
            if prediction.values.shape[1:] != (staged[0].variables, *deployment.grid.shape):
                raise ValueError("Model returned an invalid forecast shape")
            for member in range(prediction.members):
                name = deployment.sink.filename(deployment.slug, run_time, prediction, member)
                deployment.sink.write(
                    staging / name,
                    prediction,
                    member,
                    deployment.grid,
                    deployment.units,
                    tags(deployment, product, run_time, prediction),
                )
                metadata["outputs"].append(
                    {
                        "name": name,
                        "valid_time": prediction.valid_time.isoformat(),
                        "lead_hours": lead,
                        "member": member,
                        "sha256": sha256(staging / name),
                    }
                )
            LOG.info("Generated lead +%s: %s", lead, prediction.valid_time.isoformat())
        if len({item["lead_hours"] for item in metadata["outputs"]}) != steps:
            raise ValueError("Model returned fewer leads than requested")
        (staging / "manifest.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        staging.rename(target)
    LOG.info("Published complete run: %s", target)
    return target


def tags(deployment: Deployment, product: str, run_time: datetime, prediction: Prediction) -> dict:
    return {
        "model": deployment.plugin.name,
        "model_version": deployment.bundle["version"],
        "run_time": run_time.isoformat(),
        "valid_time": prediction.valid_time.isoformat(),
        "lead_hours": prediction.lead,
        "input_product": product,
    }
    # units are written by the sink itself, so they are deliberately absent here.


def manifest(args, deployment, loaded, observations, paths, fractions, missing, steps) -> dict:
    return {
        "schema_version": 1,
        "status": "complete",
        "created_at": datetime.now(UTC).isoformat(),
        "model": deployment.plugin.name,
        "model_version": deployment.plugin.version,
        "bundle_version": deployment.bundle["version"],
        "artifacts": deployment.bundle["files"],
        "normalization": dict(loaded.transform.describe()),
        "preprocess": [
            dict(stage.describe()) for stage in deployment.input_stages(loaded.preprocess)
        ],
        "postprocess": [
            dict(stage.describe()) for stage in deployment.output_stages(loaded.postprocess)
        ],
        "source": deployment.source.identity,
        "units": deployment.units,
        "crs": deployment.grid.crs,
        "grid": asdict(deployment.grid),
        "device": deployment.device,
        "cache_ttl_hours": args.cache_ttl_hours,
        "versions": {
            "nowcast-runtime": __version__,
            deployment.plugin.slug: deployment.plugin.version,
            **dict(deployment.plugin.library_versions()),
            **dict(deployment.extra_versions),
            "numpy": np.__version__,
        },
        "input_steps": len(loaded.predictor.window),
        "input_offsets": [offset.total_seconds() for offset in loaded.predictor.window.offsets],
        "step_seconds": loaded.predictor.step.total_seconds(),
        "output_steps": steps,
        "native_output_steps": loaded.predictor.output_steps,
        "masked_output_fraction": float(missing.mean()),
        "inputs": [
            {
                "name": item.name,
                "time": item.time.isoformat(),
                "source": item.location,
                "sha256": sha256(path),
                "missing_fraction": fraction,
            }
            for item, path, fraction in zip(observations, paths, fractions, strict=True)
        ],
        "outputs": [],
    }


def main(compose, slug: str, name: str, argv=None) -> int:
    """compose(args) builds the Deployment; nothing here knows a concrete adapter."""
    args = parser(slug, name).parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        if args.scheduled:
            from .scheduler import poll_once

            poll_once(args, compose)
        else:
            deployment, product = compose(args)
            run(args, deployment, product)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, EOFError) as exc:
        LOG.error("Pipeline failed: %s", exc, exc_info=args.verbose)
        return 1
    return 0
