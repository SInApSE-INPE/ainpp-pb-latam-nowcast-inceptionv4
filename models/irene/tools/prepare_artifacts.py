"""Extract inference-only weights from the published IRENE checkpoint.

The checkpoint on the Hub is a PyTorch Lightning training artefact: 770 MB of
which most is optimizer state, learning-rate schedulers, callbacks and loop
state. None of that is needed to run the model, and keeping it means every
deployment downloads and verifies it.

This produces a bundle holding tensors only, which the operational loader can
then read with ``weights_only=True``. Reading the published checkpoint still
requires ``weights_only=False`` because its hyperparameters pickle optimizer
classes, so that happens once, here, rather than on every container start.

    uv run python tools/prepare_artifacts.py \
        --checkpoint model.ckpt --bundle artifacts/v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

PREFIX = "model."
WEIGHTS = "irene.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--checkpoint", type=Path, required=True)
    cli.add_argument("--bundle", type=Path, required=True)
    cli.add_argument("--source-url", default="https://huggingface.co/it4lia/irene")
    args = cli.parse_args()

    args.bundle.mkdir(parents=True, exist_ok=True)
    target = args.bundle / WEIGHTS
    manifest_path = args.bundle / "manifest.json"
    if target.exists() or manifest_path.exists():
        raise FileExistsError("Use a new bundle directory; existing artifacts are preserved")

    source_digest = sha256(args.checkpoint)
    # weights_only=False is unavoidable here: the published checkpoint pickles
    # optimizer and scheduler classes in its hyperparameters. It is confined to
    # this one offline step, against a checkpoint whose digest is recorded.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    state = checkpoint["state_dict"]
    weights = {}
    for key, value in state.items():
        if not key.startswith(PREFIX):
            continue
        if not torch.is_tensor(value):
            raise ValueError(f"Not a tensor under {key}")
        weights[key[len(PREFIX) :]] = value
    if not weights:
        raise ValueError("No model.* tensors found in the checkpoint state dictionary")

    hyper = checkpoint.get("hyper_parameters", {})
    torch.save(weights, target)

    manifest = {
        "schema_version": 1,
        "version": args.bundle.name,
        "model": "IRENE",
        "architecture": {
            "class": "EncoderDecoder",
            "input_channels": hyper.get("input_channels"),
            "num_blocks": hyper.get("num_blocks"),
            "forecast_steps": hyper.get("forecast_steps"),
            "ensemble_size": hyper.get("ensemble_size"),
            "noisy_decoder": hyper.get("noisy_decoder"),
        },
        "source": {
            "url": args.source_url,
            "name": args.checkpoint.name,
            "sha256": source_digest,
            "license": "BSD-2-Clause",
        },
        # The dBZ transform saturates, so values below this are not representable
        # and the model cannot emit anything smaller. Recorded so the deployment
        # does not have to rediscover it.
        "output_floor_mm_h": 0.036463,
        "files": {"weights": {"name": target.name, "sha256": sha256(target)}},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    before = args.checkpoint.stat().st_size
    after = target.stat().st_size
    print(f"Extracted {len(weights)} tensors")
    print(f"  {before:>14,} bytes  {args.checkpoint.name}")
    print(f"  {after:>14,} bytes  {target.name}  ({100 * after / before:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
