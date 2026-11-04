"""Standard-library wrapper for the shared artifact download command.

Defaults to this model's own bundle, so it works from any working directory.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = MODEL_ROOT.parents[1]
sys.path.insert(0, str(WORKSPACE / "packages" / "nowcast-runtime" / "src"))
main = importlib.import_module("nowcast_runtime.download_artifacts").main

if __name__ == "__main__":
    arguments = sys.argv[1:]
    if not any(item == "--bundle" or item.startswith("--bundle=") for item in arguments):
        arguments = ["--bundle", str(MODEL_ROOT / "artifacts" / "v1"), *arguments]
    raise SystemExit(main(arguments))
