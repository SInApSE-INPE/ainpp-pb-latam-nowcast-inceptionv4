"""Versioned artifact bundle, verified against its own checksum manifest."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

BLOCK = 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(BLOCK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle(directory: Path, roles: Sequence[str]) -> tuple[dict, dict[str, Path]]:
    """Check every role a plugin declared, and only those.

    Roles are a parameter rather than a fixed triple, so a model with no
    normalization statistics, or one carrying auxiliary static fields, needs no
    change here.
    """
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported artifact manifest version")
    declared = manifest.get("files")
    if not isinstance(declared, dict):
        raise ValueError("Artifact manifest must declare a files mapping")
    paths: dict[str, Path] = {}
    for role in roles:
        if role not in declared:
            raise ValueError(f"Artifact manifest does not declare the role: {role}")
        item = declared[role]
        path = (directory / item["name"]).resolve()
        if path.parent != directory.resolve():
            raise ValueError("Artifact files must be inside their bundle directory")
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {role}")
        paths[role] = path
    return manifest, paths
