"""Managed transfers expire; supplied inputs and foreign directories never do."""

import os
import tempfile
from datetime import timedelta
from pathlib import Path

import pytest

from inceptionv4_nowcast.entrypoint import parser
from nowcast_runtime.builtin.sources import Credentials, FtpSource, RetryPolicy
from nowcast_runtime.pipeline import validate_runtime


def test_cache_defaults_to_temporary_storage_with_positive_ttl():
    args = parser().parse_args(["--config", "unused.json"])
    assert args.cache_dir.is_relative_to(Path(tempfile.gettempdir()))
    assert args.cache_ttl_hours == 24
    args.cache_ttl_hours = 0
    with pytest.raises(ValueError, match="cache-ttl-hours"):
        validate_runtime(args)


def ftp_source(cache):
    return FtpSource(
        directory="/now/latest",
        cache=cache,
        credentials=Credentials("host.invalid", "user", "secret"),
        retry=RetryPolicy(60.0, 1, 1.0, 1.0),
        verify=lambda path: None,
    )


def test_identity_never_carries_credentials(tmp_path):
    identity = ftp_source(tmp_path).identity
    assert identity == "ftp://host.invalid/now/latest"
    assert "secret" not in identity and "user" not in identity


def test_prune_removes_expired_downloads_only(tmp_path):
    cache = tmp_path / "cache"
    managed = cache / "0123456789abcdef"
    managed.mkdir(parents=True)
    foreign = cache / "user-inputs"
    foreign.mkdir()
    old = managed / "gsmap_now.20230101.0000.dat.gz"
    fresh = managed / "gsmap_now.20230102.0000.dat.gz"
    partial = managed / "interrupted.part"
    unrelated = foreign / "gsmap_now.20230101.0000.dat.gz"
    outside = tmp_path / "local-input.dat.gz"
    for path in (old, fresh, partial, unrelated, outside):
        path.write_bytes(b"input")
    now = 200_000.0
    for path in (old, partial, unrelated, outside):
        os.utime(path, (now - 25 * 3600, now - 25 * 3600))
    os.utime(fresh, (now - 23 * 3600, now - 23 * 3600))
    link = managed / "linked.dat.gz"
    try:
        link.symlink_to(outside)
    except OSError as exc:  # Windows grants symlink creation only with a privilege.
        pytest.skip(f"symlinks unavailable on this platform: {exc}")

    source = ftp_source(cache)
    assert source.prune(timedelta(hours=24), now=now) == 2
    assert not old.exists() and not partial.exists()
    assert fresh.exists() and unrelated.exists() and outside.exists() and link.is_symlink()
    assert source.prune(timedelta(hours=24), now=now) == 0


def test_prune_rejects_a_symlinked_cache_root_and_a_bad_ttl(tmp_path):
    source = ftp_source(tmp_path / "absent")
    assert source.prune(timedelta(hours=1)) == 0
    with pytest.raises(ValueError, match="cache TTL must be positive"):
        ftp_source(tmp_path).prune(timedelta(0))
