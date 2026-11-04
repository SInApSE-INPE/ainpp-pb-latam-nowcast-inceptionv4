"""Advisory, non-blocking, OS-owned exclusive lock for a single host."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

if os.name == "nt":
    import msvcrt

    def _acquire(fd: int) -> bool:
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    def _release(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _acquire(fd: int) -> bool:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True

    def _release(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


@contextmanager
def exclusive_lock(path: Path):
    """Yield True while this process holds the lock, False if another process holds it.

    The operating system owns the lock, so it is released even when the process is
    killed; a leftover file never blocks a later poll. The lock covers one host and
    one filesystem, not replicas sharing a network volume.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Do not unlink: replacing the inode would allow overlapping lock holders.
    with path.open("a") as stream:
        descriptor = stream.fileno()
        if not _acquire(descriptor):
            yield False
            return
        try:
            yield True
        finally:
            _release(descriptor)
