"""Built-in sources. Each one owns its path syntax and reports bare names."""

from __future__ import annotations

import ftplib
import hashlib
import logging
import math
import os
import re
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath

from ..ports import Observation, WindowSpec

LOG = logging.getLogger(__name__)
NAMESPACE = re.compile(r"[0-9a-f]{16}")
PARTIAL = ".part"


class IncompleteDownload(OSError):
    """The remote object may still be uploading; retry the transfer."""


@dataclass(frozen=True)
class RetryPolicy:
    timeout_seconds: float
    attempts: int
    retry_delay_seconds: float
    max_retry_delay_seconds: float

    def __post_init__(self):
        if type(self.attempts) is not int or self.attempts < 1:
            raise ValueError("attempts must be a positive integer")
        for value in (self.timeout_seconds, self.retry_delay_seconds, self.max_retry_delay_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("timeouts and retry delays must be finite and positive")
        if self.max_retry_delay_seconds < self.retry_delay_seconds:
            raise ValueError("maximum retry delay must not be less than the initial delay")

    def delay(self, attempt: int) -> float:
        return min(self.retry_delay_seconds * 2**attempt, self.max_retry_delay_seconds)


@dataclass(frozen=True)
class Credentials:
    """Read from the environment, never from configuration or a command line."""

    host: str
    user: str
    password: str

    @classmethod
    def from_env(
        cls,
        default_host: str,
        host_var: str,
        user_var: str,
        password_var: str,
        environ: Mapping[str, str] | None = None,
    ) -> Credentials:
        source = os.environ if environ is None else environ
        user, password = source.get(user_var), source.get(password_var)
        if not user or not password:
            raise ValueError(f"Set {user_var} and {password_var} for remote acquisition")
        return cls(source.get(host_var, default_host), user, password)


class LocalFiles:
    """Observations already on disk. Supplied inputs are never removed."""

    def __init__(self, directory: Path, glob: str = "*"):
        if not directory.is_dir():
            raise ValueError(f"Input directory does not exist: {directory}")
        self.directory = directory
        self.glob = glob

    @property
    def identity(self) -> str:
        return f"local:{self.directory}"

    @property
    def live(self) -> bool:
        """Supplied inputs are reprocessed on purpose, so staleness is not a fault."""
        return False

    def candidates(
        self, window: WindowSpec, run_time: datetime | None
    ) -> Sequence[tuple[str, str]]:
        # The bare name comes from pathlib, so native separators never leak into
        # the name matching a decoder performs.
        return [(path.name, str(path.resolve())) for path in self.directory.rglob(self.glob)]

    def fetch(self, observation: Observation) -> Path:
        return Path(observation.location)

    def prune(self, ttl: timedelta) -> int:
        return 0


class FtpSource:
    """FTP with bounded retries and a verified, content-addressed local cache."""

    def __init__(
        self,
        directory: str,
        cache: Path,
        credentials: Credentials,
        retry: RetryPolicy,
        verify: Callable[[Path], None],
        connect: Callable[[], ftplib.FTP] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.directory = directory
        self.cache = cache
        self.retry = retry
        self._credentials = credentials
        self._verify = verify
        self._connect = connect
        self._sleep = sleep

    @property
    def host(self) -> str:
        return self._credentials.host

    @property
    def identity(self) -> str:
        """Host and directory only: credentials never reach a run manifest."""
        return f"ftp://{self._credentials.host}{self.directory}"

    @property
    def live(self) -> bool:
        """A latest directory keeps changing; an archive template does not."""
        return not self.templated

    @property
    def templated(self) -> bool:
        """True when the directory is an archive template needing an explicit time."""
        return "%" in self.directory

    def _session(self) -> ftplib.FTP:
        if self._connect is not None:
            return self._connect()
        session = ftplib.FTP(timeout=self.retry.timeout_seconds)
        session.connect(self._credentials.host)
        session.login(self._credentials.user, self._credentials.password)
        return session

    def _request(self, operation):
        for attempt in range(self.retry.attempts):
            try:
                with self._session() as session:
                    return operation(session)
            except ftplib.all_errors:
                if attempt + 1 == self.retry.attempts:
                    # Never surface server responses or credentials in logs.
                    raise ConnectionError(
                        f"FTP request failed after {self.retry.attempts} attempts"
                    ) from None
                delay = self.retry.delay(attempt)
                LOG.warning(
                    "FTP request failed; retry %s/%s in %ss",
                    attempt + 2,
                    self.retry.attempts,
                    delay,
                )
                self._sleep(delay)
        raise AssertionError("unreachable: attempts is always positive")

    def _directories(self, window: WindowSpec, run_time: datetime | None) -> list[str]:
        if not self.templated:
            return [self.directory]
        if run_time is None:
            raise ValueError("Archive directory templates require an explicit run time")
        return sorted({moment.strftime(self.directory) for moment in window.times(run_time)})

    def candidates(
        self, window: WindowSpec, run_time: datetime | None
    ) -> Sequence[tuple[str, str]]:
        entries: list[tuple[str, str]] = []
        for directory in self._directories(window, run_time):

            def listing(session, directory=directory):
                session.cwd(directory)
                return session.nlst()

            for raw in self._request(listing):
                # Remote locations are POSIX whatever platform this client runs on.
                name = PurePosixPath(raw).name
                entries.append((name, str(PurePosixPath(directory) / name)))
        return entries

    def fetch(self, observation: Observation) -> Path:
        # Host and remote directory identity separate provider and product versions.
        digest = hashlib.sha256(f"{self.host}:{observation.location}".encode()).hexdigest()
        target = self.cache / digest[:16] / observation.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            try:
                self._verify(target)
                return target
            except (OSError, EOFError, ValueError):
                LOG.warning("Invalid cached file; downloading %s again", observation.name)
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=PARTIAL, delete=False) as handle:
            temporary = Path(handle.name)
        try:

            def retrieve(session):
                with temporary.open("wb") as stream:
                    session.retrbinary(f"RETR {observation.location}", stream.write)
                try:
                    self._verify(temporary)
                except (OSError, EOFError, ValueError) as exc:
                    raise IncompleteDownload(
                        f"Incomplete transfer: {observation.name}"
                    ) from exc

            self._request(retrieve)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def prune(self, ttl: timedelta, now: float | None = None) -> int:
        """Remove expired downloads from managed hash directories only.

        Scoping to directories whose names are sixteen hex characters means only
        this cache's own namespaces are touched, so a user directory that happens
        to sit under the same root is never considered.
        """
        if ttl <= timedelta(0):
            raise ValueError("cache TTL must be positive")
        if not self.cache.exists():
            return 0
        if self.cache.is_symlink() or not self.cache.is_dir():
            raise ValueError(f"Cache path must be a directory, not a symlink: {self.cache}")
        cutoff = (time.time() if now is None else now) - ttl.total_seconds()
        removed = 0
        for namespace in self.cache.iterdir():
            if (
                namespace.is_symlink()
                or not namespace.is_dir()
                or not NAMESPACE.fullmatch(namespace.name)
            ):
                continue
            for path in namespace.iterdir():
                if path.is_symlink() or not path.is_file():
                    continue
                try:
                    if path.stat().st_mtime < cutoff:
                        path.unlink()
                        removed += 1
                except FileNotFoundError:
                    continue
            try:
                namespace.rmdir()
            except OSError:
                pass
        return removed
