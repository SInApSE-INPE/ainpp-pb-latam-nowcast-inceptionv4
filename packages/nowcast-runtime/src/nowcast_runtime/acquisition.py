"""Turn a source listing into the exact window a run consumes.

What is left here is only the part that is the same for every provider. Listing,
transferring and pruning belong to a Source; recognizing a name belongs to a
Decoder.
"""

from __future__ import annotations

from datetime import UTC, datetime

from .ports import Decoder, Observation, Source, WindowSpec


class DataNotReady(ValueError):
    """A complete, current observation window is not available yet."""


def parse_time(value: str) -> datetime:
    """Parse an explicit run time. Alignment is checked when the window is known."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Time must include a timezone, e.g. 2026-10-19T12:00:00Z")
    return parsed.astimezone(UTC)


def catalog(
    source: Source, decoder: Decoder, window: WindowSpec, run_time: datetime | None
) -> list[Observation]:
    """Ask the source what exists and the decoder which of it is an observation."""
    observations = []
    for name, location in source.candidates(window, run_time):
        observed = decoder.observed_at(name)
        if observed is not None:
            observations.append(Observation(observed, name, location))
    return observations


def select_window(
    observations: list[Observation], window: WindowSpec, run_time: datetime | None = None
) -> list[Observation]:
    """Pick exactly the observations the window names, in chronological order.

    A partial window is never silently narrowed to an older one: a missing entry
    is DataNotReady, so a scheduled poll waits instead of publishing a forecast
    built from the wrong history.
    """
    if not observations:
        raise DataNotReady("No observations available yet")
    by_time: dict[datetime, Observation] = {}
    for item in observations:
        if item.time in by_time:
            raise ValueError(f"Ambiguous duplicate observation for {item.time.isoformat()}")
        by_time[item.time] = item
    resolved = run_time if run_time is not None else max(by_time)
    required = window.times(resolved)
    absent = [moment.isoformat() for moment in required if moment not in by_time]
    if absent:
        raise DataNotReady(f"Incomplete window; missing: {', '.join(absent)}")
    return [by_time[moment] for moment in required]


def latest_time(observations: list[Observation]) -> datetime:
    """The most recent observation time, for deciding whether a poll has work."""
    if not observations:
        raise DataNotReady("No observations available yet")
    return max(item.time for item in observations)
