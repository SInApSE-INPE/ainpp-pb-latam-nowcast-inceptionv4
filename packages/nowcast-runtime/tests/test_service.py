"""Container startup retries and scheduled polling use one configured cadence."""

import pytest

from nowcast_runtime import service


class StopAfterWaits:
    """Stop the loop after a fixed number of waits and record each interval."""

    def __init__(self, count):
        self.count = count
        self.waits = []

    def is_set(self):
        return len(self.waits) >= self.count

    def wait(self, seconds):
        self.waits.append(seconds)


@pytest.mark.parametrize("value", ["0", "-5", "abc", "1.5"])
def test_poll_interval_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="positive integer"):
        service.poll_interval_seconds({service.POLL_INTERVAL_ENV: value})


def test_poll_interval_uses_configured_seconds():
    assert service.poll_interval_seconds({service.POLL_INTERVAL_ENV: "420"}) == 420


def test_poll_interval_defaults_when_unset():
    assert service.poll_interval_seconds({}) == service.DEFAULT_POLL_INTERVAL_SECONDS


def test_bundle_retry_precedes_first_poll():
    order = []
    results = iter([1, 0])
    stop = StopAfterWaits(2)
    service.run_loop(
        stop,
        300,
        lambda: (order.append("download"), next(results))[1],
        lambda: (order.append("poll"), 0)[1],
    )
    assert order == ["download", "download", "poll"]
    assert stop.waits == [300, 300]


def test_failed_poll_retries_on_next_interval():
    polls = []

    def poll():
        polls.append(1)
        return 1 if len(polls) == 1 else 0

    stop = StopAfterWaits(2)
    service.run_loop(stop, 60, lambda: 0, poll)
    assert len(polls) == 2
    assert stop.waits == [60, 60]
