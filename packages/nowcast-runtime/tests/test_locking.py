"""The advisory lock must exclude a second holder and survive its own release."""

from nowcast_runtime.locking import exclusive_lock


def test_second_holder_is_refused_and_lock_is_reusable(tmp_path):
    lock = tmp_path / "poll.lock"
    with exclusive_lock(lock) as held:
        assert held
        with exclusive_lock(lock) as second:
            assert not second
    # Releasing must leave the lock acquirable; the file itself is never unlinked.
    with exclusive_lock(lock) as again:
        assert again
    assert lock.exists()


def test_distinct_lock_files_do_not_interfere(tmp_path):
    with exclusive_lock(tmp_path / "a.lock") as first:
        with exclusive_lock(tmp_path / "b.lock") as second:
            assert first and second


def test_missing_parent_directory_is_created(tmp_path):
    with exclusive_lock(tmp_path / "nested" / "deep" / "poll.lock") as held:
        assert held
