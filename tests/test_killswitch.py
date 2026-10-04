import json

import pytest

from papertrader.risk import KillSwitch


def test_trip_persists_across_instances(tmp_path):
    KillSwitch(tmp_path).trip("drawdown", "2020-01-02")
    again = KillSwitch(tmp_path)  # e.g. tomorrow's run, or a different process
    assert again.tripped and again.reason() == "drawdown"


def test_trip_from_another_process_is_seen_immediately(tmp_path):
    runner_view = KillSwitch(tmp_path)
    assert not runner_view.tripped
    KillSwitch(tmp_path).trip("manual: CLI")
    assert runner_view.tripped


def test_sentinel_file_trips_and_must_be_removed_by_hand(tmp_path):
    switch = KillSwitch(tmp_path)
    (tmp_path / "KILL").touch()
    assert switch.tripped and "sentinel" in switch.reason()
    with pytest.raises(PermissionError):
        switch.reset(confirm=True)
    (tmp_path / "KILL").unlink()
    switch.reset(confirm=True)
    assert not switch.tripped


def test_reset_needs_confirmation(tmp_path):
    switch = KillSwitch(tmp_path)
    switch.trip("x")
    with pytest.raises(PermissionError):
        switch.reset()
    assert switch.tripped
    switch.reset(confirm=True)
    assert not KillSwitch(tmp_path).tripped


def test_second_trip_keeps_first_reason():
    switch = KillSwitch()
    assert switch.trip("first") is True
    assert switch.trip("second") is False
    assert switch.reason() == "first"


# --- deferred trips (the paper runner) and reset generations -------------------------------


def test_a_deferred_trip_is_in_force_at_once_but_on_disk_only_after_flush(tmp_path):
    """The paper runner writes the file only after its step commits: a rolled-back step's trip
    must not survive it (see paper.py)."""
    switch = KillSwitch(tmp_path, deferred=True)
    assert switch.trip("daily loss", "2020-01-02") is True
    assert switch.tripped and switch.reason() == "daily loss"  # re-reading the file doesn't lose it
    assert switch.status() == {"tripped": True, "reason": "daily loss", "at": "2020-01-02 00:00:00"}
    assert switch.pending == {"reason": "daily loss", "at": "2020-01-02 00:00:00", "generation": 0}
    assert switch.trip("drawdown") is False and switch.reason() == "daily loss"
    assert not (tmp_path / "kill_switch.json").exists() and not KillSwitch(tmp_path).tripped

    assert switch.flush() is True
    assert KillSwitch(tmp_path).status() == {"tripped": True, "reason": "daily loss", "at": "2020-01-02 00:00:00"}
    assert switch.pending is None and switch.flush() is False

    failed = KillSwitch(tmp_path / "other", deferred=True)
    failed.trip("never committed")  # and the step then fails, so flush() never runs
    assert failed.tripped and not KillSwitch(tmp_path / "other").tripped


def test_a_trip_made_on_disk_meanwhile_keeps_its_reason(tmp_path):
    switch = KillSwitch(tmp_path, deferred=True)
    switch.trip("daily loss", "2020-01-02")
    KillSwitch(tmp_path).trip("manual: CLI")  # immediate, from another process
    assert switch.flush() is False
    assert KillSwitch(tmp_path).reason() == "manual: CLI"


def test_resets_count_generations_and_an_old_trip_is_not_restored(tmp_path):
    switch = KillSwitch(tmp_path)
    assert switch.generation == 0
    switch.reset(confirm=True)
    assert switch.generation == 0  # nothing was tripped: no reset to count
    switch.trip("x")
    switch.reset(confirm=True)
    assert KillSwitch(tmp_path).generation == 1

    committed = {"reason": "daily loss", "at": "2020-01-02 00:00:00", "generation": 1}
    assert switch.outstanding(committed) and switch.restore(committed) is True
    assert KillSwitch(tmp_path).status() == {"tripped": True, "reason": "daily loss", "at": "2020-01-02 00:00:00"}
    assert switch.restore(committed) is False  # already on
    switch.reset(confirm=True)  # a person has dealt with it
    assert switch.generation == 2 and not switch.outstanding(committed) and switch.restore(committed) is False
    assert not KillSwitch(tmp_path).tripped

    (tmp_path / "kill_switch.json").unlink()  # deleting the file is no reset: the trip still stands
    assert KillSwitch(tmp_path).restore({**committed, "generation": 2}) is True
    for junk in (None, {}, {"reason": "x"}, {"generation": "2"}, "trip"):
        assert not KillSwitch(tmp_path / "junk").outstanding(junk)


def test_an_older_file_has_generation_zero_and_a_bad_one_fails_closed(tmp_path):
    (tmp_path / "kill_switch.json").write_text(json.dumps({"tripped": True, "reason": "old", "at": None}))
    switch = KillSwitch(tmp_path)
    assert switch.tripped and switch.generation == 0
    switch.reset(confirm=True)
    assert switch.generation == 1
    (tmp_path / "kill_switch.json").write_text(json.dumps({"tripped": False, "generation": -1}))
    with pytest.raises(ValueError, match="generation must be an integer >= 0"):
        KillSwitch(tmp_path)
