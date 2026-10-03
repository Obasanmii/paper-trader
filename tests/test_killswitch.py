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
