import json

import pandas as pd
import pytest

from conftest import snapshot
from papertrader import cli as cli_module
from papertrader.cli import main as cli
from papertrader.journal import Journal


@pytest.fixture
def cfg_path(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("name: t\ndata: {synthetic: {n_days: 400, seed: 11}}\nstrategy: {name: trend}\n")
    return str(path)


def value_unmarked_position(*_):
    """What a held position with no valid mark does: PortfolioSnapshot.equity refuses to guess."""
    return snapshot(positions={"ZZZ": 10.0}).equity


def test_a_missing_mark_is_reported_as_an_error_not_a_traceback(cfg_path, monkeypatch, capsys):
    monkeypatch.setitem(cli_module.COMMANDS, "status", (value_unmarked_position, "status"))
    assert cli(["status", "-c", cfg_path]) == 1
    assert capsys.readouterr().err == "error: no valid mark for held positions: ['ZZZ']\n"


def test_backtest_closes_its_journal_when_the_run_fails(cfg_path, tmp_path, monkeypatch, capsys):
    opened = []

    class SpyJournal(Journal):
        def __init__(self, path=None):
            super().__init__(path)
            opened.append(self)

    def failing_run(data, strategy, *args, journal=None, **kwargs):
        journal.start_run("backtest", "t", strategy.describe())
        raise RuntimeError("feed died mid-run")

    monkeypatch.setattr("papertrader.journal.Journal", SpyJournal)
    monkeypatch.setattr("papertrader.engine.backtest.run_backtest", failing_run)
    out = tmp_path / "out"
    assert cli(["backtest", "-c", cfg_path, "--out", str(out)]) == 1
    assert capsys.readouterr().err == "error: feed died mid-run\n"
    [journal] = opened
    assert journal.conn is None  # closed
    reopened = Journal(out / "journal.sqlite")  # and what was logged before the failure was kept
    assert int(reopened.query("select count(*) n from runs")["n"][0]) == 1
    reopened.close()


def test_a_missing_config_file_is_an_error(tmp_path, capsys):
    assert cli(["data", "-c", str(tmp_path / "nope.yaml")]) == 1
    assert capsys.readouterr().err.startswith("error: ")


def test_malformed_yaml_a_repeated_key_or_a_directory_is_an_error_not_a_traceback(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("risk:\n  max_drawdown_pct: 0.1\n    max_position_pct: 0.2\n")  # mis-indented
    assert cli(["data", "-c", str(bad)]) == 1
    assert capsys.readouterr().err.startswith(f"error: {bad}: invalid YAML: mapping values are not allowed here")
    bad.write_text("risk: {max_drawdown_pct: 0.1}\nrisk: {max_position_pct: 0.2}\n")
    assert cli(["data", "-c", str(bad)]) == 1
    assert capsys.readouterr().err == f"error: {bad}: duplicate key 'risk' on line 2 (first on line 1)\n"
    assert cli(["data", "-c", str(tmp_path)]) == 1  # IsADirectoryError
    assert capsys.readouterr().err.startswith("error: [Errno 21]")


def paper_yaml(tmp_path, risk=""):
    path = tmp_path / "paper.yaml"
    path.write_text(
        f"name: t\ndata: {{synthetic: {{n_days: 700, seed: 11}}}}\nstrategy: {{name: equal_weight}}\n"
        f"paper: {{state_dir: {tmp_path / 'state'}}}\n{risk}"
    )
    return str(path)


def test_paper_step_prints_a_wiped_out_day_instead_of_dividing_by_zero(tmp_path, monkeypatch, capsys):
    from papertrader.engine.session import DayResult
    from papertrader.paper import PaperRunner

    days = [DayResult(pd.Timestamp("2020-01-02"), 0.0, eq, 0.0, 0.0, kill_switch=True) for eq in (0.0, float("nan"))]
    monkeypatch.setattr(PaperRunner, "step", lambda self, **kw: days)
    assert cli(["paper-step", "-c", paper_yaml(tmp_path)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("2020-01-02  equity         0.00  gross    n/a  fills 0") and out[0].endswith("[KILL SWITCH ON]")
    assert "equity          nan  gross    n/a" in out[1]


def test_paper_step_refuses_a_changed_config_until_accepted(tmp_path, capsys):
    path = paper_yaml(tmp_path)
    assert cli(["paper-step", "-c", path, "--as-of", "2016-03-01"]) == 0
    path = paper_yaml(tmp_path, risk="risk: {max_position_pct: 0.2}\n")  # tighter, still a change
    capsys.readouterr()
    assert cli(["paper-step", "-c", path, "--as-of", "2016-03-02"]) == 1
    err = capsys.readouterr().err
    assert "risk.max_position_pct: 0.25 -> 0.2" in err and "--accept-config-change" in err
    assert cli(["paper-step", "-c", path, "--as-of", "2016-03-02", "--accept-config-change"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("RISK EVENT: config_changed: risk.max_position_pct: 0.25 -> 0.2\n2016-03-02  equity")
    assert cli(["paper-step", "-c", path, "--as-of", "2016-03-03"]) == 0


def test_reset_kill_waits_for_no_step_and_says_what_it_reset(tmp_path, capsys):
    from papertrader.utils import run_lock

    path = paper_yaml(tmp_path)
    assert cli(["reset-kill", "-c", path, "--yes-i-checked"]) == 0
    assert capsys.readouterr().out == "Kill switch was not tripped; nothing to reset.\n"
    assert cli(["kill", "-c", path, "--reason", "demo"]) == 0
    capsys.readouterr()
    with run_lock(tmp_path / "state" / ".lock"):  # a paper step in progress
        assert cli(["reset-kill", "-c", path, "--yes-i-checked"]) == 1
    assert "another run may be in progress" in capsys.readouterr().err
    assert cli(["reset-kill", "-c", path, "--yes-i-checked"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Kill switch reset. It had tripped at ") and out.endswith(": manual: demo\n")
    assert json.loads((tmp_path / "state" / "kill_switch.json").read_text())["generation"] == 1
