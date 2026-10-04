import dataclasses

import numpy as np
import pytest

from conftest import make_config
from papertrader.cli import main as cli
from papertrader.data import MarketData, load_market_data
from papertrader.engine import run_backtest
from papertrader.journal import Journal
from papertrader.paper import PaperRunner
from papertrader.risk import RiskLimits
from papertrader.strategies import build_strategy


def backtest(data, cfg, name="equal_weight", **kw):
    return run_backtest(data, build_strategy(name, cfg.strategy.params if name == cfg.strategy.name else {}),
                        cfg.portfolio, cfg.costs, cfg.risk, **kw)


def test_backtest_runs_and_journals_everything(data, tmp_path):
    cfg = make_config()
    journal = Journal(tmp_path / "j.sqlite")
    res = backtest(data, cfg, "trend", start=data.dates[252], journal=journal)
    journal.close()
    assert res.equity.index[0] == data.dates[252] and res.equity.iloc[0] == pytest.approx(cfg.portfolio.initial_cash)
    assert len(res.fills) > 0 and res.summary()["costs"] > 0
    j = Journal(tmp_path / "j.sqlite")
    counts = {t: int(j.query(f"select count(*) n from {t}")["n"][0]) for t in ("decisions", "orders", "fills", "equity")}
    assert counts["equity"] == len(res.equity) and counts["fills"] == len(res.fills)
    assert counts["decisions"] > 0 and counts["orders"] >= counts["fills"]


def test_costs_only_ever_hurt(data):
    cfg = make_config()
    free = dataclasses.replace(cfg, costs=dataclasses.replace(cfg.costs, commission_bps=0, slippage_bps=0))
    paid = backtest(data, cfg, "trend", start=data.dates[252]).equity.iloc[-1]
    gratis = backtest(data, free, "trend", start=data.dates[252]).equity.iloc[-1]
    assert gratis > paid


def test_crash_trips_kill_switch_and_flattens(data):
    cfg = make_config()
    crash_day = data.dates[400]
    crashed = data.map_prices(lambda f: f.mul(np.where(f.index < crash_day, 1.0, 0.80), axis=0))  # 20% gap down
    res = backtest(crashed, cfg, "equal_weight", start=data.dates[252])
    [event] = res.risk_events
    assert event.timestamp == crash_day and "daily loss" in event.detail
    after = res.fills[res.fills["date"] > crash_day]
    assert (after["quantity"] < 0).all() and len(after) > 0  # only selling after the trip
    assert res.gross_exposure.iloc[-1] == 0  # flat
    assert res.equity.iloc[-50:].nunique() == 1  # and staying flat


def test_journal_drawdown_is_measured_from_the_risk_managers_peak(data, tmp_path):
    """The session used to keep its own peak, including an equity the daily gain limit had
    refused to believe: one bad mark left the journal showing a ~40% drawdown for good."""
    cfg = make_config()
    bad_day = data.dates[400]
    close = data.close.copy()
    close.loc[bad_day, "AAA"] *= 3.0  # one bad mark, gone the next day
    spiked = MarketData(open=data.open, high=data.high, low=data.low, close=close, volume=data.volume)
    journal = Journal(tmp_path / "j.sqlite")
    res = backtest(spiked, cfg, "equal_weight", start=data.dates[252], journal=journal)
    journal.close()
    [event] = res.risk_events
    assert event.timestamp == bad_day and event.detail.startswith("daily gain")

    j = Journal(tmp_path / "j.sqlite")
    logged = j.query("select date, equity, drawdown from equity order by date").set_index("date")
    j.close()
    believed = logged["equity"].where(logged.index != str(bad_day.date()))  # what the gain limit kept out
    peak = believed.cummax().ffill()
    np.testing.assert_allclose(logged["drawdown"], logged["equity"] / peak - 1, rtol=0, atol=1e-12)
    after = logged.loc[logged.index > str(bad_day.date()), "drawdown"]
    assert after.min() > -0.10  # no phantom drawdown from the spike's peak


def test_paper_trading_replays_the_backtest_exactly(tmp_path):
    cfg = make_config(tmp_path, strategy=dataclasses.replace(make_config().strategy, name="trend", params={"lookback": 50}))
    data, _ = load_market_data(cfg.data)
    days = data.dates[300:330]
    bt = run_backtest(data.truncate(days[-1]), build_strategy("trend", {"lookback": 50}),
                      cfg.portfolio, cfg.costs, cfg.risk, start=days[0])
    runner = PaperRunner(cfg)
    for day in days:
        runner.step(as_of=day)
    j = Journal(runner.journal_path)
    paper = j.query("select date, equity from equity order by date").set_index("date")["equity"]
    j.close()
    assert len(paper) == len(days)
    np.testing.assert_allclose(paper.to_numpy(), bt.equity.to_numpy(), rtol=0, atol=1e-6)


def test_paper_step_is_idempotent_and_limits_catch_up(tmp_path):
    cfg = make_config(tmp_path)
    data, _ = load_market_data(cfg.data)
    runner = PaperRunner(cfg)
    assert len(runner.step(as_of=data.dates[300])) == 1
    assert runner.step(as_of=data.dates[300]) == []  # same day again: nothing
    with pytest.raises(RuntimeError, match="catch up"):
        runner.step(as_of=data.dates[320])  # 20 missed days > max_catchup_days
    assert len(runner.step(as_of=data.dates[320], force=True)) == 20


def test_manual_kill_switch_via_cli_flattens_the_paper_account(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        f"name: t\ndata: {{synthetic: {{n_days: 700, seed: 11}}}}\nstrategy: {{name: equal_weight}}\n"
        f"paper: {{state_dir: {tmp_path / 'state'}}}\n"
    )
    cfg = make_config(tmp_path)
    data, _ = load_market_data(cfg.data)
    d = [str(x.date()) for x in data.dates[300:305]]
    assert cli(["paper-step", "-c", str(cfg_path), "--as-of", d[0]]) == 0
    assert cli(["paper-step", "-c", str(cfg_path), "--as-of", d[1]]) == 0  # fills the entry orders
    runner = PaperRunner(cfg)
    assert runner.status()["positions"]
    assert cli(["kill", "-c", str(cfg_path), "--reason", "testing"]) == 0
    assert cli(["paper-step", "-c", str(cfg_path), "--as-of", d[2]]) == 0  # queues flatten orders
    assert cli(["paper-step", "-c", str(cfg_path), "--as-of", d[3]]) == 0  # fills them
    status = runner.status()
    assert status["positions"] == {} and status["kill_switch"]["tripped"]
    assert cli(["reset-kill", "-c", str(cfg_path)]) == 1  # refuses without --yes-i-checked
    assert cli(["reset-kill", "-c", str(cfg_path), "--yes-i-checked"]) == 0
    assert not runner.status()["kill_switch"]["tripped"]


def test_paper_refuses_to_mix_strategies_in_one_state_dir(tmp_path):
    cfg = make_config(tmp_path)
    data, _ = load_market_data(cfg.data)
    PaperRunner(cfg).step(as_of=data.dates[300])
    other = dataclasses.replace(cfg, strategy=dataclasses.replace(cfg.strategy, params={"lookback": 20}))
    with pytest.raises(RuntimeError, match="belongs to strategy"):
        PaperRunner(other).step(as_of=data.dates[301])


def test_risk_limits_reject_bad_config():
    with pytest.raises(ValueError):
        RiskLimits(max_drawdown_pct=25)  # someone meant 25%
    with pytest.raises(ValueError):
        RiskLimits(max_position_pct=0)
