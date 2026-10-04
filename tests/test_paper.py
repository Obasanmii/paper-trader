"""The paper runner: catch-up equals on-time, crashes leave nothing half-done, adjusted
histories re-anchor the account, config drift is refused, and today's unfinished bar waits."""
import contextlib
import dataclasses
import json
import math
import re
import shutil
import sqlite3
import threading

import numpy as np
import pandas as pd
import pytest

import papertrader.paper as paper_module
from conftest import make_config
from papertrader.config import PaperConfig, StrategyConfig
from papertrader.core import Order
from papertrader.data import DataSource, SyntheticSource, load_market_data
from papertrader.engine import run_backtest
from papertrader.engine.session import TradingSession
from papertrader.journal import Journal
from papertrader.paper import PaperRunner, _rescale
from papertrader.risk import KillSwitch
from papertrader.strategies import build_strategy
from papertrader.utils import run_lock

PRICES = ["open", "high", "low", "close"]
SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE"]


class FrameSource(DataSource):
    """Fixed bars served as a vendor would serve them on `today`: nothing after it, and
    adjusted. `actions` are (symbol, ex_date, factor): once the ex-date has passed, every
    earlier price is scaled by factor (0.5 for a 2:1 split, 0.98 for a 2% dividend).
    `revisions` are (symbol, seen_from, date, factor): from seen_from on, that one bar is
    scaled, i.e. history revised rather than rescaled."""

    def __init__(self, frames, actions=(), revisions=()):
        self.frames, self.actions, self.revisions, self.today = frames, list(actions), list(revisions), None

    def load(self, symbols, start=None, end=None):
        out = {}
        for sym in symbols:
            df = self.frames[sym].loc[: self.today].copy()
            for s, ex, factor in self.actions:
                if s == sym and ex <= self.today:
                    df.loc[df.index < ex, PRICES] *= factor
            for s, seen_from, day, factor in self.revisions:
                if s == sym and seen_from <= self.today:
                    df.loc[day, PRICES] *= factor
            out[sym] = df
        return out


def traded_after(frames, sym, ex, factor):
    """The prices actually printed: from the ex-date on, the stock trades at the new scale."""
    frames = dict(frames)
    df = frames[sym].copy()
    df.loc[df.index >= ex, PRICES] *= factor
    frames[sym] = df
    return frames


def paper_config(tmp_path, n_days=400, fractional=False, **synthetic):
    cfg = make_config(tmp_path, n_days=n_days, strategy=StrategyConfig(name="equal_weight"))
    data = dataclasses.replace(cfg.data, synthetic=dataclasses.replace(cfg.data.synthetic, **synthetic))
    portfolio = dataclasses.replace(cfg.portfolio, allow_fractional=fractional)
    return dataclasses.replace(cfg, data=data, portfolio=portfolio)


def walk(runner, days, source=None, **step):
    """Run the account through `days` (each a step, as if run that evening)."""
    out = []
    for day in days:
        if source is not None:
            source.today = day
        out += runner.step(as_of=day, **step)
    return out


def fall_from(frames, day, factor=0.9):
    """Every symbol opens normally on `day`, closes `factor` lower and stays there: a real loss
    past the daily limit, too small to be held back as unconfirmed."""
    def fall(df):
        df = df.copy()
        df.loc[df.index >= day, ["high", "low", "close"]] *= factor
        df.loc[df.index > day, "open"] *= factor
        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"] = df[["open", "low", "close"]].min(axis=1)
        return df

    return {s: fall(df) for s, df in frames.items()}


def lock_is_free(runner) -> bool:
    try:
        with run_lock(runner.state_dir / ".lock"):
            return True
    except RuntimeError:
        return False


def journal(runner, sql):
    j = Journal(runner.journal_path)
    try:
        return j.query(sql)
    finally:
        j.close()


def equity(runner) -> pd.Series:
    return journal(runner, "select date, equity from equity order by date").set_index("date")["equity"]


def journal_tables(runner) -> dict:
    """Everything a run journaled, minus the random ids: two identical runs compare equal."""
    return {
        "equity": journal(runner, "select date, cash, equity, gross_exposure, net_exposure, drawdown from equity order by date"),
        "fills": journal(runner, "select date, symbol, quantity, price, commission from fills order by date, symbol"),
        "orders": journal(
            runner,
            "select date, event, symbol, quantity, reference_price, reason, detail from orders "
            "order by date, symbol, event, quantity",
        ),
        "decisions": journal(runner, "select date, symbol, target_weight from decisions order by date, symbol"),
        "risk_events": journal(runner, "select date, kind, detail from risk_events order by date, kind"),
    }


def assert_same_journal(a, b):
    ta, tb = journal_tables(a), journal_tables(b)
    for table in ta:
        pd.testing.assert_frame_equal(ta[table], tb[table], obj=table)


# --- 1. catch-up == on-time -----------------------------------------------------------


def test_catch_up_equals_on_time_on_dirty_data(tmp_path):
    """A catch-up used to clean every missed day with bars from after it, so it ended up
    elsewhere than on-time runs (170k of equity apart on the same date)."""
    def cfg(name):
        c = paper_config(tmp_path / name, n_days=1000, seed=3, inject_errors=True)
        return dataclasses.replace(c, paper=PaperConfig(state_dir=str(tmp_path / name / "state"), max_catchup_days=100))

    data, _ = load_market_data(cfg("x").data)
    spike = pd.Timestamp("2015-07-31")  # the planted 10x AAA print
    i = data.dates.get_loc(spike)
    days = data.dates[i - 12 : i + 8]

    on_time = PaperRunner(cfg("on_time"))
    walk(on_time, days)
    catch_up = PaperRunner(cfg("catch_up"))
    catch_up.step(as_of=days[0])
    assert len(catch_up.step(as_of=days[-1], force=True)) == len(days) - 1

    curve = equity(on_time)
    assert list(curve.index) == [str(d.date()) for d in days]
    pd.testing.assert_series_equal(curve, equity(catch_up), rtol=0, atol=0)
    # On time, the bad tick was the newest bar on its day: held back, not marked and not traded on.
    assert curve.pct_change().abs().max() < 0.05
    assert not on_time.status()["kill_switch"]["tripped"]
    assert journal(on_time, "select count(*) n from risk_events")["n"][0] == 0


def test_a_day_with_every_bar_held_back_is_realised_the_next_day(tmp_path):
    """A real 30% market-wide drop can't be told from bad ticks on its own day, so every
    symbol's bar waits and the day isn't processed. The next day confirms it, and both an
    on-time run and a catch-up book it then, the same way."""
    base = SyntheticSource(n_days=400, seed=11).load(SYMBOLS)
    days = base["AAA"].index[300:310]
    drop = days[5]
    crashed = {s: df.assign(**{c: df[c].where(df.index < drop, df[c] * 0.7) for c in PRICES}) for s, df in base.items()}
    source = FrameSource(crashed)
    on_time = PaperRunner(paper_config(tmp_path / "on_time"), source=source)
    processed = [r.date for r in walk(on_time, days, source)]
    assert drop not in processed and days[6] in processed  # nothing on the day itself; realised next day
    catch_up = PaperRunner(paper_config(tmp_path / "catch_up"), source=source)
    walk(catch_up, days[:3], source)
    walk(catch_up, [days[-1]], source, force=True)
    assert_same_journal(on_time, catch_up)
    events = journal(on_time, "select date, kind, detail from risk_events")
    assert list(events["date"]) == [str(days[6].date())] and events["detail"][0].startswith("daily loss")


# --- 2. crash consistency ------------------------------------------------------------


def fail_once(monkeypatch, target, name, when=lambda *a, **k: True):
    """Make target.name raise the first time `when` holds, then behave normally."""
    real = getattr(target, name)

    def flaky(*args, **kwargs):
        if when(*args, **kwargs):
            monkeypatch.setattr(target, name, real)
            raise OSError("simulated crash")
        return real(*args, **kwargs)

    monkeypatch.setattr(target, name, flaky)


@pytest.mark.parametrize("crash", ["before_commit", "after_commit", "mid_catch_up"])
def test_a_crash_and_retry_journal_every_day_exactly_once(tmp_path, monkeypatch, crash):
    data, _ = load_market_data(paper_config(tmp_path).data)
    days = data.dates[300:312]
    clean = PaperRunner(paper_config(tmp_path / "clean"))
    walk(clean, days)

    runner = PaperRunner(paper_config(tmp_path / "crash"))
    walk(runner, days[:5])
    if crash == "before_commit":  # the day's rows are written, the commit never happens
        fail_once(monkeypatch, Journal, "commit")
    elif crash == "after_commit":  # committed, then died writing the state.json copy
        fail_once(monkeypatch, paper_module, "atomic_write_json")
    else:  # the second of three catch-up days blows up
        fail_once(monkeypatch, TradingSession, "process_day", when=lambda self, date, now=None: date == days[6])
    target = days[7] if crash == "mid_catch_up" else days[5]
    with pytest.raises(OSError, match="simulated crash"):
        runner.step(as_of=target)
    assert lock_is_free(runner)
    retried = runner.step(as_of=target)
    assert [r.date for r in retried] == ([] if crash == "after_commit" else list(days[5 : list(days).index(target) + 1]))
    walk(runner, days[days > target])

    assert_same_journal(runner, clean)
    assert json.loads(runner.state_path.read_text()) == runner.load_state()  # the copy caught up too


# --- 2b. the kill switch is part of the step's transaction -------------------------------


def loss_account(tmp_path, name, drop_index=3):
    """A whole-share account over 12 days, every symbol closing 10% lower on days[drop_index]."""
    base = SyntheticSource(n_days=400, seed=11).load(SYMBOLS)
    days = base["AAA"].index[300:312]
    source = FrameSource(fall_from(base, days[drop_index]))
    return PaperRunner(paper_config(tmp_path / name), source=source), source, days


@pytest.mark.parametrize("catch_up", [False, True], ids=["on_time", "catch_up"])
def test_a_kill_switch_trip_rolls_back_with_its_step(tmp_path, monkeypatch, catch_up):
    """The switch used to persist the moment it tripped, outside the transaction. After a crash
    the retry found it on from the first replayed open: on time it cancelled the day's entries
    (and journaled a -10% loss on a flat day); in a catch-up it flattened the account at the
    close of the first missed day, on a loss two days later, hiding it."""
    drop_index = 8 if catch_up else 1
    runners = {}
    for name in ("clean", "crash"):
        runner, source, days = loss_account(tmp_path, name, drop_index)
        start, target = (days[:6], days[9]) if catch_up else (days[:1], days[1])
        walk(runner, start, source)
        source.today = target
        if name == "crash":
            fail_once(monkeypatch, Journal, "commit")
            with pytest.raises(OSError, match="simulated crash"):
                runner.step(as_of=target)
            assert not runner.status()["kill_switch"]["tripped"]  # rolled back with everything else
            assert not runner.kill_switch().tripped
        runner.step(as_of=target)
        walk(runner, days[days > target], source)
        runners[name] = runner

    assert_same_journal(runners["crash"], runners["clean"])
    events = journal(runners["crash"], "select date, kind, detail from risk_events")
    assert list(events["kind"]) == ["kill_switch_tripped"] and events["date"][0] == str(days[drop_index].date())
    assert events["detail"][0].startswith("daily loss -")
    assert runners["crash"].kill_switch().tripped


def test_a_trip_committed_but_not_written_is_applied_by_the_next_step(tmp_path, monkeypatch):
    """The trip lands in the account's committed state with the day; kill_switch.json is written
    after the commit. If the run dies in between, the next step writes it before its first open."""
    clean, source, days = loss_account(tmp_path, "clean")
    walk(clean, days, source)

    runner, source, days = loss_account(tmp_path, "crash")
    walk(runner, days[:3], source)
    fail_once(monkeypatch, KillSwitch, "flush", when=lambda self: self.pending is not None)
    source.today = days[3]
    with pytest.raises(OSError, match="simulated crash"):
        runner.step(as_of=days[3])
    assert not runner.kill_switch().tripped  # the file never got it...
    ks = runner.status()["kill_switch"]  # ...but the committed account did, and status says so
    assert ks["tripped"] and ks["reason"].startswith("daily loss -") and ks["at"].startswith(str(days[3].date()))
    assert runner.load_state()["kill_switch"]["last_auto_trip"]["generation"] == 0

    assert runner.step(as_of=days[3]) == []  # nothing to redo, but the switch is on again
    assert runner.kill_switch().tripped and runner.kill_switch().reason() == ks["reason"]
    walk(runner, days[4:], source)
    assert_same_journal(runner, clean)


@pytest.mark.parametrize("written", [True, False], ids=["written", "died_before_writing"])
def test_a_reset_after_a_committed_trip_is_respected(tmp_path, monkeypatch, written):
    """A reset bumps the switch's generation, so a committed trip from before it is history,
    not something to re-apply. reset-kill also covers a trip that never reached the file."""
    runner, source, days = loss_account(tmp_path, "x")
    walk(runner, days[:3], source)
    source.today = days[3]
    if written:
        runner.step(as_of=days[3])
    else:
        fail_once(monkeypatch, KillSwitch, "flush", when=lambda self: self.pending is not None)
        with pytest.raises(OSError, match="simulated crash"):
            runner.step(as_of=days[3])
    before = runner.reset_kill_switch()
    assert before["tripped"] and before["reason"].startswith("daily loss -")
    assert not runner.status()["kill_switch"]["tripped"] and runner.kill_switch().generation == 1

    results = walk(runner, days[4:7], source)
    assert not any(r.kill_switch for r in results)
    assert list(journal(runner, "select kind from risk_events")["kind"]) == ["kill_switch_tripped"]
    assert not runner.kill_switch().tripped


def test_a_kill_from_outside_is_journaled_once_by_the_step_that_acts_on_it(tmp_path):
    """The step cancels and flattens because of a trip it didn't make (the kill command, the KILL
    file); the journal used to have no row saying why."""
    cfg = paper_config(tmp_path)
    days = load_market_data(cfg.data)[0].dates[300:310]
    runner = PaperRunner(cfg)
    walk(runner, days[:3])
    assert runner.kill_switch().trip("manual: demo")
    walk(runner, days[3:6])
    assert runner.status()["positions"] == {}  # flattened
    assert list(journal(runner, "select kind from risk_events")["kind"]) == ["kill_switch_observed"]
    runner.reset_kill_switch()
    walk(runner, days[6:7])  # back in the market, entries queued for tomorrow's open
    (runner.state_dir / "KILL").touch()
    walk(runner, days[7:10])

    events = journal(runner, "select date, kind, detail from risk_events")
    sentinel = f"kill switch is on: sentinel file present: {runner.state_dir / 'KILL'}"
    assert list(events.itertuples(index=False)) == [
        (str(days[3].date()), "kill_switch_observed", "kill switch is on: manual: demo"),
        (str(days[7].date()), "kill_switch_observed", sentinel),
    ]
    cancelled = journal(runner, "select date, detail from orders where event = 'cancelled'")
    assert len(cancelled) == 5 and set(cancelled.itertuples(index=False)) == {
        (str(days[7].date()), "kill switch active at the open")
    }


@pytest.mark.parametrize("seen", ["same_evening", "next_day"])
def test_a_revision_trip_in_a_rolled_back_step_is_journaled_once_as_a_trip(tmp_path, monkeypatch, seen):
    """A held symbol revised inconsistently trips the switch while re-anchoring. After a crash the
    retry used to find the switch already on and journal 'history_revised' instead (or both)."""
    base = SyntheticSource(n_days=400, seed=11).load(SYMBOLS)
    days = base["AAA"].index[300:310]
    d = list(base["AAA"].index)
    i6 = d.index(days[6])
    on = days[6] if seen == "same_evening" else days[7]
    runners = {}
    for name in ("clean", "crash"):
        source = FrameSource(base)
        runners[name] = runner = PaperRunner(paper_config(tmp_path / name), source=source)
        walk(runner, days[:7], source)
        source.revisions = [("AAA", on, d[i6 - 2], 0.9), ("AAA", on, d[i6 - 1], 0.95)]
        source.today = on
        if name == "crash":
            fail_once(monkeypatch, Journal, "commit")
            with pytest.raises(OSError, match="simulated crash"):
                runner.step(as_of=on)
            assert not runner.kill_switch().tripped
        runner.step(as_of=on)
        assert [e.kind for e in runner.step_events] == ["kill_switch_tripped"]
        walk(runner, days[days > on], source)
    assert_same_journal(runners["crash"], runners["clean"])
    events = journal(runners["crash"], "select date, kind, detail from risk_events")
    assert list(events["kind"]) == ["kill_switch_tripped"] and "can't re-anchor the held position" in events["detail"][0]
    assert "AAA" not in runners["crash"].status()["positions"]


# --- 2c. a real jump past the gain limit -----------------------------------------------


def test_a_real_jump_becomes_the_base_once_a_person_resets_the_switch(tmp_path):
    """README: 'If a person checks it and resets the switch, the next close accepts the new level
    as the base.' It didn't: the gain limit kept the jump out of the base, so every later close
    repeated the 'gain' and re-tripped the switch, however often it was reset."""
    base = SyntheticSource(n_days=400, seed=11).load(SYMBOLS)
    days = base["AAA"].index[300:312]
    aaa = base["AAA"].copy()
    aaa.loc[aaa.index >= days[4], PRICES] *= 3.0  # a takeover: real, and it stays
    source = FrameSource({**base, "AAA": aaa})
    runner = PaperRunner(paper_config(tmp_path), source=source)
    events = [e for r in walk(runner, days[:8], source) for e in r.events]
    assert [e.kind for e in events] == ["kill_switch_tripped"] and events[0].detail.startswith("daily gain +")
    held_back = runner.load_state()["risk"]  # three steps after the jump, still not the base

    runner.kill_switch().reset(confirm=True)  # a person checked: a takeover, not a bad print
    [day] = walk(runner, days[8:9], source)
    assert [e.kind for e in day.events] == ["gain_accepted"] and not day.kill_switch
    assert held_back["gain_unconfirmed"] is True and held_back["last_close_equity"] < 101_000
    risk = runner.load_state()["risk"]
    assert risk["gain_unconfirmed"] is False and risk["last_close_equity"] == risk["peak_equity"] == day.equity > 130_000
    assert not any(r.kill_switch or r.events for r in walk(runner, days[9:], source))
    assert list(journal(runner, "select kind from risk_events")["kind"]) == ["kill_switch_tripped", "gain_accepted"]


# --- 3. corporate actions in adjusted data --------------------------------------------


@pytest.mark.parametrize("ex_index, catch_up", [(6, False), (1, False), (6, True)], ids=["held", "queued", "catch_up"])
def test_split_in_adjusted_data_reanchors_the_account(tmp_path, ex_index, catch_up):
    """A 2:1 split in an adjusted feed halves all earlier prices. It used to read as a 50%
    loss and trip the kill switch; now shares double and equity doesn't move."""
    base = SyntheticSource(n_days=400, seed=11).load(SYMBOLS)
    days = base["AAA"].index[300:312]
    ex = days[ex_index]
    split = FrameSource(traded_after(base, "AAA", ex, 0.5), actions=[("AAA", ex, 0.5)])
    plain = FrameSource(base)  # the same economics, no split
    runners = {}
    for name, source in (("split", split), ("plain", plain)):
        runners[name] = runner = PaperRunner(paper_config(tmp_path / name), source=source)
        if catch_up:
            walk(runner, [days[0], days[-1]], source, force=True)
        else:
            walk(runner, days, source)
    runner = runners["split"]

    pd.testing.assert_series_equal(equity(runner), equity(runners["plain"]), rtol=1e-12)
    status = runner.status()
    assert not status["kill_switch"]["tripped"]
    assert status["positions"]["AAA"]["quantity"] == 2 * runners["plain"].status()["positions"]["AAA"]["quantity"]
    [event] = journal(runner, "select date, kind, detail from risk_events").itertuples()
    queued = ex_index == 1 or catch_up  # re-anchored before the entry order filled: the order doubles
    held = 0 if queued else int(runners["plain"].status()["positions"]["AAA"]["quantity"])
    # Stamped on the first day it applies to: the ex-date, or the first day a catch-up processes.
    assert event.kind == "corporate_action" and event.date == str((days[1] if catch_up else ex).date())
    assert event.detail.startswith("AAA: price history rescaled by 0.5 ") and f"position {held} -> {2 * held}," in event.detail
    assert ("1 queued order(s)" in event.detail) == queued


def test_dividend_in_adjusted_data_is_reinvested_not_lost(tmp_path):
    """A 2% dividend scales earlier adjusted prices by 0.98. Re-anchoring keeps it in
    equity (total return), matching a backtest on the final adjusted data."""
    base = SyntheticSource(n_days=400, seed=11).load(SYMBOLS)
    days = base["AAA"].index[300:312]
    ex = days[6]
    traded = traded_after(base, "AAA", ex, 0.98)
    adjusted = FrameSource(traded, actions=[("AAA", ex, 0.98)])
    cfg = paper_config(tmp_path / "adjusted", fractional=True)
    runner = PaperRunner(cfg, source=adjusted)
    walk(runner, days, adjusted)

    final, _ = load_market_data(cfg.data, source=adjusted)  # the adjusted history as of the last day
    bt = run_backtest(final, build_strategy("equal_weight"), cfg.portfolio, cfg.costs, cfg.risk, start=days[0])
    np.testing.assert_allclose(equity(runner).to_numpy(), bt.equity.to_numpy(), rtol=1e-9)

    # The old behaviour, for scale: with nothing re-anchored the 2% drop at the ex-date is just a loss.
    unadjusted = FrameSource(traded)
    lost = PaperRunner(paper_config(tmp_path / "unadjusted", fractional=True), source=unadjusted)
    walk(lost, days, unadjusted)
    aaa = lost.status()["positions"]["AAA"]["value"]
    dividend_now = equity(runner).iloc[-1] - equity(lost).iloc[-1]
    assert dividend_now == pytest.approx(aaa * 0.02 / 0.98, rel=1e-9)
    [kind] = journal(runner, "select kind from risk_events")["kind"]
    assert kind == "corporate_action" and not runner.status()["kill_switch"]["tripped"]


def test_inconsistent_history_revision_fails_closed_for_held_symbols(tmp_path):
    base = SyntheticSource(n_days=400, seed=11).load(SYMBOLS)
    days = base["AAA"].index[300:312]
    d = list(base["AAA"].index)
    i1, i6 = d.index(days[1]), d.index(days[6])
    source = FrameSource(
        base,
        revisions=[
            ("BBB", days[1], d[i1 - 3], 0.9),  # nothing held yet on day 1: logged, nothing tripped
            ("BBB", days[1], d[i1 - 2], 0.95),
            ("AAA", days[6], d[i6 - 3], 0.9),  # held by day 6: can't re-anchor, so trip
            ("AAA", days[6], d[i6 - 2], 0.95),
        ],
    )
    runner = PaperRunner(paper_config(tmp_path), source=source)
    walk(runner, days[:6], source)
    assert not runner.status()["kill_switch"]["tripped"]
    held = runner.status()["positions"]["AAA"]["quantity"]
    walk(runner, days[6:], source)

    events = journal(runner, "select date, kind, detail from risk_events order by date")
    assert list(events["kind"]) == ["history_revised", "kill_switch_tripped"]
    assert list(events["date"]) == [str(days[1].date()), str(days[6].date())]
    assert events["detail"][0].startswith("BBB: stored closes changed by different factors")
    assert events["detail"][1].startswith("AAA: ") and "can't re-anchor the held position" in events["detail"][1]
    status = runner.status()
    assert status["kill_switch"]["tripped"] and status["kill_switch"]["reason"] == events["detail"][1]
    assert held > 0 and "AAA" not in status["positions"]  # not rescaled; flattened by the kill switch


@pytest.mark.parametrize("halt_days", [1, 2])
def test_a_halt_after_a_held_back_jump_is_not_a_revision(tmp_path, halt_days):
    """A real +40% jump on the newest bar is held back as unconfirmed; with no bar the next
    day(s) it is still the newest bar and held back again, so every stored close for those days
    was the carried-forward pre-jump mark. Re-anchoring compared those with the real closes:
    a one-day halt tripped the kill switch, a two-day halt rescaled the position by 1/1.4 on
    time but not in a catch-up (7% of equity apart)."""
    base = SyntheticSource(n_days=400, seed=11).load(SYMBOLS)
    days = base["AAA"].index[300:312]
    aaa = base["AAA"].copy()
    aaa.loc[aaa.index >= days[5], PRICES] *= 1.4  # news: it stays
    frames = {**base, "AAA": aaa.drop(index=days[6 : 6 + halt_days])}  # then trading is halted
    on_time = PaperRunner(paper_config(tmp_path / "on_time"), source=FrameSource(frames))
    walk(on_time, days, on_time.source)
    catch_up = PaperRunner(paper_config(tmp_path / "catch_up"), source=FrameSource(frames))
    walk(catch_up, [days[0], days[-1]], catch_up.source, force=True)

    assert_same_journal(on_time, catch_up)
    assert journal(on_time, "select count(*) n from risk_events")["n"][0] == 0
    assert not on_time.status()["kill_switch"]["tripped"] and "AAA" in on_time.status()["positions"]


@pytest.mark.parametrize("catch_up", [False, True], ids=["on_time", "catch_up"])
def test_a_split_right_after_a_halt_is_still_reanchored(tmp_path, catch_up):
    """Each symbol keeps its own last traded bars. Keeping the account's last three dates
    instead left a symbol halted on the older two with nothing to compare, so the split that
    followed read as a 50% loss and tripped the kill switch."""
    base = SyntheticSource(n_days=400, seed=11).load(SYMBOLS)
    days = base["AAA"].index[300:312]
    halted = {**base, "AAA": base["AAA"].drop(index=days[5:7])}  # no AAA bars for two days
    ex = days[8]  # AAA trades once more, then splits 2:1
    split = FrameSource(traded_after(halted, "AAA", ex, 0.5), actions=[("AAA", ex, 0.5)])
    plain = FrameSource(halted)
    runners = {}
    for name, source in (("split", split), ("plain", plain)):
        runners[name] = runner = PaperRunner(paper_config(tmp_path / name), source=source)
        walk(runner, [days[0], days[-1]] if catch_up else days, source, force=True)
    runner = runners["split"]

    pd.testing.assert_series_equal(equity(runner), equity(runners["plain"]), rtol=1e-12)
    assert not runner.status()["kill_switch"]["tripped"]
    kinds = list(journal(runner, "select kind from risk_events")["kind"])
    assert kinds == ["corporate_action"]


@pytest.mark.parametrize("how", ["reset_kill", "reset_directly"])
def test_a_reset_trip_is_never_re_applied_even_if_the_switch_file_is_lost(tmp_path, how):
    """The account keeps a record of its last automatic trip so a crash between commit and
    writing kill_switch.json can't lose it. Once a person has reset that trip, the record must
    not bring it back, e.g. when kill_switch.json is deleted or the state is copied elsewhere."""
    runner, source, days = loss_account(tmp_path, "x")
    walk(runner, days[:4], source)
    assert runner.kill_switch().tripped
    if how == "reset_kill":
        runner.reset_kill_switch()  # drops the record straight away
    else:
        KillSwitch(runner.state_dir).reset(confirm=True)  # the next step sees the generation move on
        walk(runner, days[4:5], source)
    assert runner.load_state()["kill_switch"]["last_auto_trip"] is None
    (runner.state_dir / "kill_switch.json").unlink()

    results = walk(runner, days[5:8], source)
    assert not any(r.kill_switch for r in results) and not runner.kill_switch().tripped
    assert list(journal(runner, "select kind from risk_events")["kind"]) == ["kill_switch_tripped"]


@pytest.mark.parametrize("glitch", ["nan_close", "late_bar", "two_bars_missing"])
def test_a_missing_or_late_bar_is_not_a_revision(tmp_path, glitch):
    """A forward-filled close is the last good mark carried over a gap, not the vendor's print.
    Compared as one, a transient NaN or a bar published a day late read as a revised history of
    a held symbol: the kill switch tripped and the account was liquidated."""
    base = SyntheticSource(n_days=400, seed=11).load(SYMBOLS)
    days = base["AAA"].index[300:312]

    class Glitchy(FrameSource):
        def load(self, symbols, start=None, end=None):
            out = super().load(symbols, start, end)
            aaa = out["AAA"].copy()
            if glitch == "nan_close" and self.today == days[7]:
                aaa.loc[days[5], "close"] = np.nan  # one download, two days back
            if glitch == "late_bar" and self.today < days[7]:
                aaa = aaa.drop(index=days[5], errors="ignore")  # published two runs late
            if glitch == "two_bars_missing" and self.today == days[7]:
                aaa = aaa.drop(index=[days[4], days[5]])  # every compared date missing once
            out["AAA"] = aaa
            return out

    runner = PaperRunner(paper_config(tmp_path), source=Glitchy(base))
    walk(runner, days, runner.source)
    assert journal(runner, "select count(*) n from risk_events")["n"][0] == 0
    status = runner.status()
    assert not status["kill_switch"]["tripped"] and set(status["positions"]) == set(SYMBOLS)


def whole(q) -> bool:
    return float(q) == int(q)


@pytest.mark.parametrize(
    "sym, factor", [("AAA", 0.98), ("AAA", 0.5), ("EEE", 2 / 3)], ids=["dividend_2pct", "split_2_for_1", "split_3_for_2"]
)
def test_whole_share_reanchoring_pays_the_fraction_out_in_cash(tmp_path, sym, factor):
    """allow_fractional: false used to end up holding 153.0612245 shares after a 2% dividend,
    and trading fractions from then on. Now the position is rounded toward zero and the rest
    paid out at the new-scale mark: equity unchanged, to the cent and beyond."""
    base = SyntheticSource(n_days=400, seed=11).load(SYMBOLS)
    days = base["AAA"].index[300:312]
    ex = days[6]
    source = FrameSource(traded_after(base, sym, ex, factor), actions=[(sym, ex, factor)])
    runner = PaperRunner(paper_config(tmp_path), source=source)  # whole shares
    walk(runner, days[:6], source)
    before = runner.status()
    source.today = ex  # the vendor has adjusted its history; re-run the evening before the ex-date
    assert runner.step(as_of=days[5]) == []
    after = runner.status()

    held, mark = before["positions"][sym]["quantity"], before["positions"][sym]["mark"]
    shares = held / factor
    kept = round(shares) if abs(shares - round(shares)) < 1e-6 else math.trunc(shares)
    in_lieu = (shares - kept) * (mark * factor)  # at the new-scale mark
    assert after["positions"][sym]["quantity"] == kept
    assert after["cash"] - before["cash"] == pytest.approx(in_lieu, rel=1e-9, abs=1e-9)
    assert after["equity"] == pytest.approx(before["equity"], rel=1e-12)
    [event] = runner.step_events
    assert event.kind == "corporate_action" and event.detail.startswith(f"{sym}: price history rescaled by {factor:.8g} ")
    assert (in_lieu != 0) == (factor != 0.5)  # a 2:1 split of whole shares is whole already
    assert (f"cash {in_lieu:+.10g} in lieu" in event.detail) == (in_lieu != 0)

    walk(runner, days[6:], source)
    fills = journal(runner, "select quantity from fills")["quantity"]
    assert len(fills) and all(whole(q) for q in fills)
    assert all(whole(p["quantity"]) for p in runner.status()["positions"].values())


def test_rescale_rounds_queued_orders_toward_zero_and_drops_empty_ones():
    """A 1-for-3 reverse split (prices x3) in a whole-share account."""
    def order(qty):
        return Order("AAA", qty, 10.0, pd.Timestamp("2020-01-02")).to_dict()

    state = {
        "broker": {"cash": 1000.0, "positions": {"AAA": 100.0, "BBB": 7.0}, "pending": [order(2.0), order(-7.0), order(9.0)]},
        "marks": {"AAA": 10.0, "BBB": 50.0},
    }
    event = _rescale(state, "AAA", 3.0, "2020-01-03", fractional=False)
    b = state["broker"]
    assert b["positions"] == {"AAA": 33.0, "BBB": 7.0} and state["marks"]["AAA"] == 30.0
    assert b["cash"] == pytest.approx(1000.0 + (100 / 3 - 33) * 30.0, rel=1e-12)  # 10.00 in lieu
    assert b["cash"] + 33 * 30.0 == pytest.approx(1000.0 + 100 * 10.0, rel=1e-12)  # equity unchanged
    assert [(o["quantity"], o["reference_price"]) for o in b["pending"]] == [(-2.0, 30.0), (3.0, 30.0)]
    assert "position 100 -> 33 (0.333333 share(s) paid out at 30: cash +10 in lieu)" in event.detail
    assert "3 queued order(s) re-anchored (1 rounded to 0 shares and dropped)" in event.detail

    fractional = {"broker": {"cash": 0.0, "positions": {"AAA": 100.0}, "pending": [order(2.0)]}, "marks": {"AAA": 10.0}}
    _rescale(fractional, "AAA", 3.0, "2020-01-03", fractional=True)
    assert fractional["broker"]["positions"]["AAA"] == pytest.approx(100 / 3) and fractional["broker"]["cash"] == 0.0
    assert fractional["broker"]["pending"][0]["quantity"] == pytest.approx(2 / 3)


# --- 4. config drift --------------------------------------------------------------------


def test_config_drift_is_refused_until_accepted_and_then_journaled(tmp_path):
    cfg = make_config(tmp_path)
    data, _ = load_market_data(cfg.data)
    d = data.dates[300:305]
    PaperRunner(cfg).step(as_of=d[0])
    looser = dataclasses.replace(
        cfg,
        risk=dataclasses.replace(cfg.risk, max_position_pct=0.5),
        costs=dataclasses.replace(cfg.costs, slippage_bps=1.0),
    )
    with pytest.raises(RuntimeError) as refused:
        PaperRunner(looser).step(as_of=d[1])
    assert "costs.slippage_bps: 5.0 -> 1.0\n  risk.max_position_pct: 0.25 -> 0.5\n" in str(refused.value)
    assert PaperRunner(cfg).load_state()["last_processed"] == str(d[0].date())  # nothing happened

    runner = PaperRunner(looser)
    [day] = runner.step(as_of=d[1], accept_config_change=True)
    [event] = runner.step_events
    assert (event.kind, event.detail) == ("config_changed", "costs.slippage_bps: 5.0 -> 1.0; risk.max_position_pct: 0.25 -> 0.5")
    logged = journal(runner, "select date, kind, detail from risk_events")
    assert list(logged.itertuples(index=False)) == [(str(d[1].date()), event.kind, event.detail)]
    assert len(runner.step(as_of=d[2])) == 1  # adopted: no flag needed any more
    with pytest.raises(RuntimeError, match=r"risk\.max_position_pct: 0\.5 -> 0\.25"):
        PaperRunner(cfg).step(as_of=d[3])  # and going back is a change too


def test_strategy_check_still_comes_first(tmp_path):
    cfg = make_config(tmp_path)
    data, _ = load_market_data(cfg.data)
    PaperRunner(cfg).step(as_of=data.dates[300])
    other = dataclasses.replace(
        cfg, strategy=StrategyConfig(name="equal_weight"), risk=dataclasses.replace(cfg.risk, allow_short=True)
    )
    with pytest.raises(RuntimeError, match="belongs to strategy"):
        PaperRunner(other).step(as_of=data.dates[301], accept_config_change=True)


def test_older_state_is_migrated_and_its_config_recorded_silently(tmp_path):
    """State used to live only in state.json, without config or close history and with
    the session's own peak_equity. It carries on exactly where it left off."""
    cfg = make_config(tmp_path)
    data, _ = load_market_data(cfg.data)
    days = data.dates[300:305]
    clean = PaperRunner(make_config(tmp_path / "clean"))
    walk(clean, days)

    runner = PaperRunner(cfg)
    walk(runner, days[:2])
    old = runner.load_state()
    assert "peak_equity" not in old and set(old["config"]) == {"data", "strategy", "portfolio", "costs", "risk"}
    for key in ("config", "close_history"):
        del old[key]
    (runner.state_dir / "state.json").write_text(json.dumps({**old, "peak_equity": 1.0}))
    conn = sqlite3.connect(runner.journal_path)
    with conn:
        conn.execute("DELETE FROM paper_state")
    conn.close()

    other_costs = dataclasses.replace(cfg, costs=dataclasses.replace(cfg.costs, commission_bps=1.0))  # same values
    walk(PaperRunner(other_costs), days[2:])
    assert_same_journal(runner, clean)  # no day replayed or skipped, no config_changed event
    state = runner.load_state()
    assert "peak_equity" not in state and state["config"] and len(state["close_history"]) == 3
    changed = dataclasses.replace(cfg, costs=dataclasses.replace(cfg.costs, commission_bps=2.0))
    with pytest.raises(RuntimeError, match=r"costs\.commission_bps: 1\.0 -> 2\.0"):
        PaperRunner(changed).step(as_of=data.dates[305])  # recorded now, so checked from here on


# --- 5. today's unfinished bar --------------------------------------------------------


def test_a_live_run_before_the_close_ignores_todays_unfinished_bar(tmp_path):
    cfg = make_config(tmp_path)
    data, _ = load_market_data(cfg.data)
    today, yesterday = data.dates[400], data.dates[399]  # the feed already has a bar for "today"
    runner = PaperRunner(cfg)
    ny = "America/New_York"

    [day] = runner.step(now=pd.Timestamp(f"{today.date()} 11:00", tz=ny))  # market open: the bar is still forming
    assert day.date == yesterday
    assert runner.step(now=pd.Timestamp(f"{today.date()} 16:14", tz=ny)) == []  # closed, but inside the buffer
    late = pd.Timestamp(f"{today.date()} 16:15", tz=ny).tz_convert("Asia/Tokyo")  # any clock works
    [day] = runner.step(now=late)
    assert day.date == today

    # An explicit as_of is a simulation after that day's close, whatever the clock says.
    sim = PaperRunner(make_config(tmp_path / "sim"))
    [day] = sim.step(as_of=today, now=pd.Timestamp(f"{today.date()} 09:00", tz=ny))
    assert day.date == today


def test_naive_clock_is_market_local_and_the_close_is_configurable(tmp_path):
    cfg = make_config(tmp_path)
    data, _ = load_market_data(cfg.data)
    today = data.dates[400]
    london = dataclasses.replace(cfg, paper=dataclasses.replace(cfg.paper, market_timezone="Europe/London",
                                                                   market_close="16:30", close_buffer_minutes=0))
    [day] = PaperRunner(london).step(now=pd.Timestamp(f"{today.date()} 16:29"))
    assert day.date == data.dates[399]
    [day] = PaperRunner(london).step(now=pd.Timestamp(f"{today.date()} 16:30"))
    assert day.date == today


# --- 6. one account per row, one run at a time ---------------------------------------


def at(cfg, state_dir, **paper):
    return dataclasses.replace(cfg, paper=dataclasses.replace(cfg.paper, state_dir=str(state_dir), **paper))


@pytest.mark.parametrize("copy", ["kept", "deleted", "stale"])
def test_moving_the_state_directory_keeps_the_account(tmp_path, monkeypatch, copy):
    """The account was keyed by the directory's absolute path. After a move the journal had no
    row under the new path, so a run started a fresh account (state.json deleted) or replayed
    committed days from a stale state.json."""
    cfg = paper_config(tmp_path)
    days = load_market_data(cfg.data)[0].dates[300:306]
    clean = PaperRunner(at(cfg, tmp_path / "clean"))
    walk(clean, days)

    runner = PaperRunner(at(cfg, tmp_path / "old"))
    walk(runner, days[:3])
    done = 3
    if copy == "stale":  # committed, then died before writing the copy
        fail_once(monkeypatch, paper_module, "atomic_write_json")
        with pytest.raises(OSError, match="simulated crash"):
            runner.step(as_of=days[3])
        done = 4
    elif copy == "deleted":  # it is only a copy, after all
        runner.state_path.unlink()
    shutil.move(tmp_path / "old", tmp_path / "new")

    moved = PaperRunner(at(cfg, tmp_path / "new"))
    assert moved.status()["last_processed"] == str(days[done - 1].date())
    walk(moved, days[done:])
    assert_same_journal(moved, clean)  # nothing restarted, replayed or skipped
    assert list(journal(moved, "select state_dir from paper_state")["state_dir"]) == ["account"]
    assert len(journal(moved, "select run_id from runs")) == 1


def test_moving_away_and_back_keeps_one_account(tmp_path):
    """Moved A -> B for one run and back (or one volume seen from a host and a container): the
    row left under A's path used to be picked up again, replaying the day run at B."""
    cfg = paper_config(tmp_path)
    days = load_market_data(cfg.data)[0].dates[300:306]
    clean = PaperRunner(at(cfg, tmp_path / "clean"))
    walk(clean, days)

    a, b = tmp_path / "A", tmp_path / "B"
    walk(PaperRunner(at(cfg, a)), days[:3])
    shutil.move(a, b)
    walk(PaperRunner(at(cfg, b)), days[3:4])
    shutil.move(b, a)
    runner = PaperRunner(at(cfg, a))
    assert [r.date for r in walk(runner, days[4:])] == list(days[4:])
    assert_same_journal(runner, clean)
    assert len(journal(runner, "select * from paper_state")) == 1


def test_a_shared_journal_never_starts_an_account_beside_another(tmp_path):
    """With paper.journal_path the key is state_dir as configured. A journal holding only other
    accounts is refused: starting afresh there, or from a stale state.json, could fork an
    account that was moved."""
    cfg = paper_config(tmp_path)
    days = load_market_data(cfg.data)[0].dates[300:306]
    shared = str(tmp_path / "shared.sqlite")
    x = PaperRunner(at(cfg, tmp_path / "x", journal_path=shared))
    walk(x, days[:2])
    y = PaperRunner(at(cfg, tmp_path / "y", journal_path=shared))
    expected = f"has no paper account under '{tmp_path / 'y'}', only under ['{tmp_path / 'x'}']"
    with pytest.raises(RuntimeError, match=re.escape(expected)):
        y.step(as_of=days[2])
    with pytest.raises(RuntimeError, match=re.escape(expected)):
        y.status()
    assert not (tmp_path / "y" / "state.json").exists()
    assert [r.date for r in x.step(as_of=days[2])] == [days[2]]  # the account itself carries on


def test_an_account_keyed_by_its_absolute_path_is_adopted(tmp_path):
    """Journals written before the key changed hold the account under the resolved state_dir."""
    cfg = paper_config(tmp_path)
    days = load_market_data(cfg.data)[0].dates[300:306]
    clean = PaperRunner(at(cfg, tmp_path / "clean"))
    walk(clean, days)

    runner = PaperRunner(cfg)
    walk(runner, days[:3])
    conn = sqlite3.connect(runner.journal_path)
    with conn:
        conn.execute("UPDATE paper_state SET state_dir = ?", (str(runner.state_dir.resolve()),))
    conn.close()
    assert runner.status()["last_processed"] == str(days[2].date())
    walk(runner, days[3:])
    assert_same_journal(runner, clean)
    assert list(journal(runner, "select state_dir from paper_state")["state_dir"]) == ["account"]


def test_a_journal_with_paper_runs_but_no_account_is_refused(tmp_path):
    cfg = paper_config(tmp_path)
    days = load_market_data(cfg.data)[0].dates[300:306]
    runner = PaperRunner(cfg)
    walk(runner, days[:2])
    conn = sqlite3.connect(runner.journal_path)
    with conn:
        conn.execute("DELETE FROM paper_state")
    conn.close()
    runner.state_path.unlink()
    with pytest.raises(RuntimeError, match="has paper runs but no account state to carry on from"):
        runner.step(as_of=days[2])
    assert len(journal(runner, "select * from runs")) == 1


def test_two_overlapping_steps_never_both_process_a_day(tmp_path, monkeypatch):
    """Behind the file lock, SQLite is the second guard: a step holds the write lock from
    before it reads the account (BEGIN IMMEDIATE). It used to read in autocommit, so a second
    step could read the same state, wait for the first commit, then commit its own copy."""
    cfg = paper_config(tmp_path)
    days = load_market_data(cfg.data)[0].dates[300:306]
    walk(PaperRunner(cfg), days[:2])
    monkeypatch.setattr(paper_module, "run_lock", lambda path: contextlib.nullcontext())  # both past the lock
    a_read, b_read = threading.Event(), threading.Event()
    load = Journal.load_paper_state

    def slow_load(self, key):
        state = load(self, key)
        if threading.current_thread().name == "A":
            a_read.set()
            b_read.wait(2)  # give B every chance to read the same state before A commits
        else:
            b_read.set()
        return state

    monkeypatch.setattr(Journal, "load_paper_state", slow_load)
    out = {}

    def run(name):
        if name == "B":
            a_read.wait(5)
        try:
            out[name] = [r.date for r in PaperRunner(cfg).step(as_of=days[2])]
        except RuntimeError as exc:  # B gave up waiting for the write lock: also fine
            out[name] = str(exc)

    threads = [threading.Thread(target=run, args=(name,), name=name) for name in "AB"]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert out["A"] == [days[2]]
    assert out["B"] == [] or "is locked: another run may be in progress" in out["B"]
    assert list(journal(PaperRunner(cfg), "select count(*) n from equity group by date")["n"]) == [1, 1, 1]


def test_the_readable_copy_is_repaired_by_a_step_with_nothing_to_do(tmp_path, monkeypatch):
    cfg = paper_config(tmp_path)
    days = load_market_data(cfg.data)[0].dates[300:306]
    runner = PaperRunner(cfg)
    walk(runner, days[:2])
    fail_once(monkeypatch, paper_module, "atomic_write_json")  # committed, then died writing the copy
    with pytest.raises(OSError, match="simulated crash"):
        runner.step(as_of=days[2])
    assert json.loads(runner.state_path.read_text())["last_processed"] == str(days[1].date())
    assert runner.step(as_of=days[2]) == []
    assert json.loads(runner.state_path.read_text()) == runner.load_state()
    runner.state_path.unlink()
    assert runner.step(as_of=days[2]) == []
    assert json.loads(runner.state_path.read_text()) == runner.load_state()


# --- 7. status ------------------------------------------------------------------------


def test_status_never_values_an_unpriced_position_at_zero(tmp_path):
    runner = PaperRunner(make_config(tmp_path))
    runner.state_dir.mkdir(parents=True)
    state = {
        "run_id": "r",
        "strategy": "trend()",
        "last_processed": "2020-01-02",
        "broker": {"cash": 1000.0, "positions": {"AAA": 10.0, "BBB": 5.0}, "pending": []},
        "marks": {"AAA": 100.0},
    }
    runner.state_path.write_text(json.dumps(state))  # older layout, read as is
    status = runner.status()
    assert status["positions"]["AAA"] == {"quantity": 10.0, "mark": 100.0, "value": 1000.0}
    assert status["positions"]["BBB"] == {"quantity": 5.0, "mark": None, "value": None}
    assert status["equity"] is None and status["unpriced"] == ["BBB"]

    state["marks"]["BBB"] = 20.0
    runner.state_path.write_text(json.dumps(state))
    status = runner.status()
    assert status["equity"] == 2100.0 and status["unpriced"] == []


def test_status_reads_an_older_journal_without_touching_it(tmp_path):
    """A journal from before paper_state existed: status falls back to state.json and
    leaves the file as it was (no table created behind the user's back)."""
    runner = PaperRunner(make_config(tmp_path))
    runner.state_dir.mkdir(parents=True)
    conn = sqlite3.connect(runner.journal_path)
    with conn:
        conn.execute("CREATE TABLE equity (run_id TEXT, date TEXT, equity REAL)")
    conn.close()
    before = runner.journal_path.read_bytes()
    assert runner.status() == {"initialised": False, "kill_switch": {"tripped": False, "reason": None, "at": None}}
    runner.state_path.write_text(json.dumps({"strategy": "s", "last_processed": "2020-01-02",
                                             "broker": {"cash": 5.0, "positions": {}}}))
    assert runner.status()["equity"] == 5.0
    assert runner.journal_path.read_bytes() == before
