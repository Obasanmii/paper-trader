"""Live-use data handling: the newest bar can't be confirmed, and one download
must clean many cut-offs exactly as each day would have seen them."""
import dataclasses

import numpy as np
import pandas as pd

from conftest import make_config
from papertrader.data import FIELDS, SyntheticSource, load_market_data, load_raw, prepare_market_data
from papertrader.data.cleaning import CleaningReport, clean_symbol


def _bars(n_days=80, symbols=("AAA", "BBB")):
    return SyntheticSource(n_days=n_days, seed=5).load(list(symbols))


def _bad_tick(df, i, factor=10.0):
    """A fat-finger close on row i (high widened to contain it, as a feed would print it)."""
    df = df.copy()
    df.iloc[i, df.columns.get_loc("close")] *= factor
    df.iloc[i, df.columns.get_loc("high")] = max(df["high"].iloc[i], df["close"].iloc[i])
    return df


def _real_jump(df, i, factor=1.5):
    """A move that persists: every price from row i on is scaled (e.g. takeover news)."""
    df = df.copy()
    for col in ("open", "high", "low", "close"):
        df.iloc[i:, df.columns.get_loc(col)] *= factor
    return df


def _issues(report, kind):
    return [i for i in report.issues if i.kind == kind]


def test_newest_bar_spike_is_held_back_only_when_asked():
    raw = _bad_tick(_bars()["AAA"], -1)
    newest = raw.index[-1]

    held_report = CleaningReport()
    held = clean_symbol(raw, "AAA", held_report, hold_unconfirmed=True)
    assert held.index[-1] == raw.index[-2]
    [issue] = _issues(held_report, "unconfirmed_move")
    assert issue.date == str(newest.date())
    move = np.log(raw["close"].iloc[-1] / raw["close"].iloc[-2])
    assert move > 2.0 and issue.action.startswith(f"held back: newest bar moved {move:+.2f} log")
    assert not _issues(held_report, "large_move")  # gone before check 8 could flag it

    # Off (research data): nothing changes; the tick is kept and flagged by check 8 as before.
    kept_report = CleaningReport()
    kept = clean_symbol(raw, "AAA", kept_report)
    assert kept.index[-1] == newest and kept["close"].iloc[-1] == raw["close"].iloc[-1]
    assert not _issues(kept_report, "unconfirmed_move")
    assert [i.date for i in _issues(kept_report, "large_move")] == [str(newest.date())]


def test_hold_back_leaves_ordinary_bars_alone():
    raw = _bars(n_days=500)["AAA"]
    report = CleaningReport()
    held = clean_symbol(raw, "AAA", report, hold_unconfirmed=True)
    assert len(report) == 0
    pd.testing.assert_frame_equal(held, clean_symbol(raw, "AAA", CleaningReport()))


def test_next_day_decides_reverted_tick_versus_real_move():
    base = _bars()["AAA"]
    yesterday, today = base.index[-2], base.index[-1]

    # Bad tick yesterday, reverted today: dropped as a spike; today itself is ordinary.
    report = CleaningReport()
    clean = clean_symbol(_bad_tick(base, -2), "AAA", report, hold_unconfirmed=True)
    assert yesterday not in clean.index and clean.index[-1] == today
    assert [i.date for i in _issues(report, "price_spike")] == [str(yesterday.date())]
    assert not _issues(report, "unconfirmed_move")

    # Real move yesterday that persisted today: held back yesterday, kept and flagged today.
    jumped = _real_jump(base, -2)
    on_the_day = CleaningReport()
    clean_symbol(jumped.loc[:yesterday], "AAA", on_the_day, hold_unconfirmed=True)
    assert [i.date for i in _issues(on_the_day, "unconfirmed_move")] == [str(yesterday.date())]
    report = CleaningReport()
    clean = clean_symbol(jumped, "AAA", report, hold_unconfirmed=True)
    assert clean.index[-1] == today and clean.loc[yesterday, "close"] == jumped.loc[yesterday, "close"]
    assert [i.date for i in _issues(report, "large_move")] == [str(yesterday.date())]
    assert not _issues(report, "unconfirmed_move") and not _issues(report, "price_spike")


def test_held_back_day_is_marked_but_untradable():
    cfg = make_config()
    raw = _bars()
    raw["AAA"] = _bad_tick(raw["AAA"], -1)
    today, yesterday = raw["AAA"].index[-1], raw["AAA"].index[-2]

    md, report = prepare_market_data(raw, cfg.data, hold_unconfirmed=True)
    assert md.dates[-1] == today  # BBB traded today, so the day stays on the calendar
    assert np.isnan(md.open.loc[today, "AAA"])  # nothing fills against the held-back bar
    assert md.close.loc[today, "AAA"] == raw["AAA"].loc[yesterday, "close"]  # marked at the last good close
    assert md.high.loc[today, "AAA"] == md.low.loc[today, "AAA"] == md.close.loc[today, "AAA"]
    assert md.volume.loc[today, "AAA"] == 0.0
    assert md.open.loc[today, "BBB"] == raw["BBB"].loc[today, "open"]
    assert [(i.symbol, i.date) for i in _issues(report, "gap_filled")] == [("AAA", str(today.date()))]

    # A lone symbol has no one else to keep the day on the calendar: it waits for confirmation.
    alone, _ = prepare_market_data({"AAA": raw["AAA"]}, cfg.data, hold_unconfirmed=True)
    assert alone.dates[-1] == yesterday


def test_prepare_from_one_download_matches_load_and_never_mutates_raw():
    cfg = make_config(n_days=400)
    dirty = dataclasses.replace(cfg.data.synthetic, inject_errors=True)  # so cleaning has something to change
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, synthetic=dirty))
    raw = load_raw(cfg.data)
    frames, before = dict(raw), {sym: df.copy(deep=True) for sym, df in raw.items()}
    dates = raw["AAA"].index.unique()

    for as_of in (dates[150], dates[-60], dates[-1], None):
        for hold in (False, True):
            md, report = prepare_market_data(raw, cfg.data, as_of=as_of, hold_unconfirmed=hold)
            ref_md, ref_report = load_market_data(cfg.data, as_of=as_of, hold_unconfirmed=hold)
            for f in FIELDS:
                pd.testing.assert_frame_equal(getattr(md, f), getattr(ref_md, f))
            pd.testing.assert_frame_equal(report.to_frame(), ref_report.to_frame())
    assert {"price_spike", "invalid_volume", "duplicate_date"} <= set(report.counts())  # cleaning did change things
    assert raw.keys() == frames.keys() and all(raw[sym] is frames[sym] for sym in frames)
    for sym, df in raw.items():
        pd.testing.assert_frame_equal(df, before[sym])
