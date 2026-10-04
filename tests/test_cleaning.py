import dataclasses
import math

import numpy as np
import pandas as pd

from conftest import make_config
from papertrader.config import CleaningConfig
from papertrader.data import SyntheticSource, load_market_data, load_raw, prepare_market_data
from papertrader.data.cleaning import CleaningReport, align, clean_symbol


def test_planted_errors_are_all_caught():
    raw = SyntheticSource(n_days=1000, inject_errors=True, seed=3).load(["AAA"])["AAA"]
    report = CleaningReport()
    clean = clean_symbol(raw, "AAA", report)
    kinds = report.counts()
    for kind in ("duplicate_date", "invalid_price", "invalid_volume", "price_spike", "ohlc_inconsistent", "stale_prices"):
        assert kinds.get(kind, 0) >= 1, f"missed {kind}: {kinds}"
    assert clean.index.is_unique and clean.index.is_monotonic_increasing
    assert (clean["high"] >= clean[["open", "close"]].max(axis=1)).all()
    assert (clean["low"] <= clean[["open", "close"]].min(axis=1)).all()
    assert (clean["volume"] >= 0).all()
    assert np.log(clean["close"]).diff().abs().max() < 0.25  # the 10x bad tick is gone


def test_clean_data_reports_nothing():
    raw = SyntheticSource(n_days=500).load(["AAA"])["AAA"]
    report = CleaningReport()
    clean_symbol(raw, "AAA", report)
    assert len(report) == 0


def test_alignment_forward_fills_marks_but_never_trades_or_backfills():
    idx = pd.bdate_range("2021-01-04", periods=10)
    full = pd.DataFrame({"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 100.0}, index=idx)
    gappy = full.drop(idx[[4, 5]])  # two missing days
    late = full.iloc[3:]  # starts trading later
    report = CleaningReport()
    md = align({"A": full, "B": gappy, "C": late}, report, max_ffill_days=3)
    assert md.close["B"].notna().all()  # marks carried across the gap
    assert md.open["B"].iloc[[4, 5]].isna().all()  # ...but untradable
    assert md.close["C"].iloc[:3].isna().all()  # no back-filling before C existed
    assert report.counts()["gap_filled"] == 2


def test_max_ffill_days_zero_turns_forward_fill_off():
    idx = pd.bdate_range("2021-01-04", periods=10)
    full = pd.DataFrame({"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 100.0}, index=idx)
    report = CleaningReport()
    md = align({"A": full, "B": full.drop(idx[[4, 5]])}, report, max_ffill_days=0)  # pandas rejects ffill(limit=0)
    assert md.close["B"].iloc[[4, 5]].isna().all()  # no bar, no mark
    assert md.close["B"].drop(idx[[4, 5]]).notna().all() and md.close["A"].notna().all()
    assert report.counts() == {"gap_too_long": 2}
    assert all("forward fill is off" in i.action for i in report.issues)


def test_a_held_back_bar_with_forward_fill_off_leaves_the_day_unmarked():
    cfg = make_config(n_days=300)
    no_ffill = dataclasses.replace(cfg.data, cleaning=CleaningConfig(max_ffill_days=0))
    raw = load_raw(cfg.data)
    # Clean data has no gaps to fill, so the whole load is unchanged (it used to crash in pandas).
    pd.testing.assert_frame_equal(load_market_data(no_ffill)[0].close, load_market_data(cfg.data)[0].close)
    jumped = raw["AAA"].copy()
    jumped.iloc[-1, :4] *= 1.5  # an unconfirmable +0.41 log move on the newest bar: held back
    raw = {**raw, "AAA": jumped}
    filled, _ = prepare_market_data(raw, cfg.data, hold_unconfirmed=True)
    md, report = prepare_market_data(raw, no_ffill, hold_unconfirmed=True)
    assert report.counts()["unconfirmed_move"] == 1
    last = md.dates[-1]
    assert math.isnan(md.close.at[last, "AAA"]) and math.isnan(md.open.at[last, "AAA"])
    assert md.close.loc[last].drop("AAA").notna().all()
    # With forward fill on, the held-back day is marked at the previous close instead (and still untradable).
    assert filled.close.at[last, "AAA"] == filled.close["AAA"].iloc[-2] and math.isnan(filled.open.at[last, "AAA"])


def test_live_style_load_cleans_only_the_past():
    cfg = make_config(n_days=400)
    full, _ = load_market_data(cfg.data)
    cut, _ = load_market_data(cfg.data, as_of=full.dates[200])
    assert cut.dates[-1] == full.dates[200]
    pd.testing.assert_frame_equal(cut.close, full.close.loc[: full.dates[200]])


def test_csv_source_round_trip(tmp_path):
    from papertrader.data import CSVSource

    frames = SyntheticSource(n_days=300).load(["AAA", "BBB"])
    for sym, df in frames.items():
        out = df.copy()
        out["adj_close"] = out["close"] * 0.5  # e.g. a 2:1 split later in history
        out.to_csv(tmp_path / f"{sym}.csv", index_label="Date")
    loaded = CSVSource(tmp_path).load(["AAA", "BBB"])
    assert list(loaded["AAA"].columns) == ["open", "high", "low", "close", "volume"]
    pd.testing.assert_series_equal(loaded["AAA"]["close"], frames["AAA"]["close"] * 0.5, check_names=False, check_freq=False)


def test_csv_source_date_range_keeps_unsorted_rows_for_cleaning(tmp_path):
    from papertrader.data import CSVSource

    df = SyntheticSource(n_days=30).load(["AAA"])["AAA"]
    df.iloc[::-1].to_csv(tmp_path / "AAA.csv", index_label="date")  # newest first, as some exports are
    loaded = CSVSource(tmp_path).load(["AAA"], start="2015-01-05", end=str(df.index[20].date()))["AAA"]
    assert sorted(loaded.index) == list(df.index[1:21])  # range applied, order left for cleaning
    report = CleaningReport()
    clean_symbol(loaded, "AAA", report)
    assert report.counts() == {"unsorted": 1}


def _fake_yfinance(monkeypatch, past_end_bar=False):
    """Stand-in for yfinance (not installed, and tests stay offline). Like the real
    history(), `end` is exclusive and bars are tz-aware exchange-local midnights."""
    import sys
    import types

    calls = []

    class Ticker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, start=None, end=None, auto_adjust=False):
            calls.append({"start": start, "end": end, "auto_adjust": auto_adjust})
            idx = pd.bdate_range("2024-01-02", "2024-01-12", tz="America/New_York")
            if end is not None:
                last = pd.Timestamp(end, tz="America/New_York") + pd.Timedelta(days=3 if past_end_bar else 0)
                idx = idx[idx < last]
            px = np.linspace(100.0, 110.0, len(idx))
            cols = {"Open": px, "High": px + 1, "Low": px - 1, "Close": px, "Volume": 1e6, "Dividends": 0.0}
            return pd.DataFrame(cols, index=pd.DatetimeIndex(idx, name="Date"))

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=Ticker))
    return calls


def test_yfinance_end_date_is_inclusive(monkeypatch):
    from papertrader.data import YFinanceSource

    calls = _fake_yfinance(monkeypatch)
    out = YFinanceSource().load(["AAA"], start="2024-01-02", end="2024-01-05")
    assert out["AAA"].index[-1] == pd.Timestamp("2024-01-05")  # yfinance alone would stop at the 4th
    assert out["AAA"].index.tz is None and list(out["AAA"].columns) == ["open", "high", "low", "close", "volume"]
    assert calls == [{"start": "2024-01-02", "end": "2024-01-06", "auto_adjust": True}]

    calls = _fake_yfinance(monkeypatch)
    YFinanceSource().load(["AAA"])
    assert calls == [{"start": None, "end": None, "auto_adjust": True}]


def test_yfinance_never_returns_bars_past_the_end_date(monkeypatch):
    from papertrader.data import YFinanceSource

    _fake_yfinance(monkeypatch, past_end_bar=True)  # e.g. a timezone mismatch lets the next day through
    out = YFinanceSource().load(["AAA"], end="2024-01-05")
    assert out["AAA"].index[-1] == pd.Timestamp("2024-01-05")
