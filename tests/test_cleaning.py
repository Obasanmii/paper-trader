import numpy as np
import pandas as pd

from papertrader.data import SyntheticSource, load_market_data
from papertrader.data.cleaning import CleaningReport, align, clean_symbol

from conftest import make_config


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
