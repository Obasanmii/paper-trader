"""Data layer: sources -> cleaning -> aligned MarketData."""
from __future__ import annotations

import pandas as pd

from papertrader.data.cleaning import CleaningReport, align, clean_symbol
from papertrader.data.market import FIELDS, MarketData
from papertrader.data.sources import CSVSource, DataSource, SyntheticSource, YFinanceSource

__all__ = [
    "CleaningReport",
    "CSVSource",
    "DataSource",
    "FIELDS",
    "MarketData",
    "SyntheticSource",
    "YFinanceSource",
    "build_source",
    "load_market_data",
]


def build_source(data_cfg) -> DataSource:
    if data_cfg.source == "synthetic":
        s = data_cfg.synthetic
        return SyntheticSource(
            n_days=s.n_days,
            start=s.start,
            seed=s.seed,
            drift=s.drift,
            vol=s.vol,
            market_corr=s.market_corr,
            regime_drift=s.regime_drift,
            regime_persistence=s.regime_persistence,
            t_dof=s.t_dof,
            inject_errors=s.inject_errors,
        )
    if data_cfg.source == "csv":
        return CSVSource(data_cfg.csv_dir, use_adjusted=data_cfg.use_adjusted)
    if data_cfg.source == "yfinance":
        return YFinanceSource()
    raise ValueError(f"unknown data source {data_cfg.source!r}")


def load_market_data(data_cfg, as_of=None, source: DataSource | None = None) -> tuple[MarketData, CleaningReport]:
    """Load, clean and align. With `as_of`, raw data is cut *before* cleaning,
    so a simulated live day only ever sees what it could have known."""
    source = source or build_source(data_cfg)
    raw = source.load(list(data_cfg.symbols), data_cfg.start, data_cfg.end)
    if as_of is not None:
        cutoff = pd.Timestamp(as_of)
        raw = {sym: df.loc[pd.DatetimeIndex(pd.to_datetime(df.index)) <= cutoff] for sym, df in raw.items()}
    report = CleaningReport()
    c = data_cfg.cleaning
    cleaned = {
        sym: clean_symbol(df, sym, report, spike_threshold=c.spike_threshold, stale_run=c.stale_run)
        for sym, df in raw.items()
    }
    return align(cleaned, report, max_ffill_days=c.max_ffill_days), report
