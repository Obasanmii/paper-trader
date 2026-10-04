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
    "load_raw",
    "prepare_market_data",
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


def load_raw(data_cfg, source: DataSource | None = None) -> dict[str, pd.DataFrame]:
    """Download once. prepare_market_data can then clean any number of cut-offs
    from the same bars, e.g. each catch-up day exactly as it looked on the day."""
    source = source or build_source(data_cfg)
    return source.load(list(data_cfg.symbols), data_cfg.start, data_cfg.end)


def prepare_market_data(
    raw: dict[str, pd.DataFrame], data_cfg, as_of=None, hold_unconfirmed: bool = False
) -> tuple[MarketData, CleaningReport]:
    """Clean and align `raw` without modifying it. With `as_of`, raw data is cut
    *before* cleaning, so a simulated live day only ever sees what it could have
    known. `hold_unconfirmed` is for live use (see clean_symbol)."""
    if as_of is not None:
        cutoff = pd.Timestamp(as_of)
        raw = {sym: df.loc[pd.DatetimeIndex(pd.to_datetime(df.index)) <= cutoff] for sym, df in raw.items()}
    report = CleaningReport()
    c = data_cfg.cleaning
    cleaned = {
        sym: clean_symbol(
            df, sym, report, spike_threshold=c.spike_threshold, stale_run=c.stale_run, hold_unconfirmed=hold_unconfirmed
        )
        for sym, df in raw.items()
    }
    return align(cleaned, report, max_ffill_days=c.max_ffill_days), report


def load_market_data(
    data_cfg, as_of=None, source: DataSource | None = None, hold_unconfirmed: bool = False
) -> tuple[MarketData, CleaningReport]:
    """Load, clean and align in one go: prepare_market_data(load_raw(...))."""
    return prepare_market_data(load_raw(data_cfg, source), data_cfg, as_of=as_of, hold_unconfirmed=hold_unconfirmed)
