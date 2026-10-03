"""Where raw bars come from.

Every source returns {symbol: DataFrame} with a tz-naive DatetimeIndex and
float columns open, high, low, close, volume. Raw means raw: sources do no
cleaning, that's cleaning.py's job, so every fix gets recorded in one place.
"""
from __future__ import annotations

import zlib
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume"]


class DataSource(ABC):
    @abstractmethod
    def load(self, symbols, start=None, end=None) -> dict[str, pd.DataFrame]:
        ...


def _slice(df: pd.DataFrame, start, end) -> pd.DataFrame:
    start = None if start is None else pd.Timestamp(start)
    end = None if end is None else pd.Timestamp(end)
    return df.loc[start:end]


def normalise(df: pd.DataFrame, adjust: bool = True) -> pd.DataFrame:
    """Lower-case columns, tz-naive midnight index, optional split/dividend adjustment."""
    df = df.rename(columns=lambda c: str(c).strip().lower().replace(" ", "_"))
    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "date"
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}; got {list(df.columns)}")
    if adjust and "adj_close" in df.columns:
        # Scale OHLC by adj_close / close so returns aren't polluted by splits and dividends.
        factor = df["adj_close"].astype(float) / df["close"].astype(float)
        df = df.copy()
        for c in ("open", "high", "low", "close"):
            df[c] = df[c].astype(float) * factor
    return df[COLUMNS].astype(float)


class CSVSource(DataSource):
    """One CSV per symbol: <directory>/<SYMBOL>.csv with a date column and OHLCV columns.

    If an `adj_close` (or "Adj Close") column is present and `use_adjusted` is
    true, OHLC are scaled to adjusted prices.
    """

    def __init__(self, directory: str | Path, date_column: str = "date", use_adjusted: bool = True):
        self.directory = Path(directory)
        self.date_column = date_column.lower()
        self.use_adjusted = use_adjusted

    def load(self, symbols, start=None, end=None):
        out = {}
        for sym in symbols:
            path = self.directory / f"{sym}.csv"
            if not path.exists():
                raise FileNotFoundError(f"no data file for {sym}: {path}")
            raw = pd.read_csv(path)
            raw.columns = [str(c).strip().lower() for c in raw.columns]
            if self.date_column not in raw.columns:
                raise ValueError(f"{path}: no '{self.date_column}' column (found {list(raw.columns)})")
            raw = raw.set_index(self.date_column)
            out[sym] = _slice(normalise(raw, adjust=self.use_adjusted), start, end)
        return out


class SyntheticSource(DataSource):
    """Deterministic fake markets with a *known* answer, for tests and demos.

    Daily log returns follow a one-factor model with fat-tailed (Student-t)
    shocks. Two knobs decide whether there is anything to find:

    * regime_drift = 0 and drift = 0: a driftless random walk. No strategy has
      a real edge, so anything that looks good here is overfit by construction.
    * regime_drift > 0: each symbol's drift flips between +regime_drift and
      -regime_drift following a persistent hidden state. Trend-following has a
      genuine edge, and a good testing process should be able to tell.

    `inject_errors=True` plants realistic data problems (a bad tick, a
    duplicate row, a broken high, a negative volume, a missing close, a
    stale run) so the cleaning step has something to catch.
    """

    def __init__(
        self,
        n_days: int = 2520,
        start: str = "2015-01-02",
        seed: int = 7,
        drift: float = 0.0,
        vol: float = 0.20,
        market_corr: float = 0.5,
        regime_drift: float = 0.0,
        regime_persistence: float = 0.995,
        t_dof: float | None = 5.0,
        start_price: float = 100.0,
        inject_errors: bool = False,
    ):
        if not 0 <= market_corr < 1:
            raise ValueError("market_corr must be in [0, 1)")
        self.n_days, self.start, self.seed = n_days, start, seed
        self.drift, self.vol, self.market_corr = drift, vol, market_corr
        self.regime_drift, self.regime_persistence = regime_drift, regime_persistence
        self.t_dof, self.start_price, self.inject_errors = t_dof, start_price, inject_errors

    def _shocks(self, rng, n):
        if self.t_dof is None or self.t_dof <= 2:
            return rng.standard_normal(n)
        return rng.standard_t(self.t_dof, n) / np.sqrt(self.t_dof / (self.t_dof - 2))  # unit variance

    def _regimes(self, rng, n):
        states = np.empty(n)
        states[0] = 1.0 if rng.random() < 0.5 else -1.0
        flips = rng.random(n) > self.regime_persistence
        for i in range(1, n):
            states[i] = -states[i - 1] if flips[i] else states[i - 1]
        return states

    def load(self, symbols, start=None, end=None):
        n = self.n_days
        dates = pd.bdate_range(self.start, periods=n)
        market = self._shocks(np.random.default_rng([self.seed, 0]), n)
        daily_vol = self.vol / np.sqrt(252)
        out = {}
        for sym in symbols:
            # Per-symbol stream: the same symbol always gets the same path,
            # whatever else is in the universe.
            rng = np.random.default_rng([self.seed, zlib.crc32(sym.encode())])
            shocks = np.sqrt(self.market_corr) * market + np.sqrt(1 - self.market_corr) * self._shocks(rng, n)
            mu = np.full(n, self.drift / 252 - 0.5 * daily_vol**2)
            if self.regime_drift > 0:
                mu = mu + self._regimes(rng, n) * self.regime_drift / 252
            r = mu + daily_vol * shocks
            close = self.start_price * np.exp(np.cumsum(r))
            prev_close = np.r_[self.start_price, close[:-1]]
            overnight = 0.3 * r + rng.normal(0.0, 0.2 * daily_vol, n)  # part of the move happens overnight
            open_ = prev_close * np.exp(overnight)
            wicks = np.abs(rng.normal(0.0, 0.5 * daily_vol, (2, n)))
            high = np.maximum(open_, close) * np.exp(wicks[0])
            low = np.minimum(open_, close) * np.exp(-wicks[1])
            volume = np.round(rng.lognormal(13.0, 0.4, n))
            df = pd.DataFrame(
                {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=dates
            )
            df.index.name = "date"
            if self.inject_errors:
                df = self._plant_errors(df, rng)
            out[sym] = _slice(df, start, end)
        return out

    @staticmethod
    def _plant_errors(df: pd.DataFrame, rng) -> pd.DataFrame:
        df = df.copy()
        n = len(df)
        # One spot per segment so planted problems never overlap each other.
        bounds = np.linspace(60, n - 60, 7).astype(int)
        spots = [int(rng.integers(lo, max(lo + 1, hi - 10))) for lo, hi in zip(bounds[:-1], bounds[1:])]
        c = {name: df.columns.get_loc(name) for name in df.columns}
        i_tick, i_high, i_vol, i_nan, i_stale, i_dup = (int(s) for s in spots)
        df.iloc[i_tick, c["close"]] *= 10.0  # fat-finger print that reverts next day
        df.iloc[i_tick, c["high"]] = df.iloc[i_tick, c["close"]]
        df.iloc[i_high, c["high"]] = df.iloc[i_high, c["low"]] * 0.99  # high below low
        df.iloc[i_vol, c["volume"]] = -500.0
        df.iloc[i_nan, c["close"]] = np.nan
        stale_px = df.iloc[i_stale, c["close"]]
        for k in range(6):  # feed stuck on one price
            df.iloc[i_stale + k, c["close"]] = stale_px
        dup = df.iloc[[i_dup]]
        return pd.concat([df, dup]).sort_index(kind="stable")


class YFinanceSource(DataSource):
    """Daily bars from Yahoo Finance via the unofficial `yfinance` package.

    Good enough for learning; not an official feed. It can change or rate-limit
    without warning, so cache what you download (see scripts/download_yahoo.py).
    Install with: pip install yfinance
    """

    def load(self, symbols, start=None, end=None):
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("yfinance is not installed: pip install yfinance (or use the csv source)") from exc
        out = {}
        for sym in symbols:
            hist = yf.Ticker(sym).history(start=start, end=end, auto_adjust=True)
            if hist is None or hist.empty:
                raise ValueError(f"yfinance returned no data for {sym}")
            out[sym] = normalise(hist, adjust=False)  # auto_adjust=True already adjusted OHLC
        return out
