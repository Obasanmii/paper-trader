"""MarketData: aligned OHLCV panels, each a (dates x symbols) DataFrame.

Conventions
-----------
* `close` is always the mark. After cleaning it is forward-filled across
  short gaps, so positions can always be valued.
* `open` is NaN on any day a symbol did not actually trade (a filled gap).
  The simulated broker will not fill an order against a NaN open, so you
  can never trade at a price that didn't exist.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

FIELDS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class MarketData:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame

    def __post_init__(self):
        idx, cols = self.close.index, list(self.close.columns)
        for name in FIELDS:
            frame = getattr(self, name)
            if not frame.index.equals(idx) or list(frame.columns) != cols:
                raise ValueError(f"MarketData.{name} is not aligned with close")
        if idx.has_duplicates or not idx.is_monotonic_increasing:
            raise ValueError("MarketData index must be strictly increasing")

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    def __len__(self) -> int:
        return len(self.close)

    def truncate(self, end) -> "MarketData":
        """Everything known at the close of `end`, and nothing after it."""
        end = pd.Timestamp(end)
        return MarketData(**{f: getattr(self, f).loc[:end] for f in FIELDS})

    def between(self, start=None, end=None) -> "MarketData":
        start = None if start is None else pd.Timestamp(start)
        end = None if end is None else pd.Timestamp(end)
        return MarketData(**{f: getattr(self, f).loc[start:end] for f in FIELDS})

    def map_prices(self, fn) -> "MarketData":
        """Apply fn to every price panel (not volume). Handy for stress tests."""
        return MarketData(
            open=fn(self.open), high=fn(self.high), low=fn(self.low), close=fn(self.close), volume=self.volume
        )
