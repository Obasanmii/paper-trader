"""Trend following: the simplest idea with a long track record, which makes it a good first test."""
from __future__ import annotations

import numpy as np
import pandas as pd

from papertrader.strategies.base import Strategy, StrategyOutput, register


@register
class TrendFollowing(Strategy):
    """Long a symbol while its close is above its `lookback`-day moving average.

    Each symbol gets an equal slice (1/N) of capital. With `vol_target` set,
    each slice is scaled by vol_target / realised volatility (capped at
    `max_scale`), so calmer assets get bigger positions.

    `band` adds hysteresis against whipsaw: enter only once the close is more
    than `band` (a fraction) above the average, exit only once it is `band`
    below it, and keep the previous position in between. A close that hugs
    the average otherwise flips the position, and pays costs, every few days.
    """

    name = "trend"

    def __init__(
        self,
        lookback: int = 100,
        vol_target: float | None = None,
        vol_window: int = 20,
        max_scale: float = 1.0,
        band: float = 0.0,
    ):
        if int(lookback) < 2:
            raise ValueError("lookback must be at least 2")
        if vol_target is not None and vol_target <= 0:
            raise ValueError("vol_target must be positive")
        if not 0 <= band < 1:  # also rejects NaN
            raise ValueError("band must be in [0, 1)")
        params = dict(lookback=int(lookback), vol_target=vol_target, vol_window=int(vol_window), max_scale=max_scale)
        if band:  # band=0 is the original rule: leave it out so describe(), and saved paper state, still match
            params["band"] = float(band)
        super().__init__(**params)
        self.lookback, self.vol_target = int(lookback), vol_target
        self.vol_window, self.max_scale, self.band = int(vol_window), max_scale, float(band)

    def in_trend(self, signal: pd.DataFrame) -> pd.DataFrame:
        """1.0 while long, 0.0 while flat. Each day either enters, exits or carries the
        previous state forward; ffill only looks back, so this stays point-in-time.
        Exiting *at* -band (not only below it) makes band=0 exactly `signal > 0`."""
        enter = signal > self.band
        stay = signal > -self.band  # False on NaN: no average yet (or a gap) means flat
        state = np.where(enter, 1.0, np.where(stay, np.nan, 0.0))
        return pd.DataFrame(state, index=signal.index, columns=signal.columns).ffill().fillna(0.0)

    def run(self, data):
        close = data.close
        sma = close.rolling(self.lookback, min_periods=self.lookback).mean()
        signal = close / sma - 1.0  # % above (+) or below (-) the average
        weights = self.in_trend(signal) / close.shape[1]
        if self.vol_target:
            vol = np.log(close).diff().rolling(self.vol_window, min_periods=self.vol_window).std() * np.sqrt(252)
            weights = weights * (self.vol_target / vol).clip(upper=self.max_scale)
        return StrategyOutput(self.finalise(weights, data), signal)
