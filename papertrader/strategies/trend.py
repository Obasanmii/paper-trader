"""Trend following: the simplest idea with a long track record, which makes it a good first test."""
from __future__ import annotations

import numpy as np

from papertrader.strategies.base import Strategy, StrategyOutput, register


@register
class TrendFollowing(Strategy):
    """Long a symbol while its close is above its `lookback`-day moving average.

    Each symbol gets an equal slice (1/N) of capital. With `vol_target` set,
    each slice is scaled by vol_target / realised volatility (capped at
    `max_scale`), so calmer assets get bigger positions.
    """

    name = "trend"

    def __init__(self, lookback: int = 100, vol_target: float | None = None, vol_window: int = 20, max_scale: float = 1.0):
        if int(lookback) < 2:
            raise ValueError("lookback must be at least 2")
        if vol_target is not None and vol_target <= 0:
            raise ValueError("vol_target must be positive")
        super().__init__(lookback=int(lookback), vol_target=vol_target, vol_window=int(vol_window), max_scale=max_scale)
        self.lookback, self.vol_target = int(lookback), vol_target
        self.vol_window, self.max_scale = int(vol_window), max_scale

    def run(self, data):
        close = data.close
        sma = close.rolling(self.lookback, min_periods=self.lookback).mean()
        signal = close / sma - 1.0  # % above (+) or below (-) the average
        weights = (signal > 0).astype(float) / close.shape[1]
        if self.vol_target:
            vol = np.log(close).diff().rolling(self.vol_window, min_periods=self.vol_window).std() * np.sqrt(252)
            weights = weights * (self.vol_target / vol).clip(upper=self.max_scale)
        return StrategyOutput(self.finalise(weights, data), signal)
