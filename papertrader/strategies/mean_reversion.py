"""Short-horizon mean reversion (long only)."""
from __future__ import annotations

import numpy as np

from papertrader.strategies.base import Strategy, StrategyOutput, register


@register
class MeanReversion(Strategy):
    """Buy symbols that closed well below their recent average.

    z = (close - mean) / std over `lookback` days. The position ramps from 0
    at z = -entry_z up to a full 1/N slice at z = -full_z, and is flat
    otherwise. Costs matter a lot here: this style trades often.
    """

    name = "mean_reversion"

    def __init__(self, lookback: int = 10, entry_z: float = 1.0, full_z: float = 2.0):
        if int(lookback) < 3:
            raise ValueError("lookback must be at least 3")
        if not 0 <= entry_z < full_z:
            raise ValueError("need 0 <= entry_z < full_z")
        super().__init__(lookback=int(lookback), entry_z=entry_z, full_z=full_z)
        self.lookback, self.entry_z, self.full_z = int(lookback), entry_z, full_z

    def run(self, data):
        close = data.close
        mean = close.rolling(self.lookback, min_periods=self.lookback).mean()
        std = close.rolling(self.lookback, min_periods=self.lookback).std()
        z = (close - mean) / std.replace(0.0, np.nan)
        strength = ((-z - self.entry_z) / (self.full_z - self.entry_z)).clip(0.0, 1.0)
        return StrategyOutput(self.finalise(strength / close.shape[1], data), z)
