"""The benchmark every idea has to beat."""
from __future__ import annotations

import numpy as np

from papertrader.strategies.base import Strategy, StrategyOutput, register


@register
class EqualWeight(Strategy):
    """Hold every symbol that has a price, in equal proportion. No timing, no cleverness."""

    name = "equal_weight"

    def __init__(self):
        super().__init__()

    def run(self, data):
        listed = data.close.notna().astype(float)
        count = listed.sum(axis=1).replace(0, np.nan)
        return StrategyOutput(self.finalise(listed.div(count, axis=0), data))
