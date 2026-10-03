"""The strategy contract.

A strategy maps market data to target portfolio weights (fraction of equity
per symbol). One rule matters above all others:

    Row t of the weights may depend only on bars dated <= t.

The engine turns row t into orders after the close of t and fills them at
the open of t+1, so a strategy that obeys the rule cannot trade on a price
it couldn't have seen. `lookahead_violations()` checks the rule by
recomputing the strategy on data truncated at several dates and comparing
with the full-data run. tests/test_no_lookahead.py runs it for every
registered strategy, so a new strategy that peeks at the future fails CI.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from papertrader.data.market import MarketData


@dataclass
class StrategyOutput:
    weights: pd.DataFrame  # dates x symbols, fraction of equity; 0 = flat
    signals: pd.DataFrame | None = None  # optional raw signal, logged next to each decision


class Strategy(ABC):
    name = "base"

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def run(self, data: MarketData) -> StrategyOutput:
        ...

    def target_weights(self, data: MarketData) -> pd.DataFrame:
        return self.run(data).weights

    @staticmethod
    def finalise(weights: pd.DataFrame, data: MarketData) -> pd.DataFrame:
        """Align to the data, clear NaN/inf, and force 0 where a symbol has no price."""
        w = weights.reindex(index=data.dates, columns=data.symbols)
        w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return w.where(data.close.notna(), 0.0).astype(float)

    def describe(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({params})"


REGISTRY: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    if cls.name in REGISTRY:
        raise ValueError(f"duplicate strategy name {cls.name!r}")
    REGISTRY[cls.name] = cls
    return cls


def build_strategy(name: str, params: dict | None = None) -> Strategy:
    if name not in REGISTRY:
        raise ValueError(f"unknown strategy {name!r}; registered: {sorted(REGISTRY)}")
    try:
        return REGISTRY[name](**(params or {}))
    except TypeError as exc:  # unknown parameter names end up here
        raise ValueError(f"bad parameters for {name}: {exc}") from exc


def lookahead_violations(strategy: Strategy, data: MarketData, check_dates=None, n_checks: int = 5, atol: float = 1e-10):
    """Return the dates where weights computed on truncated data differ from the full run.

    An empty list means no lookahead was detected at the dates checked.
    """
    full = strategy.run(data).weights
    if check_dates is None:
        positions = np.linspace(len(data) // 3, len(data) - 2, n_checks).astype(int)
        check_dates = [data.dates[i] for i in positions]
    bad = []
    for t in check_dates:
        t = pd.Timestamp(t)
        partial = strategy.run(data.truncate(t)).weights
        a = partial.loc[t].to_numpy(dtype=float)
        b = full.loc[t].to_numpy(dtype=float)
        if not np.allclose(a, b, atol=atol, rtol=0.0, equal_nan=True):
            bad.append(t)
    return bad
