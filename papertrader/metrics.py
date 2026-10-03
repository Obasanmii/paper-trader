"""Performance statistics on an equity curve."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252


def returns_from_equity(equity: pd.Series) -> pd.Series:
    return (equity / equity.shift(1) - 1.0).dropna()


def sharpe(returns, periods: int = PERIODS_PER_YEAR) -> float:
    """Annualised Sharpe ratio (risk-free rate taken as zero)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return math.nan
    sd = r.std(ddof=1)
    if not sd > 0:
        return math.nan
    return float(r.mean() / sd * math.sqrt(periods))


def max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """(deepest drawdown as a negative fraction, longest stretch under water in bars)."""
    dd = equity / equity.cummax() - 1.0
    longest = current = 0
    for under in (dd < 0).to_numpy():
        current = current + 1 if under else 0
        longest = max(longest, current)
    return float(dd.min()), longest


def performance_summary(equity: pd.Series, periods: int = PERIODS_PER_YEAR) -> dict:
    equity = equity.dropna()
    r = returns_from_equity(equity)
    n = len(r)
    years = n / periods if n else math.nan
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    cagr = (1 + total) ** (1 / years) - 1 if n and total > -1 else math.nan
    vol = float(r.std(ddof=1) * math.sqrt(periods)) if n > 1 else math.nan
    downside = float(np.sqrt(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2)) * math.sqrt(periods)) if n else math.nan
    sortino = float(r.mean() * periods / downside) if downside and downside > 0 else math.nan
    mdd, mdd_days = max_drawdown(equity)
    active = r[r != 0]
    return {
        "start": str(equity.index[0].date()),
        "end": str(equity.index[-1].date()),
        "days": n,
        "total_return": total,
        "cagr": float(cagr),
        "volatility": vol,
        "sharpe": sharpe(r, periods),
        "sortino": sortino,
        "max_drawdown": mdd,
        "max_drawdown_days": mdd_days,
        "calmar": float(cagr / abs(mdd)) if mdd < 0 and math.isfinite(cagr) else math.nan,
        "up_days_when_invested": float((active > 0).mean()) if len(active) else math.nan,
        "skew": float(r.skew()) if n > 2 else math.nan,
        "excess_kurtosis": float(r.kurt()) if n > 3 else math.nan,
    }
