"""Statistics for not fooling ourselves.

probabilistic_sharpe_ratio  P(true Sharpe > benchmark), given sample length,
                            skew and fat tails (Bailey & Lopez de Prado, 2012).
deflated_sharpe_ratio       The same, but the bar is raised to the Sharpe you'd
                            expect from the *best of N* tries of pure noise. If
                            you tested 20 variants and kept the winner, this is
                            the honest number (Bailey & Lopez de Prado, 2014).
bootstrap_sharpe_ci         Confidence interval for the Sharpe ratio from a
                            stationary block bootstrap (keeps autocorrelation).
timing_test                 Does the strategy's *timing* matter? Shift its
                            positions in time at random and see how often the
                            shifted version does as well. Separates skill from
                            simply being long a rising market.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from papertrader.metrics import PERIODS_PER_YEAR, sharpe

_N = NormalDist()
EULER_GAMMA = 0.5772156649015329


def _clean(returns) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    return r[np.isfinite(r)]


def _moments(r: np.ndarray) -> tuple[float, float, float]:
    """Per-period Sharpe, skewness, and (non-excess) kurtosis."""
    mu, sd = r.mean(), r.std(ddof=1)
    z = (r - mu) / r.std(ddof=0)
    return float(mu / sd), float(np.mean(z**3)), float(np.mean(z**4))


def probabilistic_sharpe_ratio(returns, benchmark_sharpe: float = 0.0, periods: int = PERIODS_PER_YEAR) -> float:
    """P(true annualised Sharpe > benchmark_sharpe). Wants >= 0.95 to call something significant."""
    r = _clean(returns)
    if r.size < 3 or not r.std(ddof=1) > 0:
        return math.nan
    sr, skew, kurt = _moments(r)
    sr_star = benchmark_sharpe / math.sqrt(periods)
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if denom <= 0:
        return math.nan
    return _N.cdf((sr - sr_star) * math.sqrt(r.size - 1) / math.sqrt(denom))


def expected_max_sharpe(n_trials: int, sharpe_std: float) -> float:
    """Expected best (per-period) Sharpe among n_trials strategies whose true Sharpe is zero."""
    if n_trials < 2:
        return 0.0
    return sharpe_std * (
        (1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / n_trials) + EULER_GAMMA * _N.inv_cdf(1 - 1 / (n_trials * math.e))
    )


def deflated_sharpe_ratio(returns, trial_sharpes_per_period) -> float:
    """PSR of the chosen strategy against the best-of-N-noise bar.

    `trial_sharpes_per_period` holds the *per-period* (not annualised) Sharpe
    of every variant you tried, including the one you picked.
    """
    trials = _clean(trial_sharpes_per_period)
    if trials.size < 2:
        return probabilistic_sharpe_ratio(returns)
    sr0 = expected_max_sharpe(trials.size, float(trials.std(ddof=1)))
    return probabilistic_sharpe_ratio(returns, benchmark_sharpe=sr0 * math.sqrt(PERIODS_PER_YEAR))


def per_period_sharpe(returns) -> float:
    r = _clean(returns)
    if r.size < 2 or not r.std(ddof=1) > 0:
        return math.nan
    return float(r.mean() / r.std(ddof=1))


def bootstrap_sharpe_ci(returns, n_boot: int = 2000, mean_block: int = 20, alpha: float = 0.05, seed: int = 0):
    """(low, high) percentile interval for the annualised Sharpe via the stationary bootstrap."""
    r = _clean(returns)
    n = r.size
    if n < 30:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, n))
    jump = rng.random((n_boot, n)) < 1.0 / mean_block
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, n):
        idx[:, t] = np.where(jump[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    samples = r[idx]
    sd = samples.std(axis=1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = samples.mean(axis=1) / sd * math.sqrt(PERIODS_PER_YEAR)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return (math.nan, math.nan)
    return (float(np.quantile(s, alpha / 2)), float(np.quantile(s, 1 - alpha / 2)))


def timing_test(weights: pd.DataFrame, data, start=None, end=None, n_perm: int = 500, min_shift: int = 63, seed: int = 0) -> dict:
    """Compare the strategy's gross Sharpe with circularly time-shifted copies of its own positions.

    Shifting keeps the average exposure and how positions persist, but breaks
    the link between *when* it's invested and what the market did next.
    Execution timing matches the engine: weights decided at the close of t
    are held from the open of t+1 to the open of t+2 (open-to-open returns).
    Costs are ignored here; this is a test of signal, not of profit.

    Caveat: for slow signals (months-long positions) shifted copies stay
    partly aligned with the market, which makes this test conservative.
    `min_shift` (default one quarter) limits that.
    """
    opens = data.open.ffill()
    oo = (opens / opens.shift(1) - 1.0).fillna(0.0)
    w = weights.reindex(index=data.dates, columns=data.symbols).fillna(0.0)
    held_raw = w.shift(2)  # decided at t -> earns the open(t+1) -> open(t+2) move, booked at t+2
    held = held_raw.fillna(0.0)
    window = pd.Series(held_raw.notna().all(axis=1).to_numpy(), index=data.dates)  # skip days with no decision yet
    if start is not None:
        window &= data.dates >= pd.Timestamp(start)
    if end is not None:
        window &= data.dates <= pd.Timestamp(end)
    H = held.loc[window].to_numpy()
    R = oo.loc[window].to_numpy()
    T = H.shape[0]
    if T < 3 * min_shift:
        return {"sharpe": math.nan, "p_value": math.nan, "n_perm": 0}
    actual = sharpe((H * R).sum(axis=1))
    if not math.isfinite(actual):
        return {"sharpe": actual, "p_value": math.nan, "n_perm": 0}
    rng = np.random.default_rng(seed)
    shifts = rng.integers(min_shift, T - min_shift, size=n_perm)
    null = np.array([sharpe((np.roll(H, k, axis=0) * R).sum(axis=1)) for k in shifts])
    null = null[np.isfinite(null)]
    p = (1 + int((null >= actual).sum())) / (1 + null.size)
    return {
        "sharpe": actual,
        "p_value": p,
        "n_perm": int(null.size),
        "null_mean": float(null.mean()) if null.size else math.nan,
        "null_95th": float(np.quantile(null, 0.95)) if null.size else math.nan,
    }
