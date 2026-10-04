"""Statistics for not fooling ourselves.

probabilistic_sharpe_ratio  P(true Sharpe > benchmark), given sample length,
                            skew and fat tails (Bailey & Lopez de Prado, 2012).
deflated_sharpe_ratio       The same, but the bar is raised to the Sharpe you'd
                            expect from the *best of N* tries of pure noise. If
                            you tested 20 variants and kept the winner, this is
                            the honest number (Bailey & Lopez de Prado, 2014).
bootstrap_sharpe_ci         Confidence interval for the Sharpe ratio from a
                            stationary block bootstrap (keeps autocorrelation).
bootstrap_sharpe_difference Is strategy A's Sharpe really above B's? A paired
                            bootstrap of the difference. Comparing two point
                            estimates says "yes" half the time on pure noise.
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
    of every variant you tried, including the one you picked. N counts them
    all, including variants with no Sharpe (NaN: e.g. they never traded).
    They were still tries, and dropping them would let dead parameter sets
    pad the grid without raising the bar. The spread of Sharpes comes from
    the finite ones; with fewer than two distinct ones there is no spread to
    deflate by (float noise counts as none). If dead variants padded the count, the answer is NaN, which
    no check passes: otherwise duplicates of one live result plus dead sets
    would pass undeflated. If every variant gave the same result, they were
    one trial in effect, judged by its plain PSR.
    """
    n = np.asarray(trial_sharpes_per_period, dtype=float).size
    if n < 2:
        return probabilistic_sharpe_ratio(returns)
    finite = _clean(trial_sharpes_per_period)
    if finite.size < 2 or np.ptp(finite) <= 1e-9 * np.abs(finite).max():
        return math.nan if finite.size < n else probabilistic_sharpe_ratio(returns)
    sr0 = expected_max_sharpe(n, float(finite.std(ddof=1)))
    return probabilistic_sharpe_ratio(returns, benchmark_sharpe=sr0 * math.sqrt(PERIODS_PER_YEAR))


def per_period_sharpe(returns) -> float:
    r = _clean(returns)
    if r.size < 2 or not r.std(ddof=1) > 0:
        return math.nan
    return float(r.mean() / r.std(ddof=1))


def stationary_bootstrap_indices(n: int, n_boot: int, mean_block: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, n) resampling paths for the stationary bootstrap (Politis & Romano, 1994).

    Blocks of random (geometric) length starting at random points, wrapping
    around the end. Resampling whole runs of days keeps the autocorrelation
    and volatility clustering that an i.i.d. bootstrap would destroy.
    """
    starts = rng.integers(0, n, size=(n_boot, n))
    jump = rng.random((n_boot, n)) < 1.0 / mean_block
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, n):
        idx[:, t] = np.where(jump[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def _flat_rows(samples: np.ndarray) -> np.ndarray:
    """Rows with no variance. Compared exactly: the std of a constant row is ~1e-19, not 0,
    and dividing by it would score a steady drip as a Sharpe in the quadrillions."""
    return samples.max(axis=1) == samples.min(axis=1)


def _row_sharpes(samples: np.ndarray) -> np.ndarray:
    """Annualised Sharpe of each row. A row of all zeros (in cash for the whole resample) has
    Sharpe 0; any other row without variance has none, so it is NaN for the caller to count
    against the strategy. Dropping those rows instead biases the bound upwards: for a strategy
    that is mostly flat, the dropped resamples are the ones that missed its few active days."""
    with np.errstate(invalid="ignore", divide="ignore"):
        s = samples.mean(axis=1) / samples.std(axis=1, ddof=1) * math.sqrt(PERIODS_PER_YEAR)
    return np.where(_flat_rows(samples), np.where(samples.any(axis=1), np.nan, 0.0), s)


def _quantile(values: np.ndarray, q: float) -> float:
    """np.quantile, but a quantile that reaches into -inf values is -inf, not NaN."""
    with np.errstate(invalid="ignore"):
        x = float(np.quantile(values, q))
    return -math.inf if math.isnan(x) else x


def bootstrap_sharpe_ci(returns, n_boot: int = 2000, mean_block: int = 20, alpha: float = 0.05, seed: int = 0):
    """(low, high) percentile interval for the annualised Sharpe via the stationary bootstrap.

    A resample with no Sharpe (no variance, but not all zeros) counts as -inf: it can only
    pull the interval down, never be dropped. Too little data or no variance at all gives NaNs.
    """
    r = _clean(returns)
    n = r.size
    if n < 30 or r.max() == r.min():  # no variance at all: nothing to resample
        return (math.nan, math.nan)
    idx = stationary_bootstrap_indices(n, n_boot, mean_block, np.random.default_rng(seed))
    s = _row_sharpes(r[idx])
    s[~np.isfinite(s)] = -math.inf
    return (_quantile(s, alpha / 2), _quantile(s, 1 - alpha / 2))


def _paired(returns_a, returns_b) -> tuple[np.ndarray, np.ndarray]:
    """Two return series as aligned arrays, keeping only days where both are finite.

    Series are matched on their date index, so a missing day in one can't
    silently shift the other against it.
    """
    if isinstance(returns_a, pd.Series) and isinstance(returns_b, pd.Series):
        both = pd.concat([returns_a, returns_b], axis=1, join="inner")
        a, b = both.iloc[:, 0].to_numpy(dtype=float), both.iloc[:, 1].to_numpy(dtype=float)
    else:
        a, b = np.asarray(returns_a, dtype=float), np.asarray(returns_b, dtype=float)
        if a.shape != b.shape:
            raise ValueError(f"returns_a and returns_b must be aligned, got shapes {a.shape} and {b.shape}")
    keep = np.isfinite(a) & np.isfinite(b)
    return a[keep], b[keep]


def bootstrap_sharpe_difference(
    returns_a, returns_b, n_boot: int = 2000, mean_block: int = 20, alpha: float = 0.05, seed: int = 0
) -> dict:
    """Is A's annualised Sharpe above B's? A paired stationary bootstrap of the difference.

    Both series are resampled along the *same* index paths, so days stay
    paired: a strategy and its benchmark share most of their market risk,
    and the difference is far less noisy than either Sharpe on its own.
    Returns the point difference, a one-sided lower bound (the `alpha`
    quantile of the resampled differences) and a p-value (the share of
    resampled differences <= 0). Call A better only if the bound is > 0.
    Too little data or zero variance gives NaNs, which no check should pass.

    Every resample counts. `n_degenerate` says how many had a series with
    no variance: all zeros scores Sharpe 0 (see _row_sharpes), and any
    other has no Sharpe, so its difference counts as -inf, against A.
    """
    a, b = _paired(returns_a, returns_b)
    out = {"difference": math.nan, "lower": math.nan, "p_value": math.nan, "n_boot": 0, "n_degenerate": 0}
    if a.size < 30:
        return out
    out["difference"] = sharpe(a) - sharpe(b)
    if not math.isfinite(out["difference"]):
        return out
    idx = stationary_bootstrap_indices(a.size, n_boot, mean_block, np.random.default_rng(seed))
    sa, sb = a[idx], b[idx]
    d = _row_sharpes(sa) - _row_sharpes(sb)
    d[~np.isfinite(d)] = -math.inf  # A or B has no Sharpe: a loss for A, never a dropped resample
    degenerate = int((_flat_rows(sa) | _flat_rows(sb)).sum())
    out.update(lower=_quantile(d, alpha), p_value=float(np.mean(d <= 0)), n_boot=int(d.size), n_degenerate=degenerate)
    return out


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
