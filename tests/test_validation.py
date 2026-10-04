import math

import numpy as np
import pandas as pd
import pytest

from papertrader.metrics import PERIODS_PER_YEAR, sharpe
from papertrader.validation import (
    bootstrap_sharpe_ci,
    bootstrap_sharpe_difference,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    per_period_sharpe,
    probabilistic_sharpe_ratio,
    stationary_bootstrap_indices,
    timing_test,
)


def test_psr_is_high_for_real_edge_and_middling_for_noise():
    rng = np.random.default_rng(0)
    edge = rng.normal(0.001, 0.01, 2500)  # annualised Sharpe ~1.6
    noise = rng.normal(0.0, 0.01, 2500)
    assert probabilistic_sharpe_ratio(edge) > 0.99
    assert 0.02 < probabilistic_sharpe_ratio(noise) < 0.98


def test_best_of_many_noise_strategies_is_deflated():
    rng = np.random.default_rng(1)
    trials = rng.normal(0.0, 0.01, (50, 1000))  # 50 strategies, none with any edge
    best = max(trials, key=lambda r: r.mean() / r.std())
    sharpes = [per_period_sharpe(r) for r in trials]
    assert sharpe(best) > 0.8  # looks good in isolation...
    assert probabilistic_sharpe_ratio(best) > 0.95  # ...and even "significant" naively
    assert deflated_sharpe_ratio(best, sharpes) < 0.8  # but not once you count the tries


def test_expected_max_sharpe_grows_with_trials():
    assert expected_max_sharpe(1, 0.05) == 0.0
    assert expected_max_sharpe(10, 0.05) < expected_max_sharpe(100, 0.05) < expected_max_sharpe(1000, 0.05)


def test_bootstrap_interval_contains_the_estimate():
    r = np.random.default_rng(2).normal(0.0005, 0.01, 1500)
    lo, hi = bootstrap_sharpe_ci(r, n_boot=500)
    assert lo < sharpe(r) < hi


def _noise(seed, n=1000, mu=0.0):
    return np.random.default_rng(seed).normal(mu, 0.01, n)


def _annualised_row_sharpes(samples):
    return samples.mean(axis=1) / samples.std(axis=1, ddof=1) * math.sqrt(PERIODS_PER_YEAR)


def test_stationary_bootstrap_resamples_runs_of_consecutive_days():
    idx = stationary_bootstrap_indices(500, 200, 20, np.random.default_rng(0))
    assert idx.shape == (200, 500) and idx.min() >= 0 and idx.max() < 500
    consecutive = np.diff(idx, axis=1) % 500 == 1
    assert 0.93 < consecutive.mean() < 0.97  # a new block starts with probability 1/20


def test_sharpe_ci_and_difference_share_the_bootstrap_paths():
    a, b = _noise(1, 400, 0.0005), _noise(2, 400)
    idx = stationary_bootstrap_indices(400, 300, 20, np.random.default_rng(0))
    s_a, s_b = _annualised_row_sharpes(a[idx]), _annualised_row_sharpes(b[idx])
    assert bootstrap_sharpe_ci(a, n_boot=300) == pytest.approx((np.quantile(s_a, 0.025), np.quantile(s_a, 0.975)))
    # Paired: both series are resampled on the same paths, so day t of A stays next to day t of B.
    result = bootstrap_sharpe_difference(a, b, n_boot=300)
    assert result["lower"] == pytest.approx(np.quantile(s_a - s_b, 0.05))
    assert result["p_value"] == pytest.approx(np.mean(s_a - s_b <= 0))
    assert result["difference"] == pytest.approx(sharpe(a) - sharpe(b))


def test_sharpe_difference_of_a_series_with_itself_is_not_significant():
    r = _noise(5)
    result = bootstrap_sharpe_difference(r, r, n_boot=500)
    assert result["difference"] == 0 and result["lower"] == 0 and result["p_value"] == 1
    assert not result["lower"] > 0


def test_sharpe_difference_finds_a_clear_edge_over_noise():
    result = bootstrap_sharpe_difference(_noise(6, mu=0.0015), _noise(5), n_boot=500)  # annualised Sharpe ~2.4 vs 0
    assert result["lower"] > 0 and result["p_value"] < 0.05


def test_sharpe_difference_of_independent_noise_is_not_significant():
    # The old check (point Sharpe A > point Sharpe B) passes here: A's sample Sharpe is higher.
    a, b = _noise(103), _noise(203)
    assert sharpe(a) > sharpe(b)
    result = bootstrap_sharpe_difference(a, b, n_boot=500)
    assert result["lower"] < 0 and result["p_value"] > 0.05


def test_pairing_detects_a_small_edge_on_shared_risk():
    # A = B plus a small steady edge. Their separate intervals overlap heavily,
    # but the paired difference is tight because the shared noise cancels.
    b = _noise(5)
    a = b + 0.0003  # ~0.48 of annualised Sharpe
    assert bootstrap_sharpe_ci(a, n_boot=500)[0] < bootstrap_sharpe_ci(b, n_boot=500)[1]
    assert bootstrap_sharpe_difference(a, b, n_boot=500)["lower"] > 0.3


def test_sharpe_difference_aligns_series_by_date_and_fails_closed():
    dates = pd.bdate_range("2020-01-01", periods=300)
    a = pd.Series(_noise(7, 300, 0.001), index=dates)
    b = pd.Series(_noise(8, 300), index=dates)
    shifted = bootstrap_sharpe_difference(a.iloc[5:], b, n_boot=200)  # A is missing its first 5 days
    assert shifted == bootstrap_sharpe_difference(a.iloc[5:].to_numpy(), b.iloc[5:].to_numpy(), n_boot=200)
    with pytest.raises(ValueError, match="aligned"):
        bootstrap_sharpe_difference(a.iloc[5:].to_numpy(), b.to_numpy())
    # Too short, or no variance: NaN, which no check can pass.
    assert math.isnan(bootstrap_sharpe_difference(a.iloc[:20], b.iloc[:20])["lower"])
    assert math.isnan(bootstrap_sharpe_difference(a, b * 0.0)["lower"])


def _mostly_in_cash(seed=66, n=252, active=15):
    """In cash except for one short burst, against a noisy benchmark. With seed 66, dropping
    every resample that missed the burst put the paired lower bound at +0.57: a PASS."""
    rng = np.random.default_rng(seed)
    bench = rng.normal(0.0004, 0.01, n)
    strat = np.zeros(n)
    start = rng.integers(0, n - 20)
    strat[start : start + active] = rng.normal(0.002, 0.01, active)
    return strat, bench


def _row_sharpes_cash_as_zero(samples):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(samples.any(axis=1), _annualised_row_sharpes(samples), 0.0)


def test_resamples_spent_in_cash_score_sharpe_zero_and_still_count():
    a, b = _mostly_in_cash()
    idx = stationary_bootstrap_indices(a.size, 2000, 20, np.random.default_rng(0))
    in_cash = ~a[idx].any(axis=1)
    assert 200 < in_cash.sum() < 400  # these resamples missed the burst entirely
    result = bootstrap_sharpe_difference(a, b, n_boot=2000)
    assert result["n_boot"] == 2000 and result["n_degenerate"] == in_cash.sum()
    d = _row_sharpes_cash_as_zero(a[idx]) - _annualised_row_sharpes(b[idx])
    assert result["lower"] == pytest.approx(np.quantile(d, 0.05))
    assert result["p_value"] == pytest.approx(np.mean(d <= 0))
    assert not result["lower"] > 0  # 15 lucky days in a year don't beat the benchmark
    s = _row_sharpes_cash_as_zero(a[idx])
    assert bootstrap_sharpe_ci(a, n_boot=2000) == pytest.approx((np.quantile(s, 0.025), np.quantile(s, 0.975)))


def test_a_resample_with_no_sharpe_counts_against_the_strategy():
    # A steady drip with one noisy week. Resamples that miss the week are constant: they have no
    # Sharpe at all (dividing by their ~1e-19 std would make it a huge one), and each is a loss for A.
    a = np.full(252, 0.0004)
    a[100:105] = [0.01, -0.01, 0.02, -0.015, 0.005]
    b = _noise(9, 252)
    idx = stationary_bootstrap_indices(252, 2000, 20, np.random.default_rng(0))
    constant = (a[idx] == 0.0004).all(axis=1)
    assert constant.mean() > 0.05
    result = bootstrap_sharpe_difference(a, b, n_boot=2000)
    assert result["n_boot"] == 2000 and result["n_degenerate"] == constant.sum()
    assert result["p_value"] >= constant.mean()
    assert result["lower"] == -math.inf  # over 5% of resamples are losses of unknown size: fail closed
    assert bootstrap_sharpe_ci(a, n_boot=2000)[0] == -math.inf
    assert all(math.isnan(x) for x in bootstrap_sharpe_ci(np.zeros(252)))  # nothing to resample at all


def test_dsr_counts_trials_that_never_traded():
    rng = np.random.default_rng(1)
    trials = rng.normal(0.0, 0.01, (3, 1000))
    best = max(trials, key=lambda r: r.mean() / r.std())
    finite = [per_period_sharpe(r) for r in trials]
    padded = finite + [math.nan] * 5  # five parameter sets that never traded: no Sharpe, still tries
    # N is all 8 trials; the spread comes from the 3 that have a Sharpe.
    bar = expected_max_sharpe(8, float(np.std(finite, ddof=1))) * math.sqrt(PERIODS_PER_YEAR)
    assert deflated_sharpe_ratio(best, padded) == pytest.approx(probabilistic_sharpe_ratio(best, benchmark_sharpe=bar))
    assert deflated_sharpe_ratio(best, padded) < deflated_sharpe_ratio(best, finite)  # dead trials raise the bar
    # One real trial among dead ones: no spread to deflate by, so NaN, which no check passes.
    assert math.isnan(deflated_sharpe_ratio(best, [finite[0], math.nan, math.nan]))
    assert deflated_sharpe_ratio(best, [finite[0]]) == probabilistic_sharpe_ratio(best)  # one trial: nothing to deflate


def test_timing_test_separates_skill_from_exposure(data):
    # Always-invested: shifting changes nothing, so p should be ~1 (no timing to test).
    flat = pd.DataFrame(0.2, index=data.dates, columns=data.symbols)
    assert timing_test(flat, data, n_perm=100)["p_value"] > 0.9
    # A clairvoyant strategy (reads the open two days ahead) should crush its shifted copies.
    opens = data.open.ffill()
    future = (opens.shift(-2) > opens.shift(-1)).astype(float) / len(data.symbols)
    result = timing_test(future, data, n_perm=100)
    assert result["p_value"] < 0.02 and result["sharpe"] > 3


def test_dead_trials_padding_one_live_result_fail_the_deflated_sharpe():
    """A parameter that changes nothing (vol_window without vol_target) duplicates the one live
    result, and dead sets (lookbacks longer than the history) pad N. With no spread among the
    live results that used to pass as the plain PSR while the verdict claimed 6 deflated trials."""
    r = np.random.default_rng(3).normal(0.001, 0.01, 1500)
    live = per_period_sharpe(r)
    assert math.isnan(deflated_sharpe_ratio(r, [live, live, math.nan, math.nan, math.nan, math.nan]))
    # The same result twice and nothing dead: one trial in effect, judged by its plain PSR.
    assert deflated_sharpe_ratio(r, [live, live]) == probabilistic_sharpe_ratio(r)


def test_float_noise_is_not_a_spread_between_trials():
    """A CSV round trip changes a Sharpe in its 15th digit: that is the same result twice, not two."""
    r = np.random.default_rng(3).normal(0.001, 0.01, 1500)
    live = per_period_sharpe(r)
    assert math.isnan(deflated_sharpe_ratio(r, [live, np.nextafter(live, 1.0), math.nan, math.nan]))
    assert deflated_sharpe_ratio(r, [live, live * (1 + 1e-12)]) == probabilistic_sharpe_ratio(r)
