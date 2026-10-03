import math

import numpy as np
import pandas as pd

from papertrader.metrics import sharpe
from papertrader.validation import (
    bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    per_period_sharpe,
    probabilistic_sharpe_ratio,
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


def test_timing_test_separates_skill_from_exposure(data):
    # Always-invested: shifting changes nothing, so p should be ~1 (no timing to test).
    flat = pd.DataFrame(0.2, index=data.dates, columns=data.symbols)
    assert timing_test(flat, data, n_perm=100)["p_value"] > 0.9
    # A clairvoyant strategy (reads the open two days ahead) should crush its shifted copies.
    opens = data.open.ffill()
    future = (opens.shift(-2) > opens.shift(-1)).astype(float) / len(data.symbols)
    result = timing_test(future, data, n_perm=100)
    assert result["p_value"] < 0.02 and result["sharpe"] > 3
