"""Every registered strategy must pass the truncation test. New strategies are picked up automatically.

Default parameters alone leave branches unchecked (trend's vol_target and band, the gbm model),
so every combination in every shipped config's research grid is checked too, plus explicit
cases for branches no grid reaches.
"""
import itertools
from pathlib import Path

import numpy as np
import pytest

from papertrader.config import load_config
from papertrader.strategies import REGISTRY, Strategy, StrategyOutput, build_strategy, lookahead_violations

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
FAST_PARAMS = {"ml_classifier": {"retrain_every": 63, "train_window": 300, "min_train_samples": 200}}

# Kept even when a config grid happens to cover them, so coverage never depends on configs.
EXPLICIT_CASES = [
    ("trend", {"lookback": 20, "vol_target": 0.15, "vol_window": 10, "max_scale": 2.0}),
    ("trend", {"lookback": 20, "band": 0.02}),
    ("trend", {"lookback": 60, "band": 0.05, "vol_target": 0.10}),
    ("mean_reversion", {"lookback": 20, "entry_z": 0.5, "full_z": 1.5}),
    ("ml_classifier", {**FAST_PARAMS["ml_classifier"], "model": "gbm"}),
]


def _case_id(name: str, params: dict) -> str:
    return name + ":" + (",".join(f"{k}={v}" for k, v in sorted(params.items())) or "defaults")


def grid_cases() -> dict[str, tuple[str, dict, set]]:
    """case id -> (strategy, params, configs it came from).

    Each combination of a config's research.param_grid is merged over that config's
    strategy params, as research.run_research does. The grids are small enough to run
    whole; only identical combinations (two configs sharing a grid) are run once.
    """
    cases = {}
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        cfg = load_config(path)
        name, grid = cfg.strategy.name, cfg.research.param_grid
        for combo in itertools.product(*grid.values()):  # an empty grid is one case: the config's own params
            # Fast ML settings go under the grid, so every grid value is still the one tested.
            params = {**cfg.strategy.params, **FAST_PARAMS.get(name, {}), **dict(zip(grid, combo))}
            cases.setdefault(_case_id(name, params), (name, params, set()))[2].add(path.name)
    return cases


GRID_CASES = grid_cases()


def assert_point_in_time(strategy: Strategy, data, n_checks: int = 6) -> None:
    dates = data.dates[np.linspace(len(data) // 3, len(data) - 2, n_checks).astype(int)]
    w = strategy.target_weights(data)
    assert w.notna().all().all() and (w.where(data.close.isna(), 0.0) == 0).all().all()
    # A strategy that is flat on every check date passes trivially; that would be a hole, not a pass.
    assert w.loc[dates].abs().to_numpy().sum() > 0, f"{strategy.describe()} is flat on every check date"
    assert lookahead_violations(strategy, data, check_dates=dates) == []


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_strategy_never_uses_future_data(name, data):
    strategy = build_strategy(name, FAST_PARAMS.get(name, {}))
    assert lookahead_violations(strategy, data, n_checks=4) == []


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_weights_are_finite_and_zero_without_a_price(name, data):
    w = build_strategy(name, FAST_PARAMS.get(name, {})).target_weights(data)
    assert w.shape == data.close.shape
    assert w.notna().all().all()
    assert (w.where(data.close.isna(), 0.0) == 0).all().all()


@pytest.mark.parametrize("case", sorted(GRID_CASES))
def test_shipped_param_grids_never_use_future_data(case, data):
    name, params, _ = GRID_CASES[case]
    assert_point_in_time(build_strategy(name, params), data)


@pytest.mark.parametrize(("name", "params"), EXPLICIT_CASES, ids=[_case_id(n, p) for n, p in EXPLICIT_CASES])
def test_non_default_branches_never_use_future_data(name, params, data):
    assert_point_in_time(build_strategy(name, params), data)


def test_every_shipped_config_feeds_the_grid_cases():
    """Guards against the grid cases silently going empty (configs moved, glob broken)."""
    shipped = {p.name for p in CONFIG_DIR.glob("*.yaml")}
    assert shipped and set().union(*(configs for _, _, configs in GRID_CASES.values())) == shipped


class Cheater(Strategy):
    """Buys whatever goes up tomorrow. Looks brilliant in a naive backtest."""

    name = "cheater"

    def run(self, data):
        tomorrow_up = (data.close.shift(-1) > data.close).astype(float) / len(data.symbols)
        return StrategyOutput(self.finalise(tomorrow_up, data))


def test_the_lookahead_check_catches_a_cheater(data):
    assert lookahead_violations(Cheater(), data, n_checks=4) != []


def test_the_parametrized_check_catches_a_cheater_too(data):
    with pytest.raises(AssertionError, match=r"\[Timestamp"):  # it fails on the dates, not on the sanity checks
        assert_point_in_time(Cheater(), data)
