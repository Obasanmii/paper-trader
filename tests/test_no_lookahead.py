"""Every registered strategy must pass the truncation test. New strategies are picked up automatically."""
import pytest

from papertrader.strategies import REGISTRY, Strategy, StrategyOutput, build_strategy, lookahead_violations

FAST_PARAMS = {"ml_classifier": {"retrain_every": 63, "train_window": 300, "min_train_samples": 200}}


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


class Cheater(Strategy):
    """Buys whatever goes up tomorrow. Looks brilliant in a naive backtest."""

    name = "cheater"

    def run(self, data):
        tomorrow_up = (data.close.shift(-1) > data.close).astype(float) / len(data.symbols)
        return StrategyOutput(self.finalise(tomorrow_up, data))


def test_the_lookahead_check_catches_a_cheater(data):
    assert lookahead_violations(Cheater(), data, n_checks=4) != []
