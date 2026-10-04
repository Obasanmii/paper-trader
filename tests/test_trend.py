import numpy as np
import pandas as pd
import pytest

from papertrader.strategies import Strategy, build_strategy


def trend_before_band(data, lookback=100, vol_target=None, vol_window=20, max_scale=1.0):
    """TrendFollowing.run as it was before `band` existed, kept verbatim as the reference."""
    close = data.close
    sma = close.rolling(lookback, min_periods=lookback).mean()
    signal = close / sma - 1.0
    weights = (signal > 0).astype(float) / close.shape[1]
    if vol_target:
        vol = np.log(close).diff().rolling(vol_window, min_periods=vol_window).std() * np.sqrt(252)
        weights = weights * (vol_target / vol).clip(upper=max_scale)
    return Strategy.finalise(weights, data)


def position_changes(weights: pd.DataFrame) -> int:
    return int((weights > 0).astype(int).diff().abs().sum().sum())


def with_ties_and_gaps(data):
    """Whole-number prices, so close == average happens often at lookback 2, plus a symbol
    that lists late and one with a gap: the edge cases where a state machine could drift."""
    def edit(frame):
        frame = frame.round()
        frame.iloc[:150, 0] = np.nan
        frame.iloc[300:304, 1] = np.nan
        return frame

    return data.map_prices(edit)


@pytest.mark.parametrize(
    "params",
    [{}, {"lookback": 20}, {"lookback": 2}, {"lookback": 20, "vol_target": 0.15, "vol_window": 10, "max_scale": 2.0}],
    ids=["defaults", "lookback20", "lookback2", "vol_target"],
)
@pytest.mark.parametrize("edited", [False, True], ids=["plain", "ties_and_gaps"])
def test_band_zero_reproduces_the_original_rule_exactly(data, params, edited):
    d = with_ties_and_gaps(data) if edited else data
    new = build_strategy("trend", {**params, "band": 0.0}).target_weights(d)
    pd.testing.assert_frame_equal(new, trend_before_band(d, **params), check_exact=True)


def test_the_edited_data_really_has_ties_and_gaps(data):
    d = with_ties_and_gaps(data)
    signal = build_strategy("trend", {"lookback": 2}).run(d).signals
    assert (signal == 0).sum().sum() > 100 and d.close.isna().any().any()


def test_a_band_cuts_whipsaw_on_the_shared_data(data):
    def changes(band):
        return position_changes(build_strategy("trend", {"lookback": 20, "band": band}).target_weights(data))

    assert changes(0.01) < changes(0.0)
    assert changes(0.02) < changes(0.01)


def test_band_holds_the_previous_state_inside_the_band():
    strategy = build_strategy("trend", {"band": 0.02})
    signal = pd.DataFrame({"AAA": [np.nan, 0.01, 0.03, 0.01, -0.01, -0.02, 0.01, 0.025, np.nan, 0.01]})
    held = strategy.in_trend(signal)["AAA"].tolist()
    # no average -> flat; inside the band -> hold; > +band -> enter; <= -band -> exit; a gap resets to flat
    assert held == [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]


@pytest.mark.parametrize("band", [-0.01, 1.0, 1.5, float("nan")])
def test_band_must_be_in_unit_interval(band):
    with pytest.raises(ValueError, match="band"):
        build_strategy("trend", {"band": band})


def test_describe_is_unchanged_without_a_band():
    """Paper state stores describe() and refuses to resume under a different strategy,
    so band=0 must not change it: it is the same rule."""
    assert build_strategy("trend", {}).describe() == "trend(lookback=100, vol_target=None, vol_window=20, max_scale=1.0)"
    assert "band=0.02" in build_strategy("trend", {"band": 0.02}).describe()
