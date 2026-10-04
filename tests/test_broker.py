import math

import pytest

from conftest import order
from papertrader.core import OrderStatus
from papertrader.execution import CostModel, SimulatedBroker


def test_fills_at_next_open_with_slippage_and_commission():
    b = SimulatedBroker(10_000, CostModel(commission_bps=10, slippage_bps=50))
    b.submit(order(qty=10, ref=100.0))
    fills, cancels = b.process_open("2020-01-03", {"AAA": 102.0})
    [f] = fills
    assert cancels == []
    assert f.price == pytest.approx(102.0 * 1.005)  # buys pay up
    assert f.commission == pytest.approx(10 * f.price * 0.001)
    assert b.cash == pytest.approx(10_000 - 10 * f.price - f.commission)
    assert b.positions == {"AAA": 10}


def test_sells_fill_before_buys_so_cash_is_available():
    b = SimulatedBroker(0, CostModel(0, 0))
    b.positions = {"AAA": 10}
    b.submit(order("BBB", 20, ref=50.0))  # needs the sale proceeds
    b.submit(order("AAA", -10, ref=100.0))
    fills, _ = b.process_open("2020-01-03", {"AAA": 100.0, "BBB": 50.0})
    assert [f.symbol for f in fills] == ["AAA", "BBB"]
    assert b.positions == {"BBB": 20} and b.cash == pytest.approx(0.0)


def test_unaffordable_buy_is_cut_to_cash_not_margined():
    b = SimulatedBroker(1_000, CostModel(0, 0))
    b.submit(order(qty=10, ref=90.0))  # decided at 90, opens at 150
    [f], _ = b.process_open("2020-01-03", {"AAA": 150.0})
    assert f.quantity == 6 and b.cash >= 0


def test_partial_fill_returns_the_remainder_as_a_cancellation():
    b = SimulatedBroker(1_000, CostModel(0, 0))
    o = order(qty=10, ref=90.0)
    b.submit(o)
    [f], [(rest, why)] = b.process_open("2020-01-03", {"AAA": 150.0})
    assert f.quantity == 6 and o.status is OrderStatus.PARTIAL and o.quantity == 10  # the original is untouched
    assert rest is not o and rest.status is OrderStatus.CANCELLED
    assert (rest.order_id, rest.symbol, rest.quantity, rest.reference_price) == (o.order_id, "AAA", 4, 90.0)
    assert (rest.created_at, rest.reason) == (o.created_at, o.reason)
    assert why == "insufficient cash: filled 6 of 10, remainder 4 cancelled"
    assert b.pending == []


def test_full_fill_has_no_remainder():
    b = SimulatedBroker(10_000, CostModel(0, 0))
    o = order(qty=10, ref=100.0)
    b.submit(o)
    [f], cancels = b.process_open("2020-01-03", {"AAA": 100.0})
    assert f.quantity == 10 and o.status is OrderStatus.FILLED and cancels == []


def test_no_open_price_cancels():
    b = SimulatedBroker(1_000)
    b.submit(order(qty=1))
    fills, [(o, why)] = b.process_open("2020-01-03", {"AAA": float("nan")})
    assert fills == [] and "no tradable open" in why and b.pending == []


def test_state_round_trip():
    b = SimulatedBroker(5_000)
    b.positions = {"AAA": 3}
    b.submit(order(qty=2))
    restored = SimulatedBroker.from_state(b.to_state(), CostModel())
    assert restored.cash == b.cash and restored.positions == b.positions
    assert [o.order_id for o in restored.pending] == [o.order_id for o in b.pending]


def test_zero_cost_accounting_identity():
    """With no costs, equity changes by exactly the mark-to-market P&L."""
    b = SimulatedBroker(10_000, CostModel(0, 0))
    b.submit(order(qty=50, ref=100.0))
    b.process_open("2020-01-03", {"AAA": 100.0})
    assert b.snapshot({"AAA": 100.0}).equity == pytest.approx(10_000)
    assert b.snapshot({"AAA": 110.0}).equity == pytest.approx(10_500)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"commission_bps": math.nan},  # NaN costs would turn cash into NaN
        {"slippage_bps": math.inf},
        {"min_commission": -1.0},
        {"slippage_bps": True},
        {"commission_bps": "1"},
    ],
)
def test_cost_model_needs_finite_non_negative_numbers(kwargs):
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        CostModel(**kwargs)
