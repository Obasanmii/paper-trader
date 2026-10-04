"""What happens at the open, through the real TradingSession: the price collar,
partial-fill remainders, and the gain limit's bad-tick protection."""
import numpy as np
import pandas as pd

from papertrader.config import PortfolioConfig
from papertrader.core import Order, OrderStatus
from papertrader.data.market import MarketData
from papertrader.engine import run_backtest
from papertrader.engine.session import TradingSession
from papertrader.execution import CostModel, SimulatedBroker
from papertrader.journal import Journal
from papertrader.risk import KillSwitch, RiskLimits, RiskManager
from papertrader.strategies import build_strategy
from papertrader.strategies.base import StrategyOutput

DAYS = pd.bdate_range("2021-01-04", periods=3)


def market(opens: dict, closes: dict) -> MarketData:
    o, c = pd.DataFrame(opens, index=DAYS, dtype=float), pd.DataFrame(closes, index=DAYS, dtype=float)
    return MarketData(open=o, high=np.maximum(o, c), low=np.minimum(o, c), close=c, volume=c * 0 + 1e6)


def make_session(md, weights: dict, limits=None, portfolio=None, broker=None, journal=None):
    broker = broker if broker is not None else SimulatedBroker(100_000, CostModel(0, 0))
    output = StrategyOutput(pd.DataFrame(weights, index=DAYS, dtype=float))
    risk = RiskManager(limits or RiskLimits(), KillSwitch())
    return TradingSession(md, output, broker, risk, portfolio or PortfolioConfig(), journal, run_id="t")


def gap_session(limits=None, journal=None):
    """Hold AAA, then decide to swap it for BBB; both open 60% above the decision close."""
    md = market({"AAA": [100, 100, 160], "BBB": [100, 100, 160]}, {"AAA": [100, 100, 160], "BBB": [100, 100, 160]})
    return make_session(md, {"AAA": [0.2, 0.0, 0.0], "BBB": [0.0, 0.2, 0.2]}, limits=limits, journal=journal)


def test_collar_cancels_new_risk_on_a_gap_but_lets_the_exit_fill():
    journal = Journal(":memory:")
    s = gap_session(journal=journal)
    s.process_day(DAYS[0])
    s.process_day(DAYS[1])  # fills the AAA entry, queues sell AAA + buy BBB at the 100 close
    day = s.process_day(DAYS[2])

    assert [(f.symbol, f.quantity, f.price) for f in day.fills] == [("AAA", -196.0, 160.0)]  # the exit still fills
    [(order, why)] = day.cancelled
    assert order.symbol == "BBB" and order.quantity == 196 and order.status is OrderStatus.CANCELLED
    assert why == "open 160 is 60.0% away from the decision price 100 (limit 10%)"
    assert "BBB" not in s.broker.positions
    logged = journal.query("select symbol, quantity, detail from orders where event = 'cancelled'")
    assert logged.to_dict("records") == [{"symbol": "BBB", "quantity": 196.0, "detail": why}]


def test_without_the_collar_the_gapped_buy_would_have_filled():
    s = gap_session(limits=RiskLimits(max_price_deviation_pct=1.0))
    for d in DAYS:
        day = s.process_day(d)
    assert {f.symbol for f in day.fills} == {"AAA", "BBB"} and day.cancelled == []


def test_order_with_invalid_reference_price_is_cancelled_at_the_open():
    # e.g. a corrupted state file: the collar has nothing to compare the open with, so it fails closed
    bad = Order("AAA", 10, float("nan"), DAYS[0], order_id="bad-ref")
    broker = SimulatedBroker.from_state({"cash": 100_000, "pending": [bad.to_dict()]}, CostModel(0, 0))
    md = market({"AAA": [100, 100, 100]}, {"AAA": [100, 100, 100]})
    day = make_session(md, {"AAA": [0.0] * 3}, broker=broker).process_day(DAYS[0])
    [(order, why)] = day.cancelled
    assert order.order_id == "bad-ref" and "invalid reference price" in why
    assert day.fills == [] and broker.positions == {}


def test_buy_with_no_open_price_is_cancelled_by_the_collar():
    md = market({"AAA": [100, np.nan, 100]}, {"AAA": [100, 100, 100]})
    s = make_session(md, {"AAA": [0.2] * 3})
    s.process_day(DAYS[0])
    day = s.process_day(DAYS[1])
    [(order, why)] = day.cancelled
    assert order.symbol == "AAA" and why == "no tradable open price for AAA today"


def test_partial_fill_remainder_is_journaled_as_a_cancellation():
    journal = Journal(":memory:")
    md = market({"AAA": [100, 105, 105]}, {"AAA": [100, 105, 105]})  # +5%: inside the collar, but cash runs out
    s = make_session(
        md,
        {"AAA": [1.0] * 3},
        limits=RiskLimits(max_position_pct=1.0),
        portfolio=PortfolioConfig(cash_buffer_pct=0.0),
        journal=journal,
    )
    s.process_day(DAYS[0])  # buy 1000 at the 100 close
    day = s.process_day(DAYS[1])
    [fill] = day.fills
    [(rest, why)] = day.cancelled
    assert fill.quantity == 952 and rest.quantity == 48 and rest.order_id == fill.order_id
    assert why == "insufficient cash: filled 952 of 1000, remainder 48 cancelled"
    rows = journal.query("select event, quantity, detail from orders where order_id = ? order by rowid", (fill.order_id,))
    assert rows.to_dict("records") == [
        {"event": "approved", "quantity": 1000.0, "detail": ""},
        {"event": "cancelled", "quantity": 48.0, "detail": why},
    ]


def test_bad_tick_trips_the_gain_limit_and_flattening_ignores_the_collar(data):
    """A one-day close spike (x2.7) used to raise the peak, then read as a -63% crash the next day."""
    bad_day = data.dates[400]
    close = data.close.copy()
    close.loc[bad_day] *= 2.7
    spiked = MarketData(open=data.open, high=data.high, low=data.low, close=close, volume=data.volume)
    strategy = build_strategy("equal_weight")
    res = run_backtest(spiked, strategy, PortfolioConfig(), CostModel(), RiskLimits(), start=data.dates[252])

    [event] = res.risk_events  # the gain limit, not a fake drawdown the day after
    assert event.timestamp == bad_day and "daily gain" in event.detail and "bad data" in event.detail
    # flatten orders were priced at the bad close, so the real open is ~63% away; exits are exempt from the collar
    after = res.fills[res.fills["date"] > bad_day]
    assert len(after) > 0 and (after["quantity"] < 0).all()
    assert res.gross_exposure.iloc[-1] == 0
    assert res.equity.iloc[-1] > 0.9 * res.equity.loc[data.dates[399]]  # liquidated at real prices, not a crash
