import math

import pandas as pd

from conftest import order, snapshot
from papertrader.risk import KillSwitch, RiskLimits, RiskManager, is_reducing


def manager(**limits):
    return RiskManager(RiskLimits(**limits), KillSwitch())


def check(rm, orders, snap, **kw):
    return rm.check_orders(orders, snap, **kw)


def test_reasonable_order_is_approved():
    [(o, d)] = check(manager(), [order(qty=100)], snapshot())
    assert d.approved and d.reasons == []


def test_position_limit_rejects():
    [(_, d)] = check(manager(max_position_pct=0.10), [order(qty=200)], snapshot())  # 20% of equity
    assert not d.approved
    assert "position would be 20.0%" in d.reasons[0]


def test_batch_cannot_jointly_breach_gross_exposure():
    rm = manager(max_position_pct=0.6, max_gross_exposure_pct=1.0)
    orders = [order("AAA", 500), order("BBB", 1000, ref=50.0)]  # 50% + 50%: fine together...
    assert all(d.approved for _, d in check(rm, orders, snapshot()))
    rm = manager(max_position_pct=0.6, max_gross_exposure_pct=1.0)
    orders = [order("AAA", 550), order("BBB", 1000, ref=50.0)]  # 55% + 50%: second must fail
    results = check(rm, orders, snapshot())
    assert results[0][1].approved
    assert not results[1][1].approved and "gross exposure" in results[1][1].reasons[0]


def test_shorting_blocked_unless_allowed():
    [(_, d)] = check(manager(), [order(qty=-10)], snapshot())
    assert not d.approved and "short selling is disabled" in d.reasons
    [(_, d)] = check(manager(allow_short=True), [order(qty=-10)], snapshot())
    assert d.approved


def test_fat_finger_and_price_deviation():
    [(_, d)] = check(manager(max_order_notional=5_000, max_position_pct=1.0), [order(qty=100)], snapshot())
    assert not d.approved and "order notional" in d.reasons[0]
    [(_, d)] = check(manager(), [order(qty=10, ref=130.0)], snapshot())  # decided on a stale/bad price
    assert not d.approved and "away from market" in d.reasons[0]


def test_invalid_inputs_fail_closed():
    for bad in (order(qty=math.nan), order(qty=0), order(ref=-1.0), order(symbol="ZZZ")):
        [(_, d)] = check(manager(), [bad], snapshot())
        assert not d.approved
    # a held position with no mark means we can't value the book: no new risk
    snap = snapshot(positions={"CCC": 10}, prices={"AAA": 100.0})
    [(_, d)] = check(manager(), [order(qty=1)], snap)
    assert not d.approved and "no marks" in d.reasons[0]


def test_daily_order_limit_resets_each_day():
    rm = manager(max_orders_per_day=2, max_position_pct=1.0)
    rm.start_day("2020-01-02")
    results = check(rm, [order(qty=1), order(qty=1), order(qty=1)], snapshot())
    assert [d.approved for _, d in results] == [True, True, False]
    rm.start_day("2020-01-03")
    assert check(rm, [order(qty=1)], snapshot())[0][1].approved


def test_whitelist_and_stale_data():
    [(_, d)] = check(manager(symbol_whitelist=("BBB",)), [order("AAA", 1)], snapshot())
    assert not d.approved and "whitelist" in d.reasons[0]
    [(_, d)] = check(manager(max_data_age_days=3), [order(qty=1)], snapshot(), now="2020-01-10", data_as_of="2020-01-02")
    assert not d.approved and "days old" in d.reasons[0]


def test_kill_switch_blocks_new_risk_but_allows_exits():
    rm = manager()
    rm.kill_switch.trip("test")
    snap = snapshot(positions={"AAA": 100})
    results = dict((o.quantity, d) for o, d in check(rm, [order(qty=10), order(qty=-100)], snap))
    assert not results[10].approved and "kill switch" in results[10].reasons[0]
    assert results[-100].approved and results[-100].reducing
    # flipping through zero is not "reducing"
    [(_, d)] = check(rm, [order(qty=-150)], snap)
    assert not d.approved


def test_loss_limits_trip_the_kill_switch():
    rm = manager(max_daily_loss_pct=0.05, max_drawdown_pct=0.20)
    assert rm.end_of_day("2020-01-02", 100_000) == []
    assert rm.end_of_day("2020-01-03", 97_000) == []
    [event] = rm.end_of_day("2020-01-06", 91_000)  # -6.2% in a day
    assert "daily loss" in event.detail and rm.kill_switch.tripped

    rm = manager(max_daily_loss_pct=0.5, max_drawdown_pct=0.20)
    for day, eq in zip(pd.bdate_range("2020-01-01", periods=5), [100, 110, 100, 95, 87]):
        events = rm.end_of_day(day, eq)
    assert rm.kill_switch.tripped and "drawdown" in events[0].detail  # 87 is -20.9% from 110


def test_is_reducing():
    assert is_reducing(100, -40) and is_reducing(100, -100) and is_reducing(-50, 20)
    assert not is_reducing(100, 10) and not is_reducing(100, -150) and not is_reducing(0, -1)
    assert not is_reducing(100, math.nan) and not is_reducing(100, "junk")


def test_flatten_orders_close_everything():
    rm = manager()
    orders = rm.flatten_orders(snapshot(positions={"AAA": 10, "BBB": -5}), "2020-01-02")
    assert {(o.symbol, o.quantity) for o in orders} == {("AAA", -10), ("BBB", 5)}


def test_risk_layer_does_not_import_strategies():
    import pathlib
    import re

    risk_dir = pathlib.Path(__file__).parents[1] / "papertrader" / "risk"
    pattern = re.compile(r"^\s*(from|import)\s+papertrader\.(strategies|engine|research)", re.MULTILINE)
    for path in risk_dir.glob("*.py"):
        assert not pattern.search(path.read_text()), f"{path.name} must stay independent of strategy code"
