import math

import pandas as pd
import pytest

from conftest import order, snapshot
from papertrader.config import ConfigError, config_from_dict
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
    for day, eq in zip(pd.bdate_range("2020-01-01", periods=5), [100, 110, 100, 95, 87], strict=True):
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


# --- the price collar at the open -------------------------------------------------------------


def test_collar_flags_new_risk_that_gapped_but_never_exits():
    rm = manager(max_price_deviation_pct=0.10)
    buy, sell = order("BBB", 10, ref=100.0), order("AAA", -50, ref=100.0)
    flagged = rm.check_open([buy, sell], {"AAA": 100}, {"AAA": 160.0, "BBB": 160.0})
    assert flagged == [(buy, "open 160 is 60.0% away from the decision price 100 (limit 10%)")]
    # inside the collar (either direction) is fine
    assert rm.check_open([order(qty=10, ref=100.0)], {}, {"AAA": 109.0}) == []
    assert rm.check_open([order(qty=10, ref=100.0)], {}, {"AAA": 91.0}) == []
    # gap down counts too: a new short that opens 30% lower is a different trade
    [(_, why)] = rm.check_open([order(qty=-10, ref=100.0)], {}, {"AAA": 70.0})
    assert "30.0% away" in why


def test_collar_fails_closed_on_bad_inputs():
    rm = manager()
    cases = [
        (order(ref=math.nan), {"AAA": 100.0}, "invalid reference price"),
        (order(ref=-1.0), {"AAA": 100.0}, "invalid reference price"),
        (order(), {}, "no tradable open price for AAA"),
        (order(), {"AAA": math.nan}, "no tradable open price for AAA"),
        (order(qty=math.nan), {"AAA": 100.0}, "invalid quantity"),
    ]
    for o, opens, expected in cases:
        [(flagged, why)] = rm.check_open([o], {}, opens)
        assert flagged is o and expected in why
    # an exit is never held hostage to bad data: the broker will sort out a missing open
    assert rm.check_open([order(qty=-10, ref=math.nan)], {"AAA": 10}, {}) == []


def test_collar_projects_the_batch_so_two_exits_cant_flip_a_position():
    rm = manager()
    first, second = order(qty=-60, ref=100.0), order(qty=-60, ref=100.0)  # each reduces 100, together they go short
    flagged = rm.check_open([first, second], {"AAA": 100}, {"AAA": 150.0})
    assert [o for o, _ in flagged] == [second]


# --- the gain limit -----------------------------------------------------------------------------


def test_implausible_gain_trips_and_is_not_trusted():
    rm = manager(max_daily_gain_pct=0.25)
    rm.end_of_day("2020-01-02", 100_000)
    [event] = rm.end_of_day("2020-01-03", 270_000)  # a bad tick, not a great day
    assert event.detail == "daily gain +170.00% exceeds the 25% limit: marks look like bad data"
    assert rm.kill_switch.tripped
    assert rm.peak_equity == 100_000 and rm.last_close_equity == 100_000
    # back to normal the next day: no fake -63% loss or drawdown measured from the bad mark
    assert rm.end_of_day("2020-01-06", 100_500) == []
    assert rm.peak_equity == 100_500 and rm.last_close_equity == 100_500
    assert not rm.gain_unconfirmed  # it was a bad tick: nothing left to confirm


def test_a_real_gain_becomes_the_base_once_a_person_resets_the_switch():
    """The gain limit keeps a close it doesn't believe out of the base. When the jump is real,
    every later close repeats it, so a reset used to be undone at the very next close, forever."""
    rm = manager(max_daily_gain_pct=0.25)
    rm.end_of_day("2020-01-02", 100_000)
    [event] = rm.end_of_day("2020-01-03", 136_000)
    assert event.kind == "kill_switch_tripped"
    assert rm.end_of_day("2020-01-06", 136_100) == []  # still tripped: nothing new, the base stays
    assert rm.last_close_equity == 100_000

    rm.kill_switch.reset(confirm=True)  # a person checked: a takeover, not a bad print
    [event] = rm.end_of_day("2020-01-07", 136_200)
    assert event.kind == "gain_accepted" and not rm.kill_switch.tripped
    assert event.detail == (
        "kill switch reset after the gain limit tripped: equity 136,200.00 accepted as the base (was 100,000.00)"
    )
    assert rm.last_close_equity == rm.peak_equity == 136_200 and not rm.gain_unconfirmed
    assert rm.end_of_day("2020-01-08", 136_300) == [] and not rm.kill_switch.tripped
    assert [e.kind for e in rm.events] == ["kill_switch_tripped", "gain_accepted"]


def test_a_bad_low_close_does_not_lock_the_account_out_after_it_recovers():
    """A bad LOW print trips the loss limit and becomes the base; the recovery then reads as an
    implausible gain, which used to re-trip after every reset."""
    rm = manager(max_daily_gain_pct=0.25, max_daily_loss_pct=0.05)
    rm.end_of_day("2020-01-02", 100_000)
    [event] = rm.end_of_day("2020-01-03", 76_250)
    assert event.detail.startswith("daily loss -23.75%") and rm.last_close_equity == 76_250
    assert rm.end_of_day("2020-01-06", 100_000) == []  # +31% from the bad base: not believed, already tripped
    rm.kill_switch.reset(confirm=True)
    [event] = rm.end_of_day("2020-01-07", 100_000)
    assert event.kind == "gain_accepted" and rm.last_close_equity == 100_000 and not rm.kill_switch.tripped
    assert rm.end_of_day("2020-01-08", 100_100) == [] and not rm.kill_switch.tripped


def test_accepting_a_new_base_still_checks_losses_and_drawdown():
    """A reset confirms the close the person looked at, so the accepted close is measured
    against that one (and it counts toward the peak), not against the stale pre-trip base."""
    rm = manager(max_daily_gain_pct=0.25, max_daily_loss_pct=0.05, max_drawdown_pct=0.40)
    rm.end_of_day("2020-01-02", 100_000)
    rm.end_of_day("2020-01-03", 140_000)
    rm.kill_switch.reset(confirm=True)
    accepted, loss = rm.end_of_day("2020-01-06", 94_000)
    assert accepted.kind == "gain_accepted"
    assert loss.detail == "loss -32.86% since the close confirmed by the last reset breached the 5% daily limit"
    assert rm.kill_switch.tripped and rm.last_close_equity == 94_000 and rm.peak_equity == 140_000

    rm = manager(max_daily_gain_pct=0.25, max_daily_loss_pct=0.5, max_drawdown_pct=0.20)
    rm.end_of_day("2020-01-02", 100_000)
    rm.end_of_day("2020-01-03", 90_000)  # peak 100k
    rm.end_of_day("2020-01-06", 120_000)  # +33%: not believed
    rm.kill_switch.reset(confirm=True)  # a person confirms 120k
    accepted, drawdown = rm.end_of_day("2020-01-07", 95_000)
    assert accepted.kind == "gain_accepted" and drawdown.detail.startswith("drawdown 20.83% breached")


def test_a_real_loss_right_after_a_confirmed_gain_is_caught():
    """Reviewer case G2: a -10% day straight after the reset used to be accepted unchecked."""
    rm = manager(max_daily_gain_pct=0.25, max_daily_loss_pct=0.05)
    rm.end_of_day("2020-01-02", 100_000)
    rm.end_of_day("2020-01-03", 136_000)
    rm.kill_switch.reset(confirm=True)
    accepted, loss = rm.end_of_day("2020-01-06", 122_400)
    assert accepted.kind == "gain_accepted" and loss.detail.startswith("loss -10.00% since the close confirmed")
    assert rm.kill_switch.tripped and rm.peak_equity == 136_000


def test_a_bad_print_after_a_confirmed_gain_is_held_back_again():
    """Reviewer case G3: a bad 200k print right after the reset used to become base AND peak,
    so the next real close tripped the drawdown limit after every later reset."""
    rm = manager(max_daily_gain_pct=0.25, max_daily_loss_pct=0.05, max_drawdown_pct=0.25)
    rm.end_of_day("2020-01-02", 100_000)
    rm.end_of_day("2020-01-03", 136_000)
    rm.kill_switch.reset(confirm=True)
    [event] = rm.end_of_day("2020-01-06", 200_000)  # +47% on the confirmed 136k: not believed either
    assert event.detail.startswith("daily gain +47.06% since the close confirmed by the last reset")
    assert rm.gain_unconfirmed and rm.last_close_equity == 100_000 and rm.peak_equity == 100_000
    assert rm.end_of_day("2020-01-07", 136_000) == []  # back to the real level while still tripped
    rm.kill_switch.reset(confirm=True)  # a person confirms 136k again
    [event] = rm.end_of_day("2020-01-08", 136_200)
    assert event.kind == "gain_accepted" and not rm.kill_switch.tripped
    assert rm.last_close_equity == 136_200 and rm.peak_equity == 136_200
    for day, eq in (("2020-01-09", 136_300), ("2020-01-10", 136_100)):
        assert rm.end_of_day(day, eq) == [] and not rm.kill_switch.tripped


def test_a_bad_print_while_still_tripped_is_not_what_the_reset_confirms():
    """The reset confirms the close a person looked at. A second, implausible jump that lands
    while the switch is still on (cron runs every evening) must not take its place, or the
    next real close reads as a crash from it and the bad print becomes the peak."""
    rm = manager(max_daily_gain_pct=0.25, max_daily_loss_pct=0.05, max_drawdown_pct=0.25)
    rm.end_of_day("2020-01-02", 100_000)
    rm.end_of_day("2020-01-03", 136_000)  # real jump: held back, the switch trips
    assert rm.end_of_day("2020-01-06", 200_000) == []  # bad print while tripped: +47% on the held-back close
    assert rm.unconfirmed_equity == 136_000
    rm.kill_switch.reset(confirm=True)
    [event] = rm.end_of_day("2020-01-07", 136_100)
    assert event.kind == "gain_accepted" and rm.peak_equity == 136_100 and not rm.kill_switch.tripped
    for day, eq in (("2020-01-08", 136_200), ("2020-01-09", 136_000)):
        assert rm.end_of_day(day, eq) == [] and not rm.kill_switch.tripped


def test_the_gain_flag_survives_a_restart():
    rm = manager(max_daily_gain_pct=0.25)
    rm.end_of_day("2020-01-02", 100_000)
    rm.end_of_day("2020-01-03", 136_000)
    state = rm.state()
    assert state["gain_unconfirmed"] is True and state["unconfirmed_equity"] == 136_000
    restarted = RiskManager(RiskLimits(max_daily_gain_pct=0.25), KillSwitch())  # reset: a fresh switch
    restarted.load_state(state)
    [event] = restarted.end_of_day("2020-01-06", 136_100)
    assert event.kind == "gain_accepted" and restarted.last_close_equity == 136_100
    for junk in ("true", 1, None):
        older = RiskManager(RiskLimits(), KillSwitch())
        older.load_state({**state, "gain_unconfirmed": junk})
        assert older.gain_unconfirmed is False  # only a real true re-bases: anything else checks gains as usual
    # State from before unconfirmed_equity existed: fail closed, but say why a second reset is needed.
    legacy = RiskManager(RiskLimits(max_daily_gain_pct=0.25), KillSwitch())
    legacy.load_state({k: v for k, v in state.items() if k != "unconfirmed_equity"})
    [event] = legacy.end_of_day("2020-01-06", 136_100)
    assert event.kind == "kill_switch_tripped" and "the close the reset confirmed isn't recorded" in event.detail


def test_gain_limit_can_be_disabled_and_ordinary_gains_pass():
    rm = manager(max_daily_gain_pct=None)
    rm.end_of_day("2020-01-02", 100_000)
    assert rm.end_of_day("2020-01-03", 270_000) == [] and not rm.kill_switch.tripped
    assert rm.peak_equity == 270_000
    rm = manager()
    rm.end_of_day("2020-01-02", 100_000)
    assert rm.end_of_day("2020-01-03", 124_000) == [] and not rm.kill_switch.tripped


# --- the daily order count --------------------------------------------------------------------


def test_rejected_orders_count_toward_the_daily_limit():
    rm = manager(max_orders_per_day=2, max_position_pct=0.10)
    rm.start_day("2020-01-02")
    results = check(rm, [order(qty=500), order(qty=500)], snapshot())  # 50% positions: both rejected
    assert not any(d.approved for _, d in results) and rm.orders_today == 2
    [(_, d)] = check(rm, [order(qty=1)], snapshot())  # a fine order, but the loop has used its budget
    assert not d.approved and "daily order limit reached (2)" in d.reasons
    # exits are never counted, and never blocked by the count
    [(_, d)] = check(rm, [order(qty=-10)], snapshot(positions={"AAA": 50}))
    assert d.approved and d.reducing and rm.orders_today == 3


# --- limits built directly in Python ------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"allow_short": "false"}, "allow_short must be True or False"),  # a truthy string would enable shorting
        ({"allow_short": 1}, "allow_short"),
        ({"flatten_on_kill": "no"}, "flatten_on_kill"),
        ({"max_orders_per_day": True}, "max_orders_per_day"),
        ({"max_orders_per_day": 2.5}, "max_orders_per_day"),
        ({"max_orders_per_day": -1}, "max_orders_per_day"),
        ({"max_data_age_days": "4"}, "max_data_age_days"),
        ({"max_data_age_days": False}, "max_data_age_days"),
        ({"symbol_whitelist": "SPY"}, "symbol_whitelist"),  # would whitelist S, P and Y
        ({"symbol_whitelist": ("SPY", 1)}, "symbol_whitelist"),
        ({"max_position_pct": math.nan}, "max_position_pct"),  # NaN compares False: would never fire
        ({"max_order_notional": math.inf}, "max_order_notional"),
        ({"max_order_notional": "1e5"}, "max_order_notional"),
        ({"max_daily_gain_pct": 0}, "max_daily_gain_pct"),
        ({"max_daily_gain_pct": -0.1}, "max_daily_gain_pct"),
        ({"max_daily_gain_pct": math.nan}, "max_daily_gain_pct"),
        ({"max_drawdown_pct": 25}, "fractions"),
    ],
)
def test_risk_limits_validate_types_and_ranges(kwargs, match):
    with pytest.raises(ValueError, match=match):
        RiskLimits(**kwargs)


def test_risk_limits_accept_valid_values():
    limits = RiskLimits(symbol_whitelist=["SPY"], max_data_age_days=None, max_daily_gain_pct=None, max_orders_per_day=0)
    assert limits.symbol_whitelist == ("SPY",) and limits.max_daily_gain_pct is None
    assert RiskLimits(max_daily_gain_pct=1.5).max_daily_gain_pct == 1.5  # gains above 100% are a valid limit


def test_gain_limit_in_yaml_is_validated():
    assert config_from_dict({"risk": {"max_daily_gain_pct": None}}).risk.max_daily_gain_pct is None
    with pytest.raises(ConfigError, match="max_daily_gain_pct"):
        config_from_dict({"risk": {"max_daily_gain_pct": -0.25}})
