"""Hard risk limits. They live in their own config section and are enforced
outside the strategy, so a buggy or over-eager strategy can't loosen them."""
from __future__ import annotations

import numbers
from dataclasses import dataclass

from papertrader.core import is_finite_number


@dataclass(frozen=True)
class RiskLimits:
    # Per-order and per-portfolio limits (checked on every order that adds exposure)
    max_position_pct: float = 0.25  # |position value| / equity, per symbol, after the trade
    max_gross_exposure_pct: float = 1.0  # sum of |position values| / equity, after the trade
    max_order_notional: float = 100_000.0  # fat-finger limit in account currency
    max_orders_per_day: int = 50  # runaway-loop protection: counts every new-risk order checked, approved or not
    allow_short: bool = False
    max_price_deviation_pct: float = 0.10  # collar: cancel at the open if it gapped this far from the decision price
    max_data_age_days: int | None = 4  # live only: refuse new risk on stale data
    symbol_whitelist: tuple[str, ...] | None = None  # None = any symbol in the data

    # Kill-switch triggers (checked at every close)
    max_daily_loss_pct: float = 0.05
    max_drawdown_pct: float = 0.25
    max_daily_gain_pct: float | None = 0.25  # a jump this big is more likely bad marks than skill; None = off
    flatten_on_kill: bool = True  # close all positions once the kill switch trips

    def __post_init__(self):
        # Limits are also built directly in Python, where nothing checks types: a NaN
        # limit never fires, and "false" is truthy, so it would quietly enable shorting.
        for name in (
            "max_position_pct",
            "max_gross_exposure_pct",
            "max_order_notional",
            "max_price_deviation_pct",
            "max_daily_loss_pct",
            "max_drawdown_pct",
        ):
            _require_positive(name, getattr(self, name))
        if self.max_daily_gain_pct is not None:
            _require_positive("max_daily_gain_pct", self.max_daily_gain_pct)
        if self.max_daily_loss_pct >= 1 or self.max_drawdown_pct >= 1:
            raise ValueError("loss limits are fractions: 0.05 means 5%")
        for name in ("allow_short", "flatten_on_kill"):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise ValueError(f"risk.{name} must be True or False, got {value!r}")
        if not _is_count(self.max_orders_per_day):
            raise ValueError(f"risk.max_orders_per_day must be an integer >= 0, got {self.max_orders_per_day!r}")
        if self.max_data_age_days is not None and not _is_count(self.max_data_age_days):
            raise ValueError(f"risk.max_data_age_days must be None or an integer >= 0, got {self.max_data_age_days!r}")
        if self.symbol_whitelist is not None:
            # A bare string would be iterated as one-letter symbols.
            listed = self.symbol_whitelist
            if not isinstance(listed, (tuple, list)) or not all(isinstance(s, str) for s in listed):
                raise ValueError(f"risk.symbol_whitelist must be None or a list of symbol strings, got {listed!r}")
            object.__setattr__(self, "symbol_whitelist", tuple(listed))


def _require_positive(name: str, value) -> None:
    if not is_finite_number(value) or value <= 0:
        raise ValueError(f"risk.{name} must be a positive number, got {value!r}")


def _is_count(value) -> bool:
    """A non-negative integer. bool is an int subclass, and 2.5 orders is a typo, not a limit."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool) and value >= 0
