"""Hard risk limits. They live in their own config section and are enforced
outside the strategy, so a buggy or over-eager strategy can't loosen them."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskLimits:
    # Per-order and per-portfolio limits (checked on every order that adds exposure)
    max_position_pct: float = 0.25  # |position value| / equity, per symbol, after the trade
    max_gross_exposure_pct: float = 1.0  # sum of |position values| / equity, after the trade
    max_order_notional: float = 100_000.0  # fat-finger limit in account currency
    max_orders_per_day: int = 50  # runaway-loop protection
    allow_short: bool = False
    max_price_deviation_pct: float = 0.10  # order's reference price vs latest market price
    max_data_age_days: int | None = 4  # live only: refuse new risk on stale data
    symbol_whitelist: tuple[str, ...] | None = None  # None = any symbol in the data

    # Kill-switch triggers (checked at every close)
    max_daily_loss_pct: float = 0.05
    max_drawdown_pct: float = 0.25
    flatten_on_kill: bool = True  # close all positions once the kill switch trips

    def __post_init__(self):
        for name in (
            "max_position_pct",
            "max_gross_exposure_pct",
            "max_price_deviation_pct",
            "max_daily_loss_pct",
            "max_drawdown_pct",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"risk.{name} must be a positive number, got {value!r}")
        if self.max_daily_loss_pct >= 1 or self.max_drawdown_pct >= 1:
            raise ValueError("loss limits are fractions: 0.05 means 5%")
        if self.max_order_notional <= 0:
            raise ValueError("risk.max_order_notional must be positive")
        if self.max_orders_per_day < 0:
            raise ValueError("risk.max_orders_per_day can't be negative")
        if self.max_data_age_days is not None and self.max_data_age_days < 0:
            raise ValueError("risk.max_data_age_days can't be negative")
        if self.symbol_whitelist is not None:
            object.__setattr__(self, "symbol_whitelist", tuple(self.symbol_whitelist))
