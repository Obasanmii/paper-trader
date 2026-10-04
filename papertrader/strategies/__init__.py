"""Strategies. Importing this package registers every built-in strategy.

To add one: subclass Strategy, decorate it with @register, import its module
below, and run `pytest tests/test_no_lookahead.py`. The lookahead test runs
against every registered strategy automatically.
"""
from papertrader.strategies import benchmark, mean_reversion, ml, trend  # noqa: F401  (registration)
from papertrader.strategies.base import (
    REGISTRY,
    Strategy,
    StrategyOutput,
    build_strategy,
    lookahead_violations,
    register,
)

__all__ = ["REGISTRY", "Strategy", "StrategyOutput", "build_strategy", "lookahead_violations", "register"]
