from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from papertrader.config import AppConfig, DataConfig, PaperConfig, SyntheticConfig
from papertrader.core import Order, PortfolioSnapshot
from papertrader.data import load_market_data


def make_config(tmp_path=None, n_days: int = 700, regime_drift: float = 0.0, **overrides) -> AppConfig:
    cfg = AppConfig(
        name="test",
        data=DataConfig(synthetic=SyntheticConfig(n_days=n_days, regime_drift=regime_drift, seed=11)),
    )
    if tmp_path is not None:
        cfg = dataclasses.replace(cfg, paper=PaperConfig(state_dir=str(tmp_path / "state")))
    return dataclasses.replace(cfg, **overrides) if overrides else cfg


@pytest.fixture
def cfg(tmp_path):
    return make_config(tmp_path)


@pytest.fixture(scope="session")
def data():
    md, _ = load_market_data(make_config().data)
    return md


def order(symbol="AAA", qty=10.0, ref=100.0, when="2020-01-02"):
    return Order(symbol, qty, ref, pd.Timestamp(when))


def snapshot(cash=100_000.0, positions=None, prices=None):
    return PortfolioSnapshot(cash=cash, positions=dict(positions or {}), prices=dict(prices or {"AAA": 100.0, "BBB": 50.0}))
