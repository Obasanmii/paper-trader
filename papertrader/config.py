"""Configuration: one YAML file per setup, parsed strictly.

Unknown keys are errors, not warnings. A typo in a risk limit that gets
silently ignored ("max_postion_pct: 0.1") is exactly the kind of failure
this project exists to prevent.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from papertrader.execution.paper_broker import CostModel
from papertrader.risk.limits import RiskLimits


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class SyntheticConfig:
    n_days: int = 2520
    start: str = "2015-01-02"
    seed: int = 7
    drift: float = 0.0
    vol: float = 0.20
    market_corr: float = 0.5
    regime_drift: float = 0.0
    regime_persistence: float = 0.995
    t_dof: float | None = 5.0
    inject_errors: bool = False


@dataclass(frozen=True)
class CleaningConfig:
    spike_threshold: float = 0.25
    stale_run: int = 5
    max_ffill_days: int = 3


@dataclass(frozen=True)
class DataConfig:
    source: str = "synthetic"  # synthetic | csv | yfinance
    symbols: tuple[str, ...] = ("AAA", "BBB", "CCC", "DDD", "EEE")
    start: str | None = None
    end: str | None = None
    csv_dir: str = "data/raw"
    use_adjusted: bool = True
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)

    def __post_init__(self):
        if self.source not in ("synthetic", "csv", "yfinance"):
            raise ConfigError(f"data.source must be synthetic, csv or yfinance, got {self.source!r}")
        if not self.symbols:
            raise ConfigError("data.symbols is empty")
        if len(set(self.symbols)) != len(self.symbols):
            raise ConfigError("data.symbols has duplicates")


@dataclass(frozen=True)
class StrategyConfig:
    name: str = "trend"
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PortfolioConfig:
    initial_cash: float = 100_000.0
    cash_buffer_pct: float = 0.02  # keep a little cash so gaps at the open don't force partial fills
    rebalance_band: float = 0.02  # ignore weight drifts smaller than this (cuts pointless turnover)
    min_trade_notional: float = 100.0
    allow_fractional: bool = False

    def __post_init__(self):
        if self.initial_cash <= 0:
            raise ConfigError("portfolio.initial_cash must be positive")
        if not 0 <= self.cash_buffer_pct < 1:
            raise ConfigError("portfolio.cash_buffer_pct must be in [0, 1)")


@dataclass(frozen=True)
class ResearchConfig:
    split_date: str | None = None  # in-sample ends here; everything after is out-of-sample
    warmup_days: int = 252  # every strategy is judged from the same start date
    param_grid: dict = field(default_factory=dict)
    benchmark: str = "equal_weight"
    bootstrap_samples: int = 2000
    timing_permutations: int = 500


@dataclass(frozen=True)
class PaperConfig:
    state_dir: str = "state/default"
    journal_path: str | None = None  # default: <state_dir>/journal.sqlite
    max_catchup_days: int = 5


@dataclass(frozen=True)
class AppConfig:
    name: str = "default"
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    costs: CostModel = field(default_factory=CostModel)
    risk: RiskLimits = field(default_factory=RiskLimits)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    paper: PaperConfig = field(default_factory=PaperConfig)


_NESTED = {
    AppConfig: {
        "data": DataConfig,
        "strategy": StrategyConfig,
        "portfolio": PortfolioConfig,
        "costs": CostModel,
        "risk": RiskLimits,
        "research": ResearchConfig,
        "paper": PaperConfig,
    },
    DataConfig: {"synthetic": SyntheticConfig, "cleaning": CleaningConfig},
}
_TUPLE_FIELDS = {"symbols", "symbol_whitelist"}


def _plain(value):
    """YAML turns 2021-12-31 into a date object; keep dates as ISO strings."""
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    return value


def _build(cls, raw, path: str):
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping, got {type(raw).__name__}")
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(f"{path}: unknown key(s) {unknown}. Allowed: {sorted(known)}")
    kwargs = {}
    for key, value in raw.items():
        nested = _NESTED.get(cls, {}).get(key)
        if nested is not None:
            kwargs[key] = _build(nested, value, f"{path}.{key}")
        elif key in _TUPLE_FIELDS and value is not None:
            if isinstance(value, str) or not isinstance(value, (list, tuple)):
                raise ConfigError(f"{path}.{key}: expected a list")
            kwargs[key] = tuple(str(v) for v in value)
        else:
            kwargs[key] = _plain(value)
    try:
        return cls(**kwargs)
    except ConfigError:
        raise
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def config_from_dict(raw: dict) -> AppConfig:
    return _build(AppConfig, raw, "config")


def load_config(path: str | Path) -> AppConfig:
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    return config_from_dict(raw)


def config_to_dict(cfg) -> dict:
    return dataclasses.asdict(cfg)
