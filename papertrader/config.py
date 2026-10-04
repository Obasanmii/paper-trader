"""Configuration: one YAML file per setup, parsed strictly.

Unknown keys are errors, not warnings. A typo in a risk limit that gets
silently ignored ("max_postion_pct: 0.1") is exactly the kind of failure
this project exists to prevent. Values are checked against each field's
type annotation for the same reason: `allow_short: "false"` is a non-empty
string, which is truthy, so accepting it would quietly enable shorting.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import math
import numbers
import re
import types
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

from papertrader.execution.paper_broker import CostModel
from papertrader.risk.limits import RiskLimits


class ConfigError(ValueError):
    pass


def _is_finite_number(value) -> bool:
    """bool is an int subclass, and math.isfinite overflows on huge ints."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _check(section: str, obj, name: str, rule: str, ok: Callable[[float], bool] | None = None, *, integer: bool = False):
    """Range-check a numeric field. Configs are also built directly in Python, so a
    wrong type must still be a ConfigError rather than a TypeError from the comparison."""
    value = getattr(obj, name)
    kind = numbers.Integral if integer else numbers.Real
    if not (_is_finite_number(value) and isinstance(value, kind)) or (ok and not ok(value)):
        raise ConfigError(f"{section}.{name} must be {rule}, got {value!r}")


def _check_text(section: str, obj, name: str) -> None:
    value = getattr(obj, name)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{section}.{name} must be a non-empty string, got {value!r}")


def _check_date(section: str, obj, name: str) -> dt.date | None:
    """Strict YYYY-MM-DD: a split date that silently parses as something else
    moves the in-sample/out-of-sample boundary."""
    value = getattr(obj, name)
    if value is None:
        return None
    try:
        parsed = dt.date.fromisoformat(value)
    except (TypeError, ValueError):
        parsed = None
    if parsed is None or parsed.isoformat() != value:
        raise ConfigError(f"{section}.{name} must be a date like 2020-01-31, got {value!r}")
    return parsed


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

    def __post_init__(self):
        s = "data.synthetic"
        _check(s, self, "n_days", "an integer >= 2", lambda v: v >= 2, integer=True)
        _check(s, self, "seed", "an integer >= 0", lambda v: v >= 0, integer=True)  # numpy rejects negative seeds
        _check(s, self, "drift", "a finite number")
        _check(s, self, "vol", "a positive number", lambda v: v > 0)
        _check(s, self, "market_corr", "in [0, 1)", lambda v: 0 <= v < 1)
        _check(s, self, "regime_drift", "a number >= 0", lambda v: v >= 0)
        _check(s, self, "regime_persistence", "in [0, 1]", lambda v: 0 <= v <= 1)
        if self.t_dof is not None:  # <= 2 has infinite variance, so it can't be scaled to `vol`
            _check(s, self, "t_dof", "null or a number > 2", lambda v: v > 2)
        _check_date(s, self, "start")


@dataclass(frozen=True)
class CleaningConfig:
    spike_threshold: float = 0.25
    stale_run: int = 5
    max_ffill_days: int = 3

    def __post_init__(self):
        s = "data.cleaning"
        _check(s, self, "spike_threshold", "a positive number", lambda v: v > 0)
        _check(s, self, "stale_run", "an integer >= 2", lambda v: v >= 2, integer=True)  # 1 flags every repeated close
        _check(s, self, "max_ffill_days", "an integer >= 0", lambda v: v >= 0, integer=True)  # 0: never forward-fill


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
        # A bare string would be iterated as one-letter tickers.
        if not isinstance(self.symbols, (tuple, list)) or not all(isinstance(s, str) and s.strip() for s in self.symbols):
            raise ConfigError(f"data.symbols must be a list of non-empty strings, got {self.symbols!r}")
        if not self.symbols:
            raise ConfigError("data.symbols is empty")
        if len(set(self.symbols)) != len(self.symbols):
            raise ConfigError("data.symbols has duplicates")
        start, end = _check_date("data", self, "start"), _check_date("data", self, "end")
        if start and end and start >= end:
            raise ConfigError(f"data.start ({start}) must be before data.end ({end})")


@dataclass(frozen=True)
class StrategyConfig:
    name: str = "trend"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        _check_text("strategy", self, "name")
        if not isinstance(self.params, dict):
            raise ConfigError(f"strategy.params must be a mapping, got {self.params!r}")


@dataclass(frozen=True)
class PortfolioConfig:
    initial_cash: float = 100_000.0
    cash_buffer_pct: float = 0.02  # keep a little cash so gaps at the open don't force partial fills
    rebalance_band: float = 0.02  # ignore weight drifts smaller than this (cuts pointless turnover)
    min_trade_notional: float = 100.0
    allow_fractional: bool = False

    def __post_init__(self):
        s = "portfolio"
        _check(s, self, "initial_cash", "a positive number", lambda v: v > 0)
        _check(s, self, "cash_buffer_pct", "in [0, 1)", lambda v: 0 <= v < 1)
        _check(s, self, "rebalance_band", "in [0, 1)", lambda v: 0 <= v < 1)  # negative = trade on every drift
        _check(s, self, "min_trade_notional", "a number >= 0", lambda v: v >= 0)


@dataclass(frozen=True)
class ResearchConfig:
    split_date: str | None = None  # in-sample ends here; everything after is out-of-sample
    warmup_days: int = 252  # every strategy is judged from the same start date
    param_grid: dict = field(default_factory=dict)
    benchmark: str = "equal_weight"
    bootstrap_samples: int = 2000
    timing_permutations: int = 500
    ledger_path: str | None = "state/research_ledger.sqlite"  # every trial ever run; null = remember nothing

    def __post_init__(self):
        s = "research"
        _check(s, self, "warmup_days", "an integer >= 0", lambda v: v >= 0, integer=True)
        _check(s, self, "bootstrap_samples", "an integer >= 1", lambda v: v >= 1, integer=True)
        _check(s, self, "timing_permutations", "an integer >= 1", lambda v: v >= 1, integer=True)
        _check_date(s, self, "split_date")
        if self.ledger_path is not None:  # "" would be a temporary database: a ledger that forgets
            _check_text(s, self, "ledger_path")
        if not isinstance(self.param_grid, dict):
            raise ConfigError(f"research.param_grid must be a mapping, got {self.param_grid!r}")
        for key, values in self.param_grid.items():
            # The sweep iterates each value list: a bare 50 crashes it, and a quoted '99' would be tried as 9 twice.
            if not isinstance(key, str) or not key.strip():
                raise ConfigError(f"research.param_grid keys must be parameter names, got {key!r}")
            if not isinstance(values, (list, tuple)) or not values:
                raise ConfigError(f"research.param_grid.{key} must be a non-empty list of values, got {values!r}")


@dataclass(frozen=True)
class PaperConfig:
    state_dir: str = "state/default"
    journal_path: str | None = None  # default: <state_dir>/journal.sqlite
    max_catchup_days: int = 5
    # A live run (no --as-of) before close + buffer, on the exchange's clock, ignores
    # today's bar: during the session a feed serves the in-progress bar as if it were final.
    market_timezone: str = "America/New_York"
    market_close: str = "16:00"  # HH:MM in market_timezone; quote it in YAML, which reads 16:00 as 960
    close_buffer_minutes: int = 15  # time for the feed to publish the final bar

    def __post_init__(self):
        _check_text("paper", self, "state_dir")
        # 0 or less would refuse every catch-up, pushing people to run with --force by habit.
        _check("paper", self, "max_catchup_days", "an integer >= 1", lambda v: v >= 1, integer=True)
        _check_text("paper", self, "market_timezone")
        try:
            ZoneInfo(self.market_timezone)
        except (ZoneInfoNotFoundError, ValueError, OSError):
            tz = self.market_timezone
            raise ConfigError(f"paper.market_timezone must be an IANA time zone like America/New_York, got {tz!r}") from None
        if not isinstance(self.market_close, str) or not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", self.market_close):
            raise ConfigError(f"paper.market_close must be a time like 16:00 (HH:MM), got {self.market_close!r}")
        _check("paper", self, "close_buffer_minutes", "an integer >= 0", lambda v: v >= 0, integer=True)


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


def _plain(value):
    """YAML turns 2021-12-31 into a date object; keep dates as ISO strings."""
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    return value


def _coerce(tp, value, path: str):
    """Check one YAML value against its field annotation. Dataclasses don't check
    types and YAML is loose ("false" is a string, .nan is a float), so without this a
    wrong type sails through and is later read as truthy, or compares as False."""
    if dataclasses.is_dataclass(tp):
        return _build(tp, value, path)
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        options = [a for a in args if a is not type(None)]
        return _coerce(options[0], value, path) if len(options) == 1 else value
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected true or false, got {value!r}")
    elif tp is int:
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
    elif tp is float:
        if not _is_finite_number(value):
            raise ConfigError(f"{path}: expected a finite number, got {value!r}")
        value = float(value)
    elif tp is str:
        if not isinstance(value, str):
            # YAML reads ON/NO as booleans and 0700 as octal 448; never guess the text back.
            hint = " (quote it in the YAML)" if isinstance(value, (bool, int, float)) else ""
            raise ConfigError(f"{path}: expected a string, got {value!r}{hint}")
    elif tp is dict or origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
    elif origin is tuple and len(args) == 2 and args[1] is Ellipsis:
        if isinstance(value, str) or not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {value!r}")
        value = tuple(_coerce(args[0], item, f"{path}[{i}]") for i, item in enumerate(value))
    return value


def _build(cls, raw, path: str):
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping, got {type(raw).__name__}")
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - known, key=str)
    if unknown:
        raise ConfigError(f"{path}: unknown key(s) {unknown}. Allowed: {sorted(known)}")
    hints = typing.get_type_hints(cls)  # resolves string annotations in cls's own module
    kwargs = {key: _coerce(hints[key], _plain(value), f"{path}.{key}") for key, value in raw.items()}
    try:
        return cls(**kwargs)
    except ConfigError:
        raise
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def config_from_dict(raw: dict) -> AppConfig:
    return _build(AppConfig, raw, "config")


class _StrictLoader(yaml.SafeLoader):
    """yaml.safe_load keeps the last of a repeated key, so a second `risk:` block silently
    replaces the first, limits and whitelist included, with defaults that may be looser."""

    def construct_mapping(self, node, deep=False):
        seen: dict = {}  # key -> line it first appeared on
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue  # `<<: *anchor` merges on purpose; explicit keys may override it
            key = self.construct_object(key_node, deep=True)
            line = key_node.start_mark.line + 1
            try:
                first = seen.get(key)
            except TypeError:
                continue  # unhashable: the base constructor reports it
            if first is not None:
                raise ConfigError(f"{key_node.start_mark.name}: duplicate key {key!r} on line {line} (first on line {first})")
            seen[key] = line
        return super().construct_mapping(node, deep=deep)


def load_config(path: str | Path) -> AppConfig:
    with open(path) as fh:
        try:
            raw = yaml.load(fh, Loader=_StrictLoader) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    return config_from_dict(raw)


def config_to_dict(cfg) -> dict:
    return dataclasses.asdict(cfg)
