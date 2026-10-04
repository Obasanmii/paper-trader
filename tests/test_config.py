import dataclasses
import pathlib
import re
import types
import typing

import pytest

from papertrader.config import (
    AppConfig,
    CleaningConfig,
    ConfigError,
    DataConfig,
    PaperConfig,
    PortfolioConfig,
    ResearchConfig,
    StrategyConfig,
    SyntheticConfig,
    config_from_dict,
    load_config,
)

CONFIG_DIR = pathlib.Path(__file__).parents[1] / "config"


def test_typo_in_risk_section_is_an_error():
    with pytest.raises(ConfigError, match="max_postion_pct"):
        config_from_dict({"risk": {"max_postion_pct": 0.1}})


def test_unknown_top_level_section_is_an_error():
    with pytest.raises(ConfigError, match="riks"):
        config_from_dict({"riks": {}})


def test_dates_and_lists_are_normalised():
    cfg = config_from_dict({"data": {"symbols": ["SPY", "TLT"], "start": "2010-01-01"}, "risk": {"symbol_whitelist": ["SPY"]}})
    assert cfg.data.symbols == ("SPY", "TLT") and cfg.risk.symbol_whitelist == ("SPY",)


def test_bad_values_are_rejected():
    with pytest.raises(ConfigError):
        config_from_dict({"risk": {"max_drawdown_pct": 30}})
    with pytest.raises(ConfigError):
        config_from_dict({"data": {"source": "bloomberg"}})


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_shipped_configs_load(path):
    load_config(path)


# --- types: YAML is loose, so every value is checked against its field annotation ---


def test_string_false_does_not_enable_short_selling():
    # "false" is a non-empty string, so it is truthy: accepting it would turn shorting ON.
    with pytest.raises(ConfigError, match=r"config\.risk\.allow_short: expected true or false, got 'false'"):
        config_from_dict({"risk": {"allow_short": "false"}})
    assert config_from_dict({"risk": {"allow_short": False}}).risk.allow_short is False


@pytest.mark.parametrize(
    "raw, path",
    [
        ({"risk": {"flatten_on_kill": "no"}}, "risk.flatten_on_kill"),
        ({"risk": {"allow_short": 0}}, "risk.allow_short"),
        ({"portfolio": {"allow_fractional": "false"}}, "portfolio.allow_fractional"),
        ({"portfolio": {"allow_fractional": 1}}, "portfolio.allow_fractional"),
        ({"data": {"use_adjusted": "yes"}}, "data.use_adjusted"),
        ({"data": {"synthetic": {"inject_errors": "true"}}}, "data.synthetic.inject_errors"),
    ],
)
def test_non_boolean_values_for_flags_are_rejected(raw, path):
    with pytest.raises(ConfigError, match=rf"config\.{path}: expected true or false"):
        config_from_dict(raw)


@pytest.mark.parametrize(
    "raw, path",
    [
        ({"risk": {"max_orders_per_day": 2.7}}, "risk.max_orders_per_day"),
        ({"risk": {"max_data_age_days": "4"}}, "risk.max_data_age_days"),
        ({"paper": {"max_catchup_days": 1.5}}, "paper.max_catchup_days"),
        ({"research": {"bootstrap_samples": 10.5}}, "research.bootstrap_samples"),
        ({"risk": {"max_orders_per_day": True}}, "risk.max_orders_per_day"),
        ({"paper": {"max_catchup_days": False}}, "paper.max_catchup_days"),
    ],
)
def test_int_fields_reject_fractions_strings_and_bools(raw, path):
    with pytest.raises(ConfigError, match=rf"config\.{path}: expected an integer"):
        config_from_dict(raw)


def test_integral_floats_become_ints_and_ints_become_floats():
    cfg = config_from_dict({"risk": {"max_orders_per_day": 3.0, "max_order_notional": 250000}})
    assert cfg.risk.max_orders_per_day == 3 and type(cfg.risk.max_orders_per_day) is int
    assert cfg.risk.max_order_notional == 250000.0 and type(cfg.risk.max_order_notional) is float


@pytest.mark.parametrize(
    "text, path",
    [
        ("risk: {max_position_pct: .nan}", "risk.max_position_pct"),
        ("risk: {max_drawdown_pct: .inf}", "risk.max_drawdown_pct"),
        ("costs: {slippage_bps: .inf}", "costs.slippage_bps"),
        ("portfolio: {initial_cash: .inf}", "portfolio.initial_cash"),
        ("data: {synthetic: {drift: -.inf}}", "data.synthetic.drift"),
        ("data: {synthetic: {t_dof: .nan}}", "data.synthetic.t_dof"),
        ("risk: {max_position_pct: '0.1'}", "risk.max_position_pct"),
        ("risk: {max_position_pct: true}", "risk.max_position_pct"),
    ],
)
def test_float_fields_need_finite_numbers(tmp_path, text, path):
    # NaN compares False against every limit, so a NaN limit would never trip.
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(text)
    with pytest.raises(ConfigError, match=rf"config\.{path}: expected a finite number"):
        load_config(cfg_file)


def test_optional_fields_accept_null():
    cfg = config_from_dict(
        {"data": {"synthetic": {"t_dof": None}}, "risk": {"max_data_age_days": None, "symbol_whitelist": None}}
    )
    assert cfg.data.synthetic.t_dof is None and cfg.risk.max_data_age_days is None and cfg.risk.symbol_whitelist is None


@pytest.mark.parametrize(
    "raw, match",
    [
        ({"name": 2024}, r"config\.name: expected a string"),
        ({"strategy": {"name": None}}, r"config\.strategy\.name: expected a string"),
        ({"strategy": {"params": [1, 2]}}, r"config\.strategy\.params: expected a mapping"),
        ({"research": {"param_grid": None}}, r"config\.research\.param_grid: expected a mapping"),
        ({"data": {"symbols": "SPY"}}, r"config\.data\.symbols: expected a list"),
        ({"data": {"symbols": None}}, r"config\.data\.symbols: expected a list"),
        ({"data": {"symbols": ["SPY", None]}}, r"config\.data\.symbols\[1\]: expected a string"),
        ({"risk": {"symbol_whitelist": ["SPY", ["TLT"]]}}, r"config\.risk\.symbol_whitelist\[1\]: expected a string"),
    ],
)
def test_strings_mappings_and_lists_are_type_checked(raw, match):
    with pytest.raises(ConfigError, match=match):
        config_from_dict(raw)


def test_tickers_yaml_reads_as_other_types_must_be_quoted(tmp_path):
    # Unquoted, YAML reads ON as True and 0700 as octal 448: str() would trade the wrong symbol.
    for text in ("data: {symbols: [SPY, ON]}", "data: {symbols: [0700]}"):
        cfg_file = tmp_path / "c.yaml"
        cfg_file.write_text(text)
        with pytest.raises(ConfigError, match="quote it"):
            load_config(cfg_file)
    cfg_file.write_text("data: {symbols: [SPY, 'ON', '0700']}")
    assert load_config(cfg_file).data.symbols == ("SPY", "ON", "0700")


def _leaf_fields(cls=AppConfig, prefix=()):
    for name, tp in typing.get_type_hints(cls).items():
        if dataclasses.is_dataclass(tp):
            yield from _leaf_fields(tp, (*prefix, name))
        else:
            yield (*prefix, name), tp


_WRONG = {bool: "false", int: 2.5, float: "0.1", str: 1}


def _scalar(tp):
    """bool/int/float/str, or one of those | None; anything else is None."""
    union = typing.get_origin(tp) in (typing.Union, types.UnionType)
    options = [t for t in typing.get_args(tp) if t is not type(None)] if union else [tp]
    return options[0] if len(options) == 1 and options[0] in _WRONG else None


_SCALARS = [(path, base) for path, tp in _leaf_fields() if (base := _scalar(tp))]


def test_type_checks_reach_sections_defined_in_other_modules():
    names = {".".join(path) for path, _ in _SCALARS}
    assert {"risk.allow_short", "risk.max_data_age_days", "costs.slippage_bps", "data.synthetic.t_dof"} <= names


@pytest.mark.parametrize("path, base", _SCALARS, ids=[".".join(path) for path, _ in _SCALARS])
def test_every_scalar_field_is_type_checked(path, base):
    """Generic on purpose: a field added to any section later is covered by its annotation."""
    raw = _WRONG[base]
    for key in reversed(path):
        raw = {key: raw}
    with pytest.raises(ConfigError, match=re.escape("config." + ".".join(path) + ":")):
        config_from_dict(raw)


# --- ranges: checked in __post_init__, so configs built in Python are covered too ---


@pytest.mark.parametrize(
    "raw, match",
    [
        ({"paper": {"max_catchup_days": -1}}, r"paper\.max_catchup_days"),
        ({"paper": {"max_catchup_days": 0}}, r"paper\.max_catchup_days"),
        ({"paper": {"state_dir": "  "}}, r"paper\.state_dir"),
        ({"paper": {"market_timezone": "Mars/Olympus_Mons"}}, r"paper\.market_timezone must be an IANA time zone"),
        ({"paper": {"market_timezone": "../etc/passwd"}}, r"paper\.market_timezone"),
        ({"paper": {"market_timezone": " "}}, r"paper\.market_timezone"),
        ({"paper": {"market_close": "4pm"}}, r"paper\.market_close must be a time like 16:00"),
        ({"paper": {"market_close": "24:00"}}, r"paper\.market_close"),
        ({"paper": {"market_close": "9:30"}}, r"paper\.market_close"),
        ({"paper": {"close_buffer_minutes": -1}}, r"paper\.close_buffer_minutes"),
        ({"portfolio": {"rebalance_band": -5}}, r"portfolio\.rebalance_band"),
        ({"portfolio": {"rebalance_band": 1}}, r"portfolio\.rebalance_band"),
        ({"portfolio": {"min_trade_notional": -1}}, r"portfolio\.min_trade_notional"),
        ({"portfolio": {"initial_cash": 0}}, r"portfolio\.initial_cash"),
        ({"portfolio": {"cash_buffer_pct": 1}}, r"portfolio\.cash_buffer_pct"),
        ({"data": {"synthetic": {"n_days": 1}}}, r"data\.synthetic\.n_days"),
        ({"data": {"synthetic": {"seed": -1}}}, r"data\.synthetic\.seed"),
        ({"data": {"synthetic": {"vol": 0}}}, r"data\.synthetic\.vol"),
        ({"data": {"synthetic": {"market_corr": 1}}}, r"data\.synthetic\.market_corr"),
        ({"data": {"synthetic": {"market_corr": -0.1}}}, r"data\.synthetic\.market_corr"),
        ({"data": {"synthetic": {"regime_drift": -0.3}}}, r"data\.synthetic\.regime_drift"),
        ({"data": {"synthetic": {"regime_persistence": 1.5}}}, r"data\.synthetic\.regime_persistence"),
        ({"data": {"synthetic": {"t_dof": 2}}}, r"data\.synthetic\.t_dof"),
        ({"data": {"synthetic": {"start": "Jan 2015"}}}, r"data\.synthetic\.start"),
        ({"data": {"cleaning": {"spike_threshold": 0}}}, r"data\.cleaning\.spike_threshold"),
        ({"data": {"cleaning": {"stale_run": 1}}}, r"data\.cleaning\.stale_run"),
        ({"data": {"cleaning": {"max_ffill_days": -1}}}, r"data\.cleaning\.max_ffill_days"),
        ({"data": {"start": "2020-13-01"}}, r"data\.start"),
        ({"data": {"start": "2020-01-01", "end": "2020-01-01"}}, r"data\.start .* must be before"),
        ({"data": {"symbols": ["SPY", ""]}}, r"data\.symbols"),
        ({"strategy": {"name": ""}}, r"strategy\.name"),
        ({"research": {"warmup_days": -1}}, r"research\.warmup_days"),
        ({"research": {"bootstrap_samples": 0}}, r"research\.bootstrap_samples"),
        ({"research": {"timing_permutations": 0}}, r"research\.timing_permutations"),
        ({"research": {"split_date": "soon"}}, r"research\.split_date"),
    ],
)
def test_out_of_range_values_are_rejected(raw, match):
    with pytest.raises(ConfigError, match=match):
        config_from_dict(raw)


def test_yaml_dates_are_accepted_for_date_fields(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("data: {start: 2010-01-04, end: 2020-12-31}\nresearch: {split_date: 2016-12-30}")
    cfg = load_config(cfg_file)
    assert (cfg.data.start, cfg.data.end, cfg.research.split_date) == ("2010-01-04", "2020-12-31", "2016-12-30")


@pytest.mark.parametrize(
    "build",
    [
        lambda: SyntheticConfig(n_days="10"),
        lambda: CleaningConfig(stale_run=None),
        lambda: DataConfig(symbols="SPY"),  # would otherwise trade S, P and Y
        lambda: StrategyConfig(name=None),
        lambda: PortfolioConfig(initial_cash="1e5"),
        lambda: ResearchConfig(split_date=20161230),
        lambda: PaperConfig(max_catchup_days=None),
        lambda: PaperConfig(market_timezone=None),
        lambda: PaperConfig(market_close=960),
        lambda: PaperConfig(close_buffer_minutes="15"),
    ],
)
def test_wrong_types_in_python_raise_config_error_not_type_error(build):
    with pytest.raises(ConfigError):
        build()


def test_market_close_must_be_quoted_in_yaml(tmp_path):
    """YAML 1.1 reads an unquoted 16:00 as the base-60 integer 960."""
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("paper: {market_close: 16:00}")
    with pytest.raises(ConfigError, match=r"paper\.market_close: expected a string, got 960 \(quote it in the YAML\)"):
        load_config(cfg_file)
    cfg_file.write_text('paper: {market_timezone: Europe/London, market_close: "16:30", close_buffer_minutes: 0}')
    paper = load_config(cfg_file).paper
    assert (paper.market_timezone, paper.market_close, paper.close_buffer_minutes) == ("Europe/London", "16:30", 0)


# --- research.param_grid: the sweep iterates every value list ---


@pytest.mark.parametrize(
    "grid, match",
    [
        ({"lookback": 50}, r"research\.param_grid\.lookback must be a non-empty list of values, got 50"),
        ({"lookback": "99"}, r"research\.param_grid\.lookback must be a non-empty list of values, got '99'"),  # not 9, 9
        ({"lookback": []}, r"research\.param_grid\.lookback must be a non-empty list of values, got \[\]"),
        ({"lookback": None}, r"research\.param_grid\.lookback must be a non-empty list of values, got None"),
        ({"lookback": {"a": 1}}, r"research\.param_grid\.lookback must be a non-empty list"),
        ({1: [10, 20]}, r"research\.param_grid keys must be parameter names, got 1"),
    ],
    ids=["scalar", "quoted string", "empty list", "null", "mapping", "non-string key"],
)
def test_param_grid_values_must_be_non_empty_lists(grid, match):
    with pytest.raises(ConfigError, match=match):
        config_from_dict({"research": {"param_grid": grid}})
    with pytest.raises(ConfigError, match=match):
        ResearchConfig(param_grid=grid)


def test_param_grid_lists_are_accepted(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("research: {param_grid: {lookback: [10, 20], vol_target: [null, 0.15]}}")
    assert load_config(cfg_file).research.param_grid == {"lookback": [10, 20], "vol_target": [None, 0.15]}


# --- YAML parsing: a repeated key is an error, never "last one wins" ---


@pytest.mark.parametrize(
    "text, match",
    [
        # A second risk block would silently replace the first, whitelist included, with looser defaults.
        (
            "risk: {max_drawdown_pct: 0.10, max_position_pct: 0.05, symbol_whitelist: [AAA]}\n"
            "name: x\n"
            "risk: {allow_short: false}\n",
            r"duplicate key 'risk' on line 3 \(first on line 1\)",
        ),
        (
            "risk:\n  max_drawdown_pct: 0.10\n  max_drawdown_pct: 0.30\n",
            r"duplicate key 'max_drawdown_pct' on line 3 \(first on line 2\)",
        ),
        ("risk: {max_drawdown_pct: 0.10, max_drawdown_pct: 0.30}\n", r"duplicate key 'max_drawdown_pct' on line 1"),
        ("research: {param_grid: {lookback: [10], lookback: [20, 30]}}\n", r"duplicate key 'lookback'"),
    ],
    ids=["repeated section", "repeated limit", "repeated limit, one line", "repeated grid key"],
)
def test_duplicate_yaml_keys_are_rejected(tmp_path, text, match):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(text)
    with pytest.raises(ConfigError, match=rf"c\.yaml: {match}"):
        load_config(cfg_file)


def test_yaml_merge_keys_still_override_on_purpose(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("strategy:\n  name: trend\n  params:\n    <<: {lookback: 20, vol_target: 0.1}\n    lookback: 50\n")
    assert load_config(cfg_file).strategy.params == {"lookback": 50, "vol_target": 0.1}


def test_malformed_yaml_is_a_config_error(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("risk:\n  max_drawdown_pct: 0.1\n    max_position_pct: 0.2\n")  # mis-indented
    with pytest.raises(ConfigError, match=r"c\.yaml: invalid YAML: mapping values are not allowed here"):
        load_config(cfg_file)


def test_max_ffill_days_zero_is_accepted():
    assert config_from_dict({"data": {"cleaning": {"max_ffill_days": 0}}}).data.cleaning.max_ffill_days == 0
