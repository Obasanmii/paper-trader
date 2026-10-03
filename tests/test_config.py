import pathlib

import pytest

from papertrader.config import ConfigError, config_from_dict, load_config

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
