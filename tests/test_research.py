import dataclasses
import math
import sqlite3

import pytest

import papertrader.research as research
from conftest import make_config
from papertrader.cli import main
from papertrader.config import ConfigError, ResearchConfig, config_from_dict
from papertrader.data import CleaningReport, SyntheticSource, load_market_data
from papertrader.ledger import ReturnPanel, TrialLedger
from papertrader.reporting import research_report
from papertrader.research import _write_verdict, run_research
from papertrader.validation import deflated_sharpe_ratio

# Small and fast: ~1 year in-sample after warmup, ~10 months out-of-sample, pure noise (no edge).
SPLIT, WARMUP = "2016-06-30", 100
GRID_A = {"lookback": [20, 50]}
GRID_B = {"lookback": [30, 80, 120]}


def research_cfg(**params):
    rc = ResearchConfig(
        split_date=SPLIT, warmup_days=WARMUP, param_grid=GRID_A, bootstrap_samples=200, timing_permutations=50, ledger_path=None
    )
    cfg = make_config(n_days=600, research=rc)
    return dataclasses.replace(cfg, strategy=dataclasses.replace(cfg.strategy, params=params)) if params else cfg


@pytest.fixture(scope="module")
def market():
    cfg = research_cfg()
    data, _ = load_market_data(cfg.data)
    return cfg, data


def _warnings(res) -> list[str]:
    return [line for line in res.verdict if line.startswith("WARN")]


def _counts(res) -> tuple[int, int, int]:
    return res.n_trials_total, res.n_trials_prior, res.oos_prior_evaluations


def with_research(cfg, **changes):
    return dataclasses.replace(cfg, research=dataclasses.replace(cfg.research, **changes))


def other_market(cfg, **synthetic):
    data_cfg = dataclasses.replace(cfg.data, synthetic=dataclasses.replace(cfg.data.synthetic, **synthetic))
    return load_market_data(data_cfg)[0]


def test_ledger_accumulates_trials_across_runs_with_different_grids(market, tmp_path):
    cfg, data = market
    ledger = tmp_path / "ledger.sqlite"
    first = run_research(data, cfg, param_grid=GRID_A, ledger=ledger)
    assert (first.n_trials_total, first.n_trials_prior) == (2, 0)
    assert first.verdict[0].startswith("Best of 2 in-sample trial(s): Sharpe")

    second = run_research(data, cfg, param_grid=GRID_B, ledger=ledger)
    assert len(second.trials) == 3
    assert (second.n_trials_total, second.n_trials_prior) == (5, 2)
    # The best is picked from this run's 3; the earlier 2 only raise the bar it is deflated against.
    assert second.verdict[1].startswith(
        "Best of this run's 3 in-sample trial(s) (deflated for 5 tried on this strategy and data, 2 in earlier runs): "
        f"Sharpe {max(t.sharpe for t in second.trials):.2f}"
    )
    assert any(line[6:].startswith("Deflated for 5 trials") for line in second.verdict)
    # The DSR is deflated for every trial on the question, not just this run's grid...
    every = [t.sharpe_per_period for t in first.trials + second.trials]
    assert second.dsr_in == pytest.approx(deflated_sharpe_ratio(second.in_sample.returns(), every))
    # ...so on noise, the extra trials from the earlier run can only make it harder to pass.
    alone = run_research(data, cfg, param_grid=GRID_B)
    assert alone.n_trials_total == 3
    assert second.dsr_in <= alone.dsr_in


def test_rerunning_the_same_grid_does_not_double_count_trials(market, tmp_path):
    cfg, data = market
    ledger = tmp_path / "ledger.sqlite"
    run_research(data, cfg, ledger=ledger)
    again = run_research(data, cfg, ledger=ledger)
    assert (again.n_trials_total, again.n_trials_prior) == (2, 0)


def test_second_look_at_out_of_sample_is_flagged_first_but_not_scored(market, tmp_path):
    cfg, data = market
    ledger = tmp_path / "ledger.sqlite"
    first = run_research(data, cfg, param_grid=GRID_A, ledger=ledger)
    assert first.oos_prior_evaluations == 0 and not _warnings(first)

    second = run_research(data, cfg, param_grid=GRID_B, ledger=ledger)
    assert second.oos_prior_evaluations == 1
    assert second.verdict[0] == (
        "WARN  The out-of-sample period, or one overlapping it, was already evaluated 1 time(s) for this strategy and "
        "data; it is no longer a clean holdout."
    )
    third = run_research(data, cfg, param_grid=GRID_A, ledger=ledger)
    assert third.oos_prior_evaluations == 2 and "evaluated 2 time(s)" in third.verdict[0]
    # A warning, not a check: pass/fail counts match a run without a ledger.
    clean = run_research(data, cfg, param_grid=GRID_A)
    assert (third.passed, third.checks, set(third.check_results)) == (clean.passed, clean.checks, set(clean.check_results))


def test_ledger_records_full_params_timestamps_and_oos_windows(tmp_path):
    cfg = research_cfg(vol_target=0.15)
    data, _ = load_market_data(cfg.data)
    ledger = tmp_path / "ledger.sqlite"
    res = run_research(data, cfg, ledger=ledger)
    with sqlite3.connect(ledger) as conn:
        runs = conn.execute(
            "SELECT run_id, recorded_at, strategy, source, symbols, split_date, warmup_days, oos_start, oos_end, best_params "
            "FROM runs"
        ).fetchall()
        trials = conn.execute("SELECT run_id, params, sharpe, sharpe_per_period FROM trials ORDER BY rowid").fetchall()
        panels = conn.execute("SELECT count(*) FROM panels").fetchone()[0]
    oos = res.out_of_sample.equity.index
    [(run_id, at, strategy, source, symbols, split, warmup, oos_start, oos_end, best)] = runs
    assert run_id == res.ledger_run_id and at
    assert (strategy, source, split, warmup) == ("trend", "synthetic", SPLIT, WARMUP)
    assert symbols == '["AAA", "BBB", "CCC", "DDD", "EEE"]'
    assert (oos_start, oos_end) == (str(oos[0].date()), str(oos[-1].date()))
    assert '"vol_target": 0.15' in best
    assert [t[0] for t in trials] == [run_id] * 2
    # The base strategy.params are part of each trial: they change the strategy that ran.
    assert [t[1] for t in trials] == ['{"lookback": 20, "vol_target": 0.15}', '{"lookback": 50, "vol_target": 0.15}']
    assert [t[2] for t in trials] == pytest.approx([t.sharpe for t in res.trials])
    assert [t[3] for t in trials] == pytest.approx([t.sharpe_per_period for t in res.trials])
    assert panels == 1  # the in-sample returns, kept so later runs can be matched against them


def test_without_a_ledger_nothing_is_remembered(market, tmp_path, monkeypatch):
    cfg, data = market
    monkeypatch.chdir(tmp_path)
    runs = [run_research(data, cfg, param_grid=GRID_A) for _ in range(2)]
    for res in runs:
        assert (res.n_trials_total, res.n_trials_prior, res.oos_prior_evaluations) == (2, 0, 0)
        assert res.ledger_run_id is None and not _warnings(res)
    assert list(tmp_path.iterdir()) == []  # not even the default state/ ledger
    # And a fresh ledger's first run is indistinguishable from no ledger at all.
    fresh = run_research(data, cfg, param_grid=GRID_A, ledger=tmp_path / "ledger.sqlite")
    assert fresh.verdict == runs[0].verdict
    assert fresh.dsr_in == runs[0].dsr_in


def _bump_one_day(frame):
    out = frame.copy()
    out.iloc[10] *= 1.01  # well inside the in-sample period
    return out


def _back_adjust_dividend(frame):
    out = frame.copy()
    out.iloc[:200] *= 0.996  # a 0.4% dividend goes ex on day 200: every earlier adjusted price is scaled down
    return out


def _corrected_bad_tick(frame):
    out = frame.copy()
    out.iloc[10] *= 1.10  # the first download had a 10% bad print that the vendor later corrected
    return out


def _corrected_bad_tick_at_month_end(frame):
    out = frame.copy()
    month = out.index.to_period("M")
    last = next(i for i in range(5, len(out) - 1) if month[i] != month[i + 1])  # a month's last trading day
    out.iloc[last] *= 1.10  # the print and its reversal now fall in different months
    return out


def _panel(data, split=SPLIT) -> ReturnPanel:
    return ReturnPanel.from_data(data.truncate(split))


def test_same_data_is_matched_not_hashed(market):
    cfg, data = market
    panel = _panel(data)
    # The same market history, revised or re-listed: an exact hash would call each of these new data.
    assert panel.matches(_panel(data.truncate("2017-01-31")))  # fewer out-of-sample days
    assert panel.matches(_panel(data.map_prices(lambda f: f * (1 + 1e-15))))  # re-download jitter
    assert panel.matches(_panel(data.map_prices(_back_adjust_dividend)))
    assert panel.matches(_panel(data.map_prices(_bump_one_day)))  # a corrected tick
    assert panel.matches(_panel(data.map_prices(_corrected_bad_tick)))  # a big one: one outlier day is left out
    assert panel.matches(_panel(data.map_prices(_corrected_bad_tick_at_month_end)))  # ...from the monthly sums too
    reordered = load_market_data(dataclasses.replace(cfg.data, symbols=cfg.data.symbols[::-1]))[0]
    assert reordered.symbols != data.symbols and _panel(reordered).digest == panel.digest
    assert panel.matches(_panel(data, "2016-06-29")) and panel.matches(_panel(data, "2016-07-01"))  # split nudged
    # Other market histories don't, even on the same symbols and dates.
    assert not panel.matches(_panel(other_market(cfg, seed=12)))
    # Same daily shocks, plus hidden trends: daily returns still correlate ~0.996, monthly ones don't.
    assert not panel.matches(_panel(other_market(cfg, regime_drift=0.3)))
    subset = load_market_data(dataclasses.replace(cfg.data, symbols=cfg.data.symbols[:4]))[0]
    assert not panel.matches(_panel(subset))
    assert not panel.matches(_panel(data, "2016-01-29"))  # in-sample 28% shorter: under 90% of it is shared
    assert panel.matches(_panel(data, "2016-01-29"), nested=True)  # ...but it is the same market
    assert not panel.matches(_panel(other_market(cfg, seed=12)), nested=True)


def test_warmup_change_and_split_nudge_keep_the_count_and_the_warning(market, tmp_path):
    cfg, data = market
    ledger = tmp_path / "ledger.sqlite"
    assert _counts(run_research(data, cfg, param_grid=GRID_A, ledger=ledger)) == (2, 0, 0)
    # Same out-of-sample window, one more warmup day: the holdout is just as spent.
    warmup = run_research(data, with_research(cfg, warmup_days=WARMUP + 1), param_grid=GRID_B, ledger=ledger)
    assert _counts(warmup) == (5, 2, 1) and _warnings(warmup)
    # A split a business day earlier or later looks at almost the same out-of-sample days.
    earlier = run_research(data, with_research(cfg, split_date="2016-06-29"), param_grid=GRID_A, ledger=ledger)
    assert _counts(earlier) == (5, 3, 2) and _warnings(earlier)
    later = run_research(data, with_research(cfg, split_date="2016-07-01"), param_grid={"lookback": [65]}, ledger=ledger)
    assert _counts(later) == (6, 5, 3) and "evaluated 3 time(s)" in later.verdict[0]


def test_reordered_symbols_and_a_back_adjusted_redownload_keep_the_count_and_the_warning(tmp_path):
    raw = SyntheticSource(n_days=600, seed=11).load(["AAA", "BBB", "CCC"])
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()

    def download(rows: int, dividend: bool = False):
        for sym, df in raw.items():
            out = df.iloc[:rows].copy()
            out["adj_close"] = out["close"]
            if dividend:
                out.iloc[:200, out.columns.get_loc("adj_close")] *= 0.996  # ex-dividend in-sample, on day 200
            out.to_csv(csv_dir / f"{sym}.csv", index_label="date")

    def research_on(symbols, **kwargs):
        data_cfg = {"source": "csv", "csv_dir": str(csv_dir), "symbols": symbols, "use_adjusted": True}
        cfg = config_from_dict({"data": data_cfg, "research": dataclasses.asdict(research_cfg().research)})
        return run_research(load_market_data(cfg.data)[0], cfg, ledger=tmp_path / "ledger.sqlite", **kwargs)

    download(580)
    assert _counts(research_on(["AAA", "BBB", "CCC"])) == (2, 0, 0)
    download(600, dividend=True)  # weeks later: more days, and every earlier adjusted close revised
    second = research_on(["CCC", "AAA", "BBB"], param_grid=GRID_B)
    assert _counts(second) == (5, 2, 1) and _warnings(second)


def test_the_same_bars_from_a_csv_copy_keep_the_count_and_the_warning(market, tmp_path):
    """config/example_yahoo.yaml says: download once, then switch data.source to csv. The source
    used to be part of the ledger's identity, so following that advice reset the count."""
    cfg, data = market
    ledger = tmp_path / "ledger.sqlite"
    assert _counts(run_research(data, cfg, param_grid=GRID_A, ledger=ledger)) == (2, 0, 0)
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    for sym, df in SyntheticSource(n_days=600, seed=11).load(list(cfg.data.symbols)).items():
        df.to_csv(csv_dir / f"{sym}.csv", index_label="date")
    csv_cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, source="csv", csv_dir=str(csv_dir)))
    copy = run_research(load_market_data(csv_cfg.data)[0], csv_cfg, param_grid=GRID_B, ledger=ledger)
    assert _counts(copy) == (5, 2, 1) and _warnings(copy)


def test_a_moved_start_date_still_finds_the_spent_holdout(market, tmp_path):
    """Dropping the first months of history changes the in-sample window by far more than 10%,
    so its trials count afresh, but the out-of-sample window is the one already looked at."""
    cfg, data = market
    ledger = tmp_path / "ledger.sqlite"
    run_research(data, cfg, param_grid=GRID_A, ledger=ledger)
    later_start = data.between(start=data.dates[120])
    assert _counts(run_research(later_start, cfg, param_grid=GRID_A, ledger=ledger)) == (2, 0, 1)


def test_a_sliding_window_still_finds_the_spent_holdout(market, tmp_path):
    """Start and split moved together (walk-forward): neither in-sample window is 90% covered
    by the other, but they share most of a year of the same market, and the new out-of-sample
    period lies inside the one already looked at."""
    cfg, data = market
    ledger = tmp_path / "ledger.sqlite"
    run_research(data, cfg, param_grid=GRID_A, ledger=ledger)
    slid = data.between(start=data.dates[120])
    moved = run_research(slid, with_research(cfg, split_date="2016-09-30"), param_grid=GRID_A, ledger=ledger)
    assert _counts(moved) == (2, 0, 1) and _warnings(moved)
    # A different market on the same window is still not the same market.
    other = other_market(cfg, seed=12).between(start=data.dates[120])
    assert _counts(run_research(other, with_research(cfg, split_date="2016-09-30"), param_grid=GRID_A, ledger=ledger)) == (
        2,
        0,
        0,
    )


def test_other_strategies_and_other_markets_get_their_own_count(market, tmp_path):
    cfg, data = market
    ledger = tmp_path / "ledger.sqlite"
    run_research(data, cfg, ledger=ledger)
    assert _counts(run_research(data, cfg, strategy_name="mean_reversion", param_grid={}, ledger=ledger)) == (1, 0, 0)
    assert _counts(run_research(other_market(cfg, seed=12), cfg, param_grid=GRID_B, ledger=ledger)) == (3, 0, 0)
    assert _counts(run_research(other_market(cfg, regime_drift=0.3), cfg, param_grid=GRID_B, ledger=ledger)) == (3, 0, 0)
    # A split moved by months is a different in-sample window, so its trials count afresh, but it
    # is still the same market: its out-of-sample period overlaps the ones already looked at.
    moved = run_research(data, with_research(cfg, split_date="2016-01-29"), param_grid=GRID_B, ledger=ledger)
    assert _counts(moved) == (3, 0, 1)


def test_parameter_sets_that_never_trade_still_count(market, tmp_path):
    cfg, data = market
    ledger = tmp_path / "ledger.sqlite"
    dead = run_research(data, cfg, param_grid={"lookback": [20, 4000]}, ledger=ledger)  # 4000 > the history
    assert math.isnan(dead.trials[1].sharpe_per_period)
    # Two trials, one Sharpe: no spread to deflate by, so the check fails rather than quietly becoming a PSR.
    assert math.isnan(dead.dsr_in) and dead.check_results["deflated_sharpe"] is False
    assert any(line.startswith("FAIL  Deflated for 2 trials, no DSR can be computed") for line in dead.verdict)
    # Read back from the ledger, the dead trial is still one of the N.
    later = run_research(data, cfg, param_grid=GRID_B, ledger=ledger)
    assert later.n_trials_total == 5
    every = [t.sharpe_per_period for t in dead.trials + later.trials]
    assert later.dsr_in == pytest.approx(deflated_sharpe_ratio(later.in_sample.returns(), every))
    assert later.dsr_in < deflated_sharpe_ratio(later.in_sample.returns(), [x for x in every if not math.isnan(x)])


def test_a_run_recorded_meanwhile_is_counted_before_the_verdict_is_returned(market, tmp_path, monkeypatch):
    cfg, data = market
    ledger = tmp_path / "ledger.sqlite"
    other = {}
    write_verdict = research._write_verdict

    def another_terminal_finishes_first(res):
        write_verdict(res)
        if not other:  # between this run reading the ledger and recording itself
            other["run"] = None
            other["run"] = run_research(data, cfg, param_grid=GRID_B, ledger=ledger)

    monkeypatch.setattr(research, "_write_verdict", another_terminal_finishes_first)
    mine = run_research(data, cfg, param_grid=GRID_A, ledger=ledger)
    assert _counts(other["run"]) == (3, 0, 0)
    assert _counts(mine) == (5, 3, 1) and _warnings(mine)  # recounted under the ledger's write lock
    every = [t.sharpe_per_period for t in mine.trials + other["run"].trials]
    assert mine.dsr_in == pytest.approx(deflated_sharpe_ratio(mine.in_sample.returns(), every))
    with sqlite3.connect(ledger) as conn:
        assert conn.execute("SELECT count(*) FROM runs").fetchone()[0] == 2


def test_an_old_format_ledger_is_refused_not_reset(tmp_path):
    old = tmp_path / "ledger.sqlite"
    with sqlite3.connect(old) as conn:  # the version-1 layout: runs keyed by an exact hash of the closes
        conn.execute("CREATE TABLE trials (key TEXT, run_id TEXT, recorded_at TEXT, params TEXT, sharpe REAL)")
        conn.execute("INSERT INTO trials VALUES ('k', 'r', 't', '{}', 0.1)")
    conn.close()
    with pytest.raises(RuntimeError, match="not a version-2 trial ledger .* rename this file"):
        TrialLedger(old)
    with sqlite3.connect(old) as conn:  # left exactly as it was
        assert conn.execute("SELECT count(*) FROM trials").fetchone()[0] == 1
    conn.close()


def test_report_says_the_best_is_from_this_run(market, tmp_path, monkeypatch):
    cfg, data = market
    ledger = tmp_path / "ledger.sqlite"
    titles = []
    monkeypatch.setattr("papertrader.reporting.plot_equity", lambda curves, path, title, **kw: titles.append(title))
    research_report(cfg, run_research(data, cfg, ledger=ledger), CleaningReport(), tmp_path / "r1")
    res = run_research(data, cfg, param_grid=GRID_B, ledger=ledger)
    report = research_report(cfg, res, CleaningReport(), tmp_path / "r2").read_text()
    assert titles == [
        "test: trend, best of 2 in-sample, then out-of-sample",
        "test: trend, best of this run's 3 in-sample (deflated for 5 tried), then out-of-sample",
    ]
    assert f"This run is `{res.ledger_run_id}`." in report
    assert "## This run's 3 in-sample trials\n\nThe best is chosen from these only." in report


def test_unusable_ledger_fails_before_the_sweep(market, tmp_path):
    cfg, data = market
    bad = tmp_path / "ledger.sqlite"
    bad.write_bytes(b"this is not a database" * 100)
    with pytest.raises(RuntimeError, match="research ledger"):
        run_research(data, cfg, ledger=bad)
    with pytest.raises(RuntimeError, match="research ledger"):
        TrialLedger(tmp_path)  # a directory


def test_benchmark_check_needs_the_bootstrap_lower_bound_above_zero(market):
    cfg, data = market
    res = run_research(data, cfg)
    assert res.check_results["beats_benchmark_oos"] == (res.oos_vs_benchmark["lower"] > 0)

    def verdict_for(**diff):
        r = dataclasses.replace(res, oos_vs_benchmark=diff)
        _write_verdict(r)
        line = next(x for x in r.verdict if "benchmark out-of-sample" in x)
        return r.check_results["beats_benchmark_oos"], line

    # A positive point difference is not enough: the lower bound has to clear zero.
    ok, line = verdict_for(difference=0.41, lower=-0.07, p_value=0.2)
    assert not ok
    assert line.startswith(
        "FAIL  Did not beat the benchmark out-of-sample: Sharpe difference +0.41, one-sided 95% lower bound -0.07 ("
    )
    ok, line = verdict_for(difference=0.41, lower=0.07, p_value=0.01)
    assert ok
    assert line.startswith("PASS  Beat the benchmark out-of-sample: Sharpe difference +0.41, one-sided 95% lower bound +0.07 (")
    ok, line = verdict_for(difference=math.nan, lower=math.nan, p_value=math.nan)  # fail closed
    assert not ok and "n/a" in line


def test_ledger_path_config():
    assert ResearchConfig().ledger_path == "state/research_ledger.sqlite"
    assert config_from_dict({"research": {"ledger_path": None}}).research.ledger_path is None
    with pytest.raises(ConfigError, match="research.ledger_path"):
        ResearchConfig(ledger_path=" ")


def test_cli_research_uses_the_configured_ledger(tmp_path, capsys):
    ledger = tmp_path / "ledger.sqlite"
    cfg_file = tmp_path / "research.yaml"
    cfg_file.write_text(
        "name: tiny\n"
        "data: {synthetic: {n_days: 600, seed: 11}}\n"
        f"research: {{split_date: {SPLIT}, warmup_days: {WARMUP}, param_grid: {{lookback: [20, 50]}}, "
        f"bootstrap_samples: 200, timing_permutations: 50, ledger_path: '{ledger}'}}\n"
    )
    args = ["research", "-c", str(cfg_file), "--out", str(tmp_path / "report")]
    assert main(args) == 0
    assert ledger.exists() and "WARN" not in capsys.readouterr().out
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "WARN  The out-of-sample period, or one overlapping it, was already evaluated 1 time(s)" in out
    assert f"Trial ledger: {ledger} (this run: " in out
    report = (tmp_path / "report" / "report.md").read_text()
    verdict = report.split("## Verdict")[1]
    assert verdict.lstrip().startswith("- **WARN  The out-of-sample period, or one overlapping it, was already evaluated 1")
