"""Does the research protocol say "no" to noise and "yes" to a real edge?

Runs the full protocol on many synthetic markets of each kind and counts how
often each check passes. On the random walk there is nothing to find, so
every pass is a false positive. On the trending market the edge is real, so
passing measures how much power the protocol has with this much data.

    python scripts/calibrate_protocol.py --seeds 10
"""
from __future__ import annotations

import argparse
import dataclasses
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from papertrader.config import load_config  # noqa: E402
from papertrader.data import load_market_data  # noqa: E402
from papertrader.metrics import sharpe  # noqa: E402
from papertrader.research import run_research  # noqa: E402

WORLDS = {
    "random walk (no edge)": "config/demo_random_walk.yaml",
    "trending (real edge)": "config/demo_trending.yaml",
}
CHECKS = [
    ("naive_psr", "In-sample winner 'significant' (PSR >= 0.95, no correction)"),
    ("deflated_sharpe", "Still significant after deflating for the number of trials"),
    ("oos_interval_above_zero", "Out-of-sample Sharpe interval above zero"),
    ("beats_benchmark_oos", "Beats the benchmark out-of-sample"),
    ("timing", "Timing beats shifted copies (p < 0.05)"),
    ("all", "Passes every protocol check"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    table = {}
    for label, path in WORLDS.items():
        base = load_config(path)
        rows = []
        for seed in range(1, args.seeds + 1):
            t0 = time.time()
            cfg = dataclasses.replace(
                base,
                data=dataclasses.replace(base.data, synthetic=dataclasses.replace(base.data.synthetic, seed=seed)),
                research=dataclasses.replace(base.research, bootstrap_samples=500, timing_permutations=200),
            )
            data, _ = load_market_data(cfg.data)
            res = run_research(data, cfg)
            row = dict(res.check_results)
            row["naive_psr"] = res.psr_in >= 0.95
            row["all"] = res.passed == res.checks
            row["is_sharpe"] = res.best.sharpe
            row["oos_sharpe"] = sharpe(res.out_of_sample.returns())
            rows.append(row)
            print(
                f"  {label:<22} seed {seed:>2}: best IS Sharpe {row['is_sharpe']:5.2f} -> OOS {row['oos_sharpe']:5.2f}, "
                f"passed {res.passed}/{res.checks} ({time.time() - t0:.0f}s)",
                flush=True,
            )
        table[label] = rows

    print(f"\nShare of {args.seeds} markets where each check passed:\n")
    names = list(table)
    print(f"{'':<62}" + "".join(f"{n:>24}" for n in names))
    for key, text in CHECKS:
        cells = "".join(f"{sum(bool(r.get(key)) for r in table[n]) / len(table[n]):>24.0%}" for n in names)
        print(f"{text:<62}{cells}")
    for stat, text in (("is_sharpe", "Median best in-sample Sharpe"), ("oos_sharpe", "Median out-of-sample Sharpe")):
        cells = "".join(f"{statistics.median(r[stat] for r in table[n]):>24.2f}" for n in names)
        print(f"{text:<62}{cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
