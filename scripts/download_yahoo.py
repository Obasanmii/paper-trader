"""Download daily bars once with yfinance and save them as CSVs, so every later
run reads the same frozen data (reproducible, and no rate limits).

    pip install yfinance
    python scripts/download_yahoo.py -c config/example_yahoo.yaml
    # then set `data.source: csv` in the config

Yahoo data via yfinance is unofficial: fine for learning, not for anything
you'd bet on. Check the cleaning report (`python -m papertrader data -c ...`)
after every download.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from papertrader.config import load_config  # noqa: E402
from papertrader.data import YFinanceSource  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg.data.csv_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = YFinanceSource().load(list(cfg.data.symbols), cfg.data.start, cfg.data.end)
    for sym, df in frames.items():
        path = out / f"{sym}.csv"
        df.to_csv(path, index_label="date")
        print(f"{sym}: {len(df):,} rows, {df.index[0].date()} to {df.index[-1].date()} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
