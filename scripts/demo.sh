#!/usr/bin/env bash
# The whole tour in one go (about a minute). Run from the repo root.
set -euo pipefail
PY="$(command -v python || command -v python3)"
PT="$PY -m papertrader"

echo "== 1. Tests: risk limits, kill switch, no-lookahead contract, paper == backtest"
"$PY" -m pytest -q

echo; echo "== 2. Data cleaning catches planted errors"
$PT data -c config/demo_dirty_data.yaml

echo; echo "== 3. Research on pure noise: the protocol should say no"
$PT research -c config/demo_random_walk.yaml

echo; echo "== 4. Research on a market with a real edge"
$PT research -c config/demo_trending.yaml

echo; echo "== 5. Backtest report (strategy vs benchmark, costs, risk)"
$PT backtest -c config/demo_trending.yaml | tail -6

echo; echo "== 6. Paper trading, a day at a time, then the kill switch"
rm -rf state/demo_paper
for d in 2023-06-01 2023-06-02 2023-06-05 2023-06-06; do $PT paper-step -c config/demo_paper.yaml --as-of $d; done
$PT kill -c config/demo_paper.yaml --reason "demo"
$PT paper-step -c config/demo_paper.yaml --as-of 2023-06-07
$PT paper-step -c config/demo_paper.yaml --as-of 2023-06-08
$PT reset-kill -c config/demo_paper.yaml --yes-i-checked

echo; echo "Reports are in reports/, paper state and the SQLite journal in state/demo_paper/."
