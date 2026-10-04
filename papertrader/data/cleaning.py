"""Validate and clean raw bars. Every change is recorded; nothing is fixed silently.

Checks, in order, per symbol:
  1. sort by date, drop duplicate dates (keep the last print)
  2. drop rows with missing / non-positive prices
  3. zero out negative or missing volume
  4. drop one-day close spikes that fully revert the next day (bad ticks)
     (live only, hold_unconfirmed) hold back a newest bar whose close moved
     more than the spike threshold: nothing can confirm it until the next bar
  5. repair open spikes that revert by the close (open := previous close)
  6. cap absurd high/low wicks at the candle body
  7. widen high/low so they contain open and close
  8. flag (don't touch) large moves that persist: possible splits or corporate actions
  9. flag (don't touch) runs of identical closes: possible stale feed

Then all symbols are aligned to one calendar. Short gaps are forward-filled
for *marking only*: close is carried forward, open is left NaN so nothing
can trade on a day that didn't happen. Nothing is ever back-filled, since
back-filling copies future prices into the past.

A note on lookahead: check 4 looks one bar ahead to confirm a tick reverted
(checks 5-7 only look within the same bar). That's fine for historical
research data (the bad print was never a real price), but on the newest bar
there is no next bar yet, so a bad tick can't be told from a real move. One
10x print there nearly tripled marked equity for a day, and the next day's
apparent loss tripped the kill switch and liquidated the account. So live
callers pass `hold_unconfirmed=True`: a big move on the newest bar is held
back (`unconfirmed_move`). After alignment that day is an ordinary filled
gap, marked at the last good close and untradable (or absent, if no other
symbol traded it). A day later check 4 either drops the bar as a bad tick or
keeps it as a large move. Risk limits stay a second line of defence, not
the first. The paper runner cleans data truncated to the current day, so it
only ever uses what it could have known.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from papertrader.data.market import FIELDS, MarketData

PRICE_COLS = ["open", "high", "low", "close"]


@dataclass
class Issue:
    symbol: str
    date: str | None
    kind: str
    action: str
    detail: str = ""


@dataclass
class CleaningReport:
    issues: list[Issue] = field(default_factory=list)

    def add(self, symbol, date, kind, action, detail=""):
        date = None if date is None else str(pd.Timestamp(date).date())
        self.issues.append(Issue(symbol, date, kind, action, detail))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([vars(i) for i in self.issues], columns=["symbol", "date", "kind", "action", "detail"])

    def counts(self) -> dict[str, int]:
        if not self.issues:
            return {}
        return {k: int(v) for k, v in self.to_frame().groupby("kind").size().items()}

    def __len__(self) -> int:
        return len(self.issues)

    def summary(self) -> str:
        if not self.issues:
            return "No data issues found."
        lines = [f"{len(self.issues)} data issue(s):"]
        for kind, count in sorted(self.counts().items()):
            lines.append(f"  {kind:<20} {count}")
        return "\n".join(lines)


def clean_symbol(
    df: pd.DataFrame,
    symbol: str,
    report: CleaningReport,
    spike_threshold: float = 0.25,
    stale_run: int = 5,
    hold_unconfirmed: bool = False,
) -> pd.DataFrame:
    """Clean one symbol's bars. `spike_threshold` is an absolute log return (0.25 ~ 28%).

    `hold_unconfirmed` is for live use, where the newest bar is today's: see the
    lookahead note above. Leave it off for historical data, which then cleans
    exactly as before.
    """
    df = df.copy()
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index))

    # 1. order and duplicates
    if not df.index.is_monotonic_increasing:
        report.add(symbol, None, "unsorted", "sorted by date")
        df = df.sort_index(kind="stable")
    dup = df.index.duplicated(keep="last")
    for d in df.index[dup]:
        report.add(symbol, d, "duplicate_date", "kept the last row")
    df = df.loc[~dup]

    # 2. invalid prices
    px = df[PRICE_COLS]
    bad = ~np.isfinite(px).all(axis=1) | (px <= 0).any(axis=1)
    for d in df.index[bad]:
        report.add(symbol, d, "invalid_price", "dropped row")
    df = df.loc[~bad].copy()

    # 3. volume
    vol = df["volume"]
    bad_vol = ~np.isfinite(vol) | (vol < 0)
    for d in df.index[bad_vol]:
        report.add(symbol, d, "invalid_volume", "set to 0", f"was {vol[d]}")
    df.loc[bad_vol, "volume"] = 0.0

    # 4. close spikes that revert the next day
    r = np.log(df["close"]).diff()
    nxt = r.shift(-1)
    spike = (
        (r.abs() > spike_threshold)
        & (nxt.abs() > spike_threshold)
        & (np.sign(r) != np.sign(nxt))
        & ((r + nxt).abs() < spike_threshold / 2)
    )
    for d in df.index[spike]:
        report.add(symbol, d, "price_spike", "dropped row (reverted next day: bad tick)", f"{r[d]:+.2f} then {nxt[d]:+.2f} log")
    df = df.loc[~spike].copy()
    if hold_unconfirmed:
        df = _hold_back_newest(df, symbol, report, spike_threshold)

    # 5. open spikes that revert by the close
    prev_close = df["close"].shift(1)
    gap = np.log(df["open"] / prev_close)
    intraday = np.log(df["close"] / df["open"])
    open_spike = (
        (gap.abs() > spike_threshold)
        & (intraday.abs() > spike_threshold)
        & (np.sign(gap) != np.sign(intraday))
        & ((gap + intraday).abs() < spike_threshold / 2)
    )
    for d in df.index[open_spike]:
        report.add(symbol, d, "open_spike", "open set to previous close", f"gap {gap[d]:+.2f} log")
    df.loc[open_spike, "open"] = prev_close[open_spike]

    # 6. absurd wicks
    body_hi = df[["open", "close"]].max(axis=1)
    body_lo = df[["open", "close"]].min(axis=1)
    wild_hi = np.log(df["high"] / body_hi) > spike_threshold
    wild_lo = np.log(body_lo / df["low"]) > spike_threshold
    for d in df.index[wild_hi]:
        report.add(symbol, d, "wild_high", "high capped at max(open, close)")
    for d in df.index[wild_lo]:
        report.add(symbol, d, "wild_low", "low capped at min(open, close)")
    df.loc[wild_hi, "high"] = body_hi[wild_hi]
    df.loc[wild_lo, "low"] = body_lo[wild_lo]

    # 7. OHLC consistency
    hi_needed = df[PRICE_COLS].max(axis=1)
    lo_needed = df[PRICE_COLS].min(axis=1)
    inconsistent = (df["high"] < hi_needed) | (df["low"] > lo_needed)
    for d in df.index[inconsistent]:
        report.add(symbol, d, "ohlc_inconsistent", "high/low widened to contain open and close")
    df.loc[inconsistent, "high"] = hi_needed[inconsistent]
    df.loc[inconsistent, "low"] = lo_needed[inconsistent]

    # 8. large persistent moves (flag only)
    r = np.log(df["close"]).diff()
    for d in df.index[r.abs() > spike_threshold]:
        report.add(symbol, d, "large_move", "kept: check for a split or corporate action", f"{r[d]:+.2f} log")

    # 9. stale runs (flag only)
    same = df["close"].diff().eq(0)
    run_id = (~same).cumsum()
    run_len = same.groupby(run_id).transform("sum")
    starts = same & ~same.shift(1, fill_value=False) & (run_len >= stale_run)
    for d in df.index[starts]:
        report.add(symbol, d, "stale_prices", "kept: check the feed", f"{int(run_len[d]) + 1} identical closes")

    return df


def _hold_back_newest(df: pd.DataFrame, symbol: str, report: CleaningReport, spike_threshold: float) -> pd.DataFrame:
    """Check 4 can't judge the newest bar, so a big move there waits one day.
    Marking at yesterday's close for a day is cheap; trading on a bad tick isn't."""
    if len(df) < 2:
        return df  # no previous close to measure the move against
    prev, last = df["close"].iloc[-2], df["close"].iloc[-1]
    move = float(np.log(last / prev))
    if abs(move) <= spike_threshold:
        return df
    report.add(
        symbol,
        df.index[-1],
        "unconfirmed_move",
        f"held back: newest bar moved {move:+.2f} log and can't be confirmed until the next bar",
        f"close {last:.4g} vs previous {prev:.4g}",
    )
    return df.iloc[:-1].copy()


def align(frames: dict[str, pd.DataFrame], report: CleaningReport, max_ffill_days: int = 3) -> MarketData:
    """Put every symbol on one calendar. Short gaps are marked-to-last-close, never traded.
    `max_ffill_days=0` fills nothing: a day without a bar has no mark either."""
    if not frames:
        raise ValueError("no data to align")
    calendar = pd.DatetimeIndex(sorted(set().union(*[f.index for f in frames.values()])))
    panels = {f: {} for f in FIELDS}
    too_long_why = f"gap longer than {max_ffill_days} days" if max_ffill_days else "forward fill is off"
    for sym, df in frames.items():
        df = df.reindex(calendar)
        close_ff = df["close"].ffill(limit=max_ffill_days) if max_ffill_days else df["close"]  # pandas rejects limit=0
        filled = df["close"].isna() & close_ff.notna()
        for d in calendar[filled.to_numpy()]:
            report.add(sym, d, "gap_filled", "close carried forward; open left empty (untradable)")
        first_valid = df["close"].first_valid_index()
        last_valid = df["close"].last_valid_index()
        if first_valid is None:
            raise ValueError(f"{sym}: no valid prices after cleaning")
        inside = (calendar >= first_valid) & (calendar <= last_valid)
        too_long = (df["close"].isna() & close_ff.isna()).to_numpy() & inside
        for d in calendar[too_long]:
            report.add(sym, d, "gap_too_long", f"left empty ({too_long_why})")
        df = df.copy()
        df["close"] = close_ff
        df.loc[filled, "high"] = close_ff[filled]
        df.loc[filled, "low"] = close_ff[filled]
        df.loc[filled, "open"] = np.nan
        df.loc[filled, "volume"] = 0.0
        for f in FIELDS:
            panels[f][sym] = df[f]
    symbols = list(frames)
    return MarketData(**{f: pd.DataFrame(panels[f], index=calendar)[symbols] for f in FIELDS})
