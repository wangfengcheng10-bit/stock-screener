"""NASDAQ 100 daily gap analysis.

Downloads the *maximum* available daily history for the NASDAQ 100 index
(^NDX) from Yahoo Finance, detects every true price gap between two
consecutive trading days, and classifies each gap by how it was later
resolved:

  * FULLY  CLOSED  — price later traded all the way back through the gap,
                     filling the empty space completely.
  * PARTIALLY CLOSED — price traded *into* the gap but never all the way
                     across it (the gap was dented but not filled).
  * STILL OPEN     — price has not re-entered the gap zone at all yet.

A "true gap" (a.k.a. a real chart gap / window) is defined as an actual
empty band on the chart:

  * Gap UP   : today's LOW  > yesterday's HIGH   -> empty band
               (yesterday_high, today_low)
  * Gap DOWN : today's HIGH < yesterday's LOW    -> empty band
               (today_high, yesterday_low)

Fill logic (looking only at bars *after* the gap day):
  * Gap UP is fully filled once a later LOW  <= yesterday_high (gap bottom).
             It is partially filled if a later LOW  <  today_low (entered
             the band) but never reached the bottom.
  * Gap DOWN is fully filled once a later HIGH >= yesterday_low (gap top).
             It is partially filled if a later HIGH >  today_high (entered
             the band) but never reached the top.

Usage:
    python3 nasdaq100_gap_analysis.py
    python3 nasdaq100_gap_analysis.py --ticker ^NDX --min-gap-pct 0.0
"""

import argparse
import sys

import numpy as np
import pandas as pd
import yfinance as yf


def get_history(ticker: str) -> pd.DataFrame:
    """Download the maximum available daily OHLC history for `ticker`."""
    df = yf.download(
        ticker,
        period="max",
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if df is None or df.empty:
        raise SystemExit(f"No data returned for {ticker!r}.")
    # yfinance returns a ticker-level MultiIndex on the columns; flatten it.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    return df


def load_csv(path: str) -> pd.DataFrame:
    """Load daily OHLC history from a local CSV.

    Column names are matched case-insensitively. A `Date` column becomes the
    index. Raw (unadjusted) Open/High/Low/Close are used deliberately: gap
    detection is about the prices actually printed on the chart, so split /
    dividend adjustment would invent or erase gaps on ex-dates.
    """
    raw = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in raw.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        raise SystemExit(
            f"CSV {path!r} is missing a required column (one of {names}). "
            f"Found: {list(raw.columns)}"
        )

    date_c = pick("date", "datetime", "timestamp")
    o, h, l, c = (pick("open"), pick("high"), pick("low"), pick("close"))
    keep = [date_c, o, h, l, c]
    rename = {date_c: "Date", o: "Open", h: "High", l: "Low", c: "Close"}
    split_c = cols.get("split")
    if split_c:
        keep.append(split_c)
        rename[split_c] = "Split"
    df = raw[keep].copy().rename(columns=rename)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df.sort_values("Date").set_index("Date")
    if df.empty:
        raise SystemExit(f"No usable OHLC rows in {path!r}.")

    # Back-adjust OHLC for stock splits so a split ex-date is not mistaken for
    # a price gap. Dividends are intentionally NOT adjusted out: an ex-dividend
    # gap is a real overnight move on the chart. The split factor for a bar is
    # the product of every split ratio that takes effect *after* that bar.
    if "Split" in df.columns and (df["Split"] != 1.0).any():
        rev_cum = df["Split"][::-1].cumprod().shift(1).fillna(1.0)[::-1]
        for col in ("Open", "High", "Low", "Close"):
            df[col] = df[col] / rev_cum
        df = df.drop(columns=["Split"])
    elif "Split" in df.columns:
        df = df.drop(columns=["Split"])
    return df


def find_gaps(df: pd.DataFrame, min_gap_pct: float = 0.0) -> pd.DataFrame:
    """Return one row per true gap between consecutive trading days.

    `min_gap_pct` filters out microscopic gaps (as a % of the prior close).
    """
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    dates = df.index

    records = []
    n = len(df)
    for i in range(1, n):
        prev_high = high[i - 1]
        prev_low = low[i - 1]
        prev_close = close[i - 1]
        today_high = high[i]
        today_low = low[i]

        gap_up = today_low > prev_high
        gap_down = today_high < prev_low
        if not (gap_up or gap_down):
            continue

        if gap_up:
            direction = "up"
            gap_bottom = prev_high      # bottom edge of empty band
            gap_top = today_low         # top edge of empty band
        else:
            direction = "down"
            gap_bottom = today_high
            gap_top = prev_low

        gap_size = gap_top - gap_bottom
        gap_pct = gap_size / prev_close * 100.0
        if gap_pct < min_gap_pct:
            continue

        # ---- Look forward to classify how the gap was resolved. ----
        fut_high = high[i + 1:]
        fut_low = low[i + 1:]

        status = "still_open"
        days_to_full = np.nan
        if direction == "up":
            # entered the band?
            entered = fut_low < gap_top
            # fully filled once a low reaches the bottom edge
            full_mask = fut_low <= gap_bottom
            if full_mask.any():
                status = "full"
                days_to_full = int(np.argmax(full_mask)) + 1
            elif entered.any():
                status = "partial"
        else:
            entered = fut_high > gap_bottom
            full_mask = fut_high >= gap_top
            if full_mask.any():
                status = "full"
                days_to_full = int(np.argmax(full_mask)) + 1
            elif entered.any():
                status = "partial"

        records.append(
            {
                "date": dates[i],
                "direction": direction,
                "prev_close": prev_close,
                "gap_bottom": gap_bottom,
                "gap_top": gap_top,
                "gap_size": gap_size,
                "gap_pct": gap_pct,
                "status": status,
                "days_to_full": days_to_full,
            }
        )

    return pd.DataFrame(records)


def _fmt_row(label, count, total):
    pct = (count / total * 100.0) if total else 0.0
    return f"{label:<26}{count:>8,}{pct:>11.1f}%"


def summarize(gaps: pd.DataFrame, df: pd.DataFrame, ticker: str) -> str:
    lines = []
    total = len(gaps)
    start = df.index[0].date()
    end = df.index[-1].date()
    n_days = len(df)

    full = int((gaps["status"] == "full").sum())
    partial = int((gaps["status"] == "partial").sum())
    still_open = int((gaps["status"] == "still_open").sum())

    up = gaps[gaps["direction"] == "up"]
    down = gaps[gaps["direction"] == "down"]

    lines.append("=" * 60)
    lines.append(f"  DAILY GAP ANALYSIS  —  {ticker}")
    lines.append("=" * 60)
    lines.append(f"  History window : {start}  ->  {end}")
    lines.append(f"  Trading days   : {n_days:,}")
    lines.append(f"  True gaps found: {total:,}")
    if total:
        lines.append(
            f"  Gap frequency  : {total / n_days * 100:.1f}% of days "
            f"(1 every {n_days / total:.1f} days)"
        )
    lines.append("")

    # ---- Main classification table ----
    lines.append("  GAP RESOLUTION (all gaps)")
    lines.append("  " + "-" * 46)
    lines.append(f"  {'Category':<26}{'Count':>8}{'Share':>12}")
    lines.append("  " + "-" * 46)
    lines.append("  " + _fmt_row("Closed completely (full)", full, total))
    lines.append("  " + _fmt_row("Closed partially", partial, total))
    lines.append("  " + _fmt_row("Still open (never filled)", still_open, total))
    lines.append("  " + "-" * 46)
    lines.append("  " + _fmt_row("TOTAL", total, total))
    lines.append("")

    # ---- Breakdown by direction ----
    lines.append("  BY DIRECTION")
    lines.append("  " + "-" * 56)
    lines.append(
        f"  {'Direction':<12}{'Gaps':>8}{'Full':>8}{'Partial':>9}{'Open':>7}"
        f"{'%Full':>8}"
    )
    lines.append("  " + "-" * 56)
    for name, sub in (("Gap up", up), ("Gap down", down)):
        t = len(sub)
        f = int((sub["status"] == "full").sum())
        p = int((sub["status"] == "partial").sum())
        o = int((sub["status"] == "still_open").sum())
        pf = (f / t * 100) if t else 0.0
        lines.append(
            f"  {name:<12}{t:>8,}{f:>8,}{p:>9,}{o:>7,}{pf:>7.1f}%"
        )
    lines.append("  " + "-" * 56)
    lines.append("")

    # ---- Time-to-fill stats for the fully closed gaps ----
    filled = gaps[gaps["status"] == "full"]
    if not filled.empty:
        d = filled["days_to_full"]
        lines.append("  TIME TO FULLY CLOSE (filled gaps only)")
        lines.append("  " + "-" * 46)
        lines.append(f"  {'Median trading days':<30}{int(d.median()):>10,}")
        lines.append(f"  {'Mean trading days':<30}{d.mean():>10.1f}")
        lines.append(f"  {'Filled same/next day (<=1)':<30}"
                     f"{int((d <= 1).sum()):>10,}")
        lines.append(f"  {'Filled within 5 days':<30}"
                     f"{int((d <= 5).sum()):>10,}")
        lines.append(f"  {'Filled within 21 days (~1mo)':<30}"
                     f"{int((d <= 21).sum()):>10,}")
        lines.append(f"  {'Longest to fill':<30}{int(d.max()):>10,}")
        lines.append("")

    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="NASDAQ 100 daily gap analysis")
    ap.add_argument("--ticker", default="^NDX",
                    help="Yahoo Finance symbol (default: ^NDX = NASDAQ 100)")
    ap.add_argument("--min-gap-pct", type=float, default=0.0,
                    help="Ignore gaps smaller than this %% of prior close")
    ap.add_argument("--csv-input", default=None,
                    help="Analyse a local daily-OHLC CSV instead of Yahoo "
                         "(needs Date/Open/High/Low/Close columns)")
    ap.add_argument("--label", default=None,
                    help="Display name for the series (default: ticker or file)")
    ap.add_argument("--csv", default=None,
                    help="Optional path to write the full per-gap table as CSV")
    args = ap.parse_args(argv)

    if args.csv_input:
        print(f"Loading daily history from {args.csv_input} ...",
              file=sys.stderr)
        df = load_csv(args.csv_input)
        label = args.label or args.csv_input
    else:
        print(f"Downloading maximum daily history for {args.ticker} ...",
              file=sys.stderr)
        df = get_history(args.ticker)
        label = args.label or args.ticker
    gaps = find_gaps(df, min_gap_pct=args.min_gap_pct)

    print(summarize(gaps, df, label))

    if args.csv:
        gaps.to_csv(args.csv, index=False)
        print(f"\nPer-gap detail written to {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
