# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the screener

```bash
python3 stock_screener.py QQQ                 # pass ETF ticker directly
python3 stock_screener.py                     # interactive prompt
python3 stock_screener.py --compare MU SNDK   # side-by-side fundamental comparison of 2+ stocks
```

`--compare` (`-c`) skips ETF/holdings logic and runs `compare_fundamentals()`, which scores each ticker with `fundamental_score()` and prints a side-by-side table (best value per metric marked with ◄) plus a verdict.

Dependencies: `yfinance`, `numpy`, `pandas` (all installed globally via pip3).

## Architecture

`stock_screener.py` is a single-file CLI tool. Data flows in one pass per ticker:

1. **ETF holdings** — `get_etf_holdings()` fetches the ETF's top ~30 holdings via `yf.Ticker.funds_data.top_holdings`, with a fallback to `info["holdings"]`.
2. **Price history** — `get_history()` downloads 2 years of daily OHLCV once per ticker and passes the DataFrame to all downstream functions (avoids redundant network calls).
3. **Scoring** — two independent scores, each 0–5:
   - `technical_score(hist)` — RSI, 50/200 MA position, 3-month momentum
   - `fundamental_score(info)` — revenue growth, gross margins, ROE, PEG, earnings growth
   - `combined_rating()` sums them to a 1–10 scale (one decimal)
4. **Trend** — `analyze_trend(hist)` runs ADX/±DI to classify direction and strength (Strong/Developing/Ranging × Uptrend/Downtrend).
5. **Demand zones & entry** — `demand_zone_analysis(hist, price)`:
   - `_swing_lows()` finds local price minima (8-bar window each side)
   - `_cluster_zones()` groups lows within 2.5% tolerance, scores by touches × volume × recency
   - `_fibonacci_levels()` computes 23.6/38.2/50/61.8% retracements from 52-week high→low
   - Entry signal is one of: **AT ZONE** (within 3%), **WAIT FOR PULLBACK** (within 12%), **EXTENDED**
6. **Output** — summary table of top 7, then `print_detail()` per stock covering all four sections.

## Key design constraints

- History is fetched with `period="2y"` to give swing-low detection enough price history; changing to `"1y"` will degrade demand zone quality.
- All `hist` columns are squeezed (`.squeeze()`) because yfinance returns single-column DataFrames with a ticker-level MultiIndex that breaks scalar operations.
- Zone strength scoring weights recency heavily (`0.3 + 0.7 * recency`) so recent support levels rank above ancient ones with similar touch counts.
