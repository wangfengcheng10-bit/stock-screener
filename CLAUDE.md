# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the screener

```bash
python3 stock_screener.py QQQ        # pass ETF ticker directly
python3 stock_screener.py            # interactive prompt
```

Dependencies: `yfinance`, `numpy`, `pandas` (all installed globally via pip3).

## Running the AI-infrastructure trading agent

```bash
python3 ai_infra_agent.py                      # scan the full AI-infra universe
python3 ai_infra_agent.py --sector power       # one sub-sector (substring match)
python3 ai_infra_agent.py --tickers NVDA,VRT   # custom list
python3 ai_infra_agent.py --account 25000 --risk-pct 1   # add position sizing
python3 ai_infra_agent.py --min-score 8 --max-names 10   # only the strongest setups
```

`ai_infra_agent.py` builds a daily watchlist of AI-infrastructure trade setups.
It reuses the screener's primitives (`import stock_screener as ss`) and adds the
trade-planning layer the screener lacks (take-profit targets, regression
trend line, volume confirmation, position sizing, watchlist persistence).

## AI-infra agent architecture

`ai_infra_agent.py` scans a curated `AI_INFRA_UNIVERSE` (dict of sub-sector →
ticker list: GPUs/compute, custom silicon, memory/HBM, semi equipment,
networking/optical, power & cooling, independent power/nuclear, data-centre
REITs, hyperscalers) instead of an ETF's holdings. Per ticker:

1. **Reused primitives** — `analyze_ticker()` calls `ss.get_history`,
   `ss.technical_score`, `ss.fundamental_score`, `ss.analyze_trend`,
   `ss.demand_zone_analysis`. Price is taken from the last `Close` (robust);
   `yf.Ticker.info` is best-effort and only feeds fundamentals.
2. **Added indicators** — `trendline_channel()` (linear-regression trend line,
   slope %/week, ±2σ channel position), `volume_signal()` (relative volume +
   accumulation read), `_resistance_above()` (clustered swing highs → overhead
   targets).
3. **Trade plan** — `build_trade_setup()` derives a numeric ENTRY zone from the
   nearest demand zone below price, a STOP just under it, and a TAKE-PROFIT
   ladder at 2R/3R/5R cross-checked against resistance and the 52w high.
   Optional `--account`/`--risk-pct` add share-count position sizing.
4. **Watchlist** — `is_interesting()` keeps buy-zone/pullback setups (plus
   strong-trend leaders); `setup_score()` ranks them. `save_watchlist()` writes
   `watchlists/ai_infra_watchlist_<date>.{json,csv}` each run.

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
