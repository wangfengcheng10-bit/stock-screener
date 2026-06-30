#!/usr/bin/env python3
"""
AI Infrastructure Trading Agent
================================

Scans a curated universe of AI-infrastructure stocks — grouped by sub-sector
(compute/GPUs, memory, equipment, networking/optical, power & cooling, data
centre REITs, hyperscalers) — and builds a daily watchlist of the most
interesting technical setups.

For every name it runs a daily technical pass (moving averages, regression
trend line, volume, Fibonacci, RSI, ADX trend) and, where a setup exists,
produces a full trade plan: ENTRY zone, STOP LOSS and TAKE-PROFIT targets
with risk/reward.

Each run pulls free market data (yfinance / Yahoo) and writes a timestamped
watchlist to ./watchlists/ as JSON + CSV.

Usage
-----
    python3 ai_infra_agent.py                       # scan everything
    python3 ai_infra_agent.py --sector power        # only one sub-sector (substring match)
    python3 ai_infra_agent.py --min-score 7         # only stronger setups
    python3 ai_infra_agent.py --account 25000 --risk-pct 1   # add position sizing
    python3 ai_infra_agent.py --tickers NVDA,AMD,VRT          # custom list

This is NOT financial advice. It is a research tool. Always do your own work.
"""

import os
import csv
import json
import logging
import argparse
from datetime import datetime

import numpy as np
import yfinance as yf

# Keep yfinance's per-ticker download warnings out of the report; we already
# skip unreachable/illiquid names quietly.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# Reuse the proven analysis primitives from the existing screener.
import stock_screener as ss


# ── AI-infrastructure universe (grouped by sub-sector) ────────────────────────
# Curated, US-listed tickers that Yahoo serves. Edit freely — the agent simply
# scans whatever is here. Recent IPOs without 2y of history are skipped quietly.

AI_INFRA_UNIVERSE = {
    "Compute · GPUs & AI Accelerators": ["NVDA", "AMD", "AVGO", "TSM", "ARM", "INTC"],
    "Custom Silicon · ASIC / Connectivity": ["MRVL", "ALAB", "CRDO", "SMCI"],
    "Memory & Storage (HBM)":            ["MU", "WDC", "STX", "SNDK"],
    "Semiconductor Equipment":           ["ASML", "AMAT", "LRCX", "KLAC", "TER", "ENTG"],
    "Networking & Optical Interconnect": ["ANET", "CIEN", "COHR", "LITE", "APH"],
    "Power & Electrification (AI load)": ["VRT", "ETN", "PWR", "GEV", "ENPH"],
    "Independent Power / Nuclear":       ["CEG", "VST", "NRG", "TLN"],
    "Data-Centre REITs":                 ["EQIX", "DLR"],
    "Hyperscalers / Cloud":              ["MSFT", "GOOGL", "AMZN", "META", "ORCL"],
}


# ── Extra indicators (not in the base screener) ───────────────────────────────

def _swing_highs(close, window=8):
    """Local price maxima — used to map overhead resistance / take-profit zones."""
    prices = close.values
    highs = []
    for i in range(window, len(prices) - window):
        if prices[i] == prices[i - window:i + window + 1].max():
            highs.append({"price": float(prices[i]), "idx": i})
    return highs


def _resistance_above(close, price, tolerance=0.025):
    """Cluster swing highs sitting above the current price → resistance levels,
    nearest first. These are natural places for price to stall / take profit."""
    highs = [h for h in _swing_highs(close) if h["price"] > price * 1.005]
    highs.sort(key=lambda h: h["price"])
    levels, used = [], set()
    for i, h in enumerate(highs):
        if i in used:
            continue
        cluster = [h["price"]]
        used.add(i)
        for j in range(i + 1, len(highs)):
            if j in used:
                continue
            if abs(highs[j]["price"] - h["price"]) / h["price"] < tolerance:
                cluster.append(highs[j]["price"])
                used.add(j)
        levels.append(round(sum(cluster) / len(cluster), 2))
    return levels


def trendline_channel(close, lookback=60):
    """Linear-regression trend line over the last `lookback` bars.

    Returns slope (% per week), where price sits versus the fitted line, and
    position within a ±2σ channel (0% = lower rail, 100% = upper rail)."""
    try:
        y = close.iloc[-lookback:].astype(float).values
        if len(y) < 20:
            return {}
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        fitted = slope * x + intercept
        line_now = float(fitted[-1])
        resid = y - fitted
        sigma = float(resid.std()) or 1e-9
        price = float(y[-1])
        # channel position: 0 = lower rail (-2σ), 1 = upper rail (+2σ)
        chan_pos = (price - (line_now - 2 * sigma)) / (4 * sigma)
        mean_price = float(y.mean()) or 1e-9
        return {
            "slope_pct_per_week": round(slope / mean_price * 100 * 5, 2),
            "rising":             slope > 0,
            "price_vs_line_pct":  round((price - line_now) / line_now * 100, 2),
            "channel_pos_pct":    round(max(0.0, min(1.0, chan_pos)) * 100, 0),
        }
    except Exception:
        return {}


def volume_signal(hist):
    """Relative volume and a crude accumulation read from the last session."""
    try:
        vol = hist["Volume"].squeeze().astype(float)
        close = hist["Close"].squeeze().astype(float)
        avg20 = float(vol.iloc[-20:].mean()) or 1e-9
        avg5 = float(vol.iloc[-5:].mean())
        last = float(vol.iloc[-1])
        up_day = float(close.iloc[-1]) >= float(close.iloc[-2])
        return {
            "rel_volume":   round(last / avg20, 2),       # today vs 20-day avg
            "vol_trend":    round(avg5 / avg20, 2),        # 5-day vs 20-day avg
            "up_day":       up_day,
            "confirming":   (last / avg20) > 1.3 and up_day,
        }
    except Exception:
        return {}


# ── Trade setup: entry / stop / take-profit ───────────────────────────────────

def build_trade_setup(hist, price, dza, account=0.0, risk_pct=1.0):
    """Turn demand-zone analysis into a concrete, numeric trade plan.

    Entry comes from the nearest support (demand) zone below price; the stop
    sits just under that zone; take-profit targets are R-multiples of the
    measured risk, cross-checked against overhead resistance and the 52w high.
    """
    close = hist["Close"].squeeze().astype(float)
    zones = dza.get("zones", [])
    hi52 = dza.get("hi52", 0.0)

    # nearest support zone below price (closest = highest 'high' under price)
    below = [z for z in zones if z["high"] <= price * 1.03]
    below.sort(key=lambda z: z["high"], reverse=True)

    if below:
        zone = below[0]
        zone_low, zone_high = zone["low"], zone["high"]
    else:
        # fallback support: recent 20-day low / 50-day MA, whichever is nearer below
        recent_low = float(close.iloc[-20:].min())
        ma50 = float(close.rolling(50).mean().iloc[-1])
        support = max([s for s in (recent_low, ma50) if s < price], default=recent_low)
        zone_low, zone_high = round(support * 0.99, 2), round(support * 1.01, 2)
        zone = {"low": zone_low, "high": zone_high, "touches": 0}

    at_zone = price <= zone_high * 1.03
    # Buy now if we're at the zone, otherwise plan the entry at the top of the zone.
    entry = round(price if at_zone else zone_high, 2)
    stop = round(zone_low * 0.985, 2)
    risk = max(entry - stop, 1e-6)
    risk_pct_trade = round(risk / entry * 100, 1)

    # R-multiple take-profit ladder
    tp1 = round(entry + 2.0 * risk, 2)
    tp2 = round(entry + 3.0 * risk, 2)
    tp3 = round(entry + 5.0 * risk, 2)

    # structural references (where price may actually stall)
    resistance = _resistance_above(close, price)
    nearest_res = resistance[0] if resistance else None

    targets = [
        {"label": "TP1 (2R)", "price": tp1, "rr": 2.0},
        {"label": "TP2 (3R)", "price": tp2, "rr": 3.0},
        {"label": "TP3 (5R, runner)", "price": tp3, "rr": 5.0},
    ]

    # position sizing (optional)
    sizing = None
    if account and account > 0:
        risk_dollars = account * (risk_pct / 100.0)
        shares = int(risk_dollars // risk)
        sizing = {
            "account":        round(account, 2),
            "risk_pct":       risk_pct,
            "risk_dollars":   round(risk_dollars, 2),
            "shares":         shares,
            "position_value": round(shares * entry, 2),
        }

    return {
        "at_zone":        at_zone,
        "entry":          entry,
        "entry_zone":     [zone_low, zone_high],
        "stop":           stop,
        "risk_per_share": round(risk, 2),
        "risk_pct":       risk_pct_trade,
        "targets":        targets,
        "nearest_resistance": nearest_res,
        "all_resistance": resistance[:3],
        "hi52":           hi52,
        "sizing":         sizing,
    }


# ── Setup classification & scoring (what makes a name "interesting") ──────────

def classify_setup(signal):
    s = (signal or "").upper()
    if s.startswith("AT ZONE"):
        return "BUY ZONE"
    if s.startswith("WAIT FOR PULLBACK"):
        return "WATCH · PULLBACK"
    if s.startswith("EXTENDED"):
        return "EXTENDED"
    return "NO SETUP"


def setup_score(rating, setup_class, trend, rsi, vol, tline):
    """0-15ish ranking of how actionable/interesting the setup is right now."""
    score = float(rating or 0)

    score += {"BUY ZONE": 3.0, "WATCH · PULLBACK": 2.0, "EXTENDED": 0.0}.get(setup_class, 0.0)

    label = (trend or {}).get("label", "")
    bullish = (trend or {}).get("bullish", False)
    if "Strong" in label and bullish:
        score += 1.5
    elif "Developing" in label and bullish:
        score += 0.75
    elif not bullish and ("Strong" in label or "Developing" in label):
        score -= 1.5

    if rsi is not None:
        if 35 <= rsi <= 55:   score += 1.0     # clean pullback territory
        elif rsi < 30:        score += 0.5     # oversold bounce
        elif rsi > 70:        score -= 1.0     # overbought, chasing

    if (trend or {}).get("ma_slope", 0) > 0:
        score += 0.5
    if vol.get("confirming"):
        score += 0.5
    if tline.get("rising") and (tline.get("price_vs_line_pct", -1) or 0) > -3:
        score += 0.5

    return round(score, 2)


def is_interesting(setup_class, trend, rating):
    """Goes on the watchlist if it's near a buy zone, or a strong-trend leader."""
    if setup_class in ("BUY ZONE", "WATCH · PULLBACK"):
        return True
    # Don't let momentum leaders fall off just because they're extended.
    label = (trend or {}).get("label", "")
    if setup_class == "EXTENDED" and "Strong" in label and (trend or {}).get("bullish") and rating >= 7:
        return True
    return False


# ── Per-ticker analysis ───────────────────────────────────────────────────────

def analyze_ticker(ticker, sector, account=0.0, risk_pct=1.0):
    hist = ss.get_history(ticker)
    if hist is None:
        return None

    close = hist["Close"].squeeze().astype(float)
    price = float(close.iloc[-1])

    # fundamentals are best-effort (the info endpoint is flaky/slow)
    info = {}
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}
    name = info.get("shortName", ticker)

    t_score, td = ss.technical_score(hist)
    f_score, fd = ss.fundamental_score(info)
    rating = ss.combined_rating(t_score, f_score)
    trend = ss.analyze_trend(hist)
    dza = ss.demand_zone_analysis(hist, price)

    tline = trendline_channel(close)
    vol = volume_signal(hist)
    setup = build_trade_setup(hist, price, dza, account, risk_pct)

    signal = dza.get("entry", {}).get("signal", "")
    setup_class = classify_setup(signal)
    rsi = td.get("rsi")
    score = setup_score(rating, setup_class, trend, rsi, vol, tline)

    return {
        "ticker":       ticker,
        "name":         name,
        "sector":       sector,
        "price":        round(price, 2),
        "rating":       rating,
        "tech_score":   round(t_score, 1),
        "fund_score":   round(f_score, 1),
        "tech_details": td,
        "fund_details": fd,
        "trend":        trend,
        "trendline":    tline,
        "volume":       vol,
        "demand_zone":  dza,
        "setup":        setup,
        "setup_class":  setup_class,
        "setup_score":  score,
        "interesting":  is_interesting(setup_class, trend, rating),
    }


# ── Output ────────────────────────────────────────────────────────────────────

ICON = {
    "BUY ZONE":         "🟢",
    "WATCH · PULLBACK": "🟡",
    "EXTENDED":         "🔵",
    "NO SETUP":         "⚪",
}


def print_watchlist_table(watchlist):
    W = 92
    print(f"\n{'═'*W}")
    print(f"  AI-INFRASTRUCTURE WATCHLIST  —  {len(watchlist)} setups  "
          f"({datetime.now().strftime('%Y-%m-%d')})")
    print(f"{'═'*W}")
    print(f"\n  {'#':<3}{'Tkr':<7}{'Sector':<30}{'Price':>9}  {'Setup':<17}"
          f"{'Score':>6}  {'RSI':>4}  {'Trend'}")
    print(f"  {'─'*88}")
    for i, c in enumerate(watchlist, 1):
        icon = ICON.get(c["setup_class"], " ")
        rsi = c["tech_details"].get("rsi")
        rsi_s = f"{rsi:.0f}" if rsi is not None else " — "
        trend = (c["trend"].get("label", "") or "")[:20]
        print(f"  {i:<3}{c['ticker']:<7}{c['sector'][:29]:<30}${c['price']:>7.2f}  "
              f"{icon} {c['setup_class']:<15}{c['setup_score']:>6}  {rsi_s:>4}  {trend}")


def print_trade_plan(rank, c):
    s = c["setup"]
    tr = c["trend"]
    tl = c["trendline"]
    v = c["volume"]
    td = c["tech_details"]
    dza = c["demand_zone"]
    sep = "─" * 78

    print(f"\n  {ICON.get(c['setup_class'],' ')} #{rank}  {c['ticker']} — {c['name']}")
    print(f"      {c['sector']}")
    print(f"      Rating {c['rating']}/10 (Tech {c['tech_score']} · Fund {c['fund_score']})"
          f"   |   Setup score {c['setup_score']}   |   {c['setup_class']}")
    print(f"  {sep}")

    # daily technical read
    rsi = td.get("rsi")
    parts = []
    if rsi is not None:
        parts.append(f"RSI {rsi:.0f}")
    if td.get("above_ma50") is not None:
        parts.append("> 50MA" if td["above_ma50"] else "< 50MA")
    if td.get("golden_cross") is not None:
        parts.append("Golden Cross" if td["golden_cross"] else "Death Cross")
    if td.get("momentum_3m") is not None:
        parts.append(f"3m {td['momentum_3m']:+.0f}%")
    print(f"    Technical : {'  ·  '.join(parts)}")

    if tr:
        print(f"    Trend     : {tr.get('label','N/A')}  "
              f"(ADX {tr.get('adx','?')}, +DI {tr.get('pdi','?')}/-DI {tr.get('mdi','?')})")
    if tl:
        dir_word = "rising" if tl.get("rising") else "falling"
        print(f"    Trendline : {dir_word} {tl.get('slope_pct_per_week',0):+.1f}%/wk  "
              f"price {tl.get('price_vs_line_pct',0):+.1f}% vs line  "
              f"(channel {tl.get('channel_pos_pct',0):.0f}%)")
    if v:
        conf = "✓ confirming" if v.get("confirming") else "neutral"
        print(f"    Volume    : rel {v.get('rel_volume','?')}×  trend {v.get('vol_trend','?')}×  → {conf}")

    # the trade plan
    el, eh = s["entry_zone"]
    print(f"\n    ►  TRADE PLAN")
    tag = "buy now (at zone)" if s["at_zone"] else "buy on pullback into zone"
    print(f"       Entry      ${el:.2f} – ${eh:.2f}   ({tag})  · ref ${s['entry']:.2f}")
    print(f"       Stop loss  ${s['stop']:.2f}   (−{s['risk_pct']:.1f}% · ${s['risk_per_share']:.2f}/share risk)")
    for t in s["targets"]:
        gain = (t["price"] - s["entry"]) / s["entry"] * 100
        print(f"       {t['label']:<18} ${t['price']:.2f}   (+{gain:.0f}% · {t['rr']:.0f}R)")
    if s["nearest_resistance"]:
        cap = " ⚠ may cap TP1" if s["nearest_resistance"] < s["targets"][0]["price"] else ""
        print(f"       Resistance ${s['nearest_resistance']:.2f} (next overhead){cap}")
    if s.get("hi52"):
        print(f"       52w high   ${s['hi52']:.2f}")
    if s.get("sizing"):
        z = s["sizing"]
        print(f"       Size       {z['shares']} sh ≈ ${z['position_value']:,.0f}  "
              f"(risks ${z['risk_dollars']:,.0f} = {z['risk_pct']}% of ${z['account']:,.0f})")

    # demand zones + fib context
    fibs = dza.get("fibs", {})
    if fibs:
        near = [f"{k} ${val}" for k, val in fibs.items()
                if abs((c['price'] - val) / c['price']) < 0.04]
        if near:
            print(f"       Fib (near) {'  '.join(near)}")


def save_watchlist(watchlist, outdir="watchlists"):
    os.makedirs(outdir, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d")
    rows = []
    for c in watchlist:
        s = c["setup"]
        rows.append({
            "ticker":      c["ticker"],
            "name":        c["name"],
            "sector":      c["sector"],
            "price":       c["price"],
            "setup":       c["setup_class"],
            "setup_score": c["setup_score"],
            "rating":      c["rating"],
            "rsi":         c["tech_details"].get("rsi"),
            "trend":       c["trend"].get("label"),
            "entry_low":   s["entry_zone"][0],
            "entry_high":  s["entry_zone"][1],
            "stop":        s["stop"],
            "tp1":         s["targets"][0]["price"],
            "tp2":         s["targets"][1]["price"],
            "tp3":         s["targets"][2]["price"],
            "risk_pct":    s["risk_pct"],
        })

    json_path = os.path.join(outdir, f"ai_infra_watchlist_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump({"generated": datetime.now().isoformat(timespec="seconds"),
                   "count": len(rows), "watchlist": rows}, f, indent=2)

    csv_path = os.path.join(outdir, f"ai_infra_watchlist_{stamp}.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    return json_path, csv_path


# ── Main ──────────────────────────────────────────────────────────────────────

def build_universe(args):
    if args.tickers:
        return {"Custom": [t.strip().upper() for t in args.tickers.split(",") if t.strip()]}
    if args.sector:
        key = args.sector.lower()
        sel = {k: v for k, v in AI_INFRA_UNIVERSE.items() if key in k.lower()}
        return sel or AI_INFRA_UNIVERSE
    return AI_INFRA_UNIVERSE


def main():
    p = argparse.ArgumentParser(description="AI-Infrastructure Trading Agent")
    p.add_argument("--sector", help="only scan sub-sectors matching this substring")
    p.add_argument("--tickers", help="comma-separated custom ticker list (overrides sectors)")
    p.add_argument("--min-score", type=float, default=0.0, help="min setup score for the watchlist")
    p.add_argument("--max-names", type=int, default=15, help="cap watchlist length")
    p.add_argument("--account", type=float, default=0.0, help="account size for position sizing")
    p.add_argument("--risk-pct", type=float, default=1.0, help="%% of account risked per trade")
    p.add_argument("--no-save", action="store_true", help="don't write watchlist files")
    args = p.parse_args()

    universe = build_universe(args)
    total = sum(len(v) for v in universe.values())

    W = 92
    print(f"\n{'═'*W}")
    print(f"  AI-INFRASTRUCTURE TRADING AGENT   ·   daily technical scan")
    print(f"  {total} names across {len(universe)} sub-sectors   ·   {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'═'*W}\n")

    results, scanned, done = [], 0, 0
    for sector, tickers in universe.items():
        print(f"  ▸ {sector}")
        for ticker in tickers:
            done += 1
            print(f"      [{done:>2}/{total}] {ticker:<6}", end="\r")
            try:
                c = analyze_ticker(ticker, sector, args.account, args.risk_pct)
            except Exception:
                c = None
            if c:
                scanned += 1
                results.append(c)
        print(" " * 40, end="\r")

    if not results:
        print("\n  No data returned. Yahoo may be unreachable from this network "
              "(the screener needs outbound access to query.finance.yahoo.com).")
        return

    # build the watchlist
    watchlist = [c for c in results if c["interesting"] and c["setup_score"] >= args.min_score]
    watchlist.sort(key=lambda c: c["setup_score"], reverse=True)
    watchlist = watchlist[:args.max_names]

    print_watchlist_table(watchlist)

    print(f"\n\n{'═'*W}")
    print(f"  TRADE PLANS   (entry · stop · take-profit)")
    print(f"{'═'*W}")
    if watchlist:
        for rank, c in enumerate(watchlist, 1):
            print_trade_plan(rank, c)
    else:
        print("\n  Nothing meets the watchlist criteria today — no clean pullbacks or"
              " buy zones. Sit on your hands.")

    if not args.no_save and watchlist:
        jp, cp = save_watchlist(watchlist)
        print(f"\n  Saved → {jp}")
        print(f"          {cp}")

    print(f"\n{'═'*W}")
    print(f"  Scanned {scanned}/{total} names · {len(watchlist)} on watchlist · "
          f"Legend: 🟢 buy zone  🟡 pullback  🔵 momentum/extended")
    print(f"  Not financial advice. Free Yahoo data, delayed. Always do your own research.")
    print(f"{'═'*W}\n")


if __name__ == "__main__":
    main()
