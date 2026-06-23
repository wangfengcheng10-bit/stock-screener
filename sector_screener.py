#!/usr/bin/env python3
"""
sector_screener.py — Weekly Sector Stock Scanner

Usage:
  python3 sector_screener.py technology
  python3 sector_screener.py XLK
  python3 sector_screener.py             # interactive prompt

Sectors: technology, healthcare, energy, financials, industrials,
         consumer_discretionary, consumer_staples, utilities,
         real_estate, materials, communication
"""

import yfinance as yf
import numpy as np
import pandas as pd
import sys
import argparse
from datetime import datetime


# ── Sector Definitions ────────────────────────────────────────────────────────

SECTOR_MAP = {
    "technology":             "XLK",
    "tech":                   "XLK",
    "healthcare":             "XLV",
    "health":                 "XLV",
    "energy":                 "XLE",
    "financials":             "XLF",
    "finance":                "XLF",
    "industrials":            "XLI",
    "industrial":             "XLI",
    "consumer_discretionary": "XLY",
    "discretionary":          "XLY",
    "consumer_staples":       "XLP",
    "staples":                "XLP",
    "utilities":              "XLU",
    "real_estate":            "XLRE",
    "realestate":             "XLRE",
    "materials":              "XLB",
    "communication":          "XLC",
    "communications":         "XLC",
    "comm_services":          "XLC",
}

SECTOR_NAMES = {
    "XLK":  "Technology",
    "XLV":  "Healthcare",
    "XLE":  "Energy",
    "XLF":  "Financials",
    "XLI":  "Industrials",
    "XLY":  "Consumer Discretionary",
    "XLP":  "Consumer Staples",
    "XLU":  "Utilities",
    "XLRE": "Real Estate",
    "XLB":  "Materials",
    "XLC":  "Communication Services",
}

W = 74   # output line width


# ── Sector Resolution ─────────────────────────────────────────────────────────

def resolve_sector(arg):
    """Return (etf_ticker, sector_name) from a sector name or ETF ticker."""
    clean = arg.lower().strip().replace(" ", "_")
    if clean in SECTOR_MAP:
        etf = SECTOR_MAP[clean]
        return etf, SECTOR_NAMES.get(etf, etf)
    upper = arg.upper().strip()
    if upper in SECTOR_NAMES:
        return upper, SECTOR_NAMES[upper]
    return upper, SECTOR_NAMES.get(upper, upper)


# ── Holdings ──────────────────────────────────────────────────────────────────

def get_sector_holdings(etf_ticker, max_stocks=25):
    """Fetch top holdings from a sector ETF."""
    etf = yf.Ticker(etf_ticker)
    try:
        holdings_df = etf.funds_data.top_holdings
        if holdings_df is not None and not holdings_df.empty:
            tickers = [t for t in holdings_df.index if isinstance(t, str) and len(t) <= 6]
            if tickers:
                return tickers[:max_stocks]
    except Exception:
        pass
    try:
        raw = etf.info.get("holdings", [])
        tickers = [h.get("symbol") for h in raw if h.get("symbol")]
        if tickers:
            return tickers[:max_stocks]
    except Exception:
        pass
    return []


# ── Weekly History ────────────────────────────────────────────────────────────

def get_weekly_history(ticker, period="3y"):
    """Download weekly OHLCV. Returns None if data is insufficient."""
    hist = yf.download(ticker, period=period, interval="1wk",
                       progress=False, auto_adjust=True)
    if hist.empty or len(hist) < 30:
        return None
    # Flatten ticker-level MultiIndex columns produced by yfinance
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    return hist


# ── Indicators ────────────────────────────────────────────────────────────────

def _rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def _adx(hist, period=14):
    """ADX + ±DI on an OHLCV DataFrame (weekly or daily)."""
    try:
        high  = hist["High"].squeeze()
        low   = hist["Low"].squeeze()
        close = hist["Close"].squeeze()

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)

        up   = high.diff()
        down = -low.diff()
        pdm  = np.where((up > down) & (up > 0),   up,   0.0)
        mdm  = np.where((down > up) & (down > 0), down, 0.0)

        atr  = tr.rolling(period).mean()
        pdi  = 100 * pd.Series(pdm, index=tr.index).rolling(period).mean() / atr
        mdi  = 100 * pd.Series(mdm, index=tr.index).rolling(period).mean() / atr
        dx   = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
        adx  = float(dx.rolling(period).mean().iloc[-1])
        return adx, float(pdi.iloc[-1]), float(mdi.iloc[-1])
    except Exception:
        return 0.0, 0.0, 0.0


# ── Support & Resistance (Weekly) ─────────────────────────────────────────────

def _find_swing_points(series, window=5):
    """Find local maxima (resistance) and minima (support) in a price series."""
    vals = series.values
    n    = len(vals)
    highs, lows = [], []
    for i in range(window, n - window):
        chunk = vals[i - window: i + window + 1]
        if vals[i] == chunk.max():
            highs.append({"price": float(vals[i]), "idx": i})
        if vals[i] == chunk.min():
            lows.append({"price": float(vals[i]), "idx": i})
    return highs, lows


def _cluster_levels(points, tolerance=0.025):
    """
    Merge nearby price levels into zones.
    Scores each zone by number of touches × recency weight.
    """
    if not points:
        return []
    sorted_pts = sorted(points, key=lambda x: x["price"])
    used, zones = set(), []
    n = len(sorted_pts)

    for i in range(n):
        if i in used:
            continue
        cluster = [sorted_pts[i]]
        used.add(i)
        base = sorted_pts[i]["price"]
        for j in range(i + 1, n):
            if j in used:
                continue
            if abs(sorted_pts[j]["price"] - base) / base < tolerance:
                cluster.append(sorted_pts[j])
                used.add(j)

        prices  = [c["price"] for c in cluster]
        idxs    = [c["idx"]   for c in cluster]
        recency = max(idxs) / max(n, 1)
        zones.append({
            "price":    round(float(np.mean(prices)), 2),
            "low":      round(min(prices) * 0.990, 2),
            "high":     round(max(prices) * 1.010, 2),
            "touches":  len(cluster),
            "recency":  round(recency, 2),
            "strength": round(len(cluster) * (0.3 + 0.7 * recency), 2),
        })

    return sorted(zones, key=lambda z: z["strength"], reverse=True)


def support_resistance_weekly(weekly_hist, current_price):
    """Return clustered support and resistance zones from weekly OHLCV."""
    try:
        close = weekly_hist["Close"].squeeze()
        high  = weekly_hist["High"].squeeze()
        low   = weekly_hist["Low"].squeeze()

        close_highs, close_lows = _find_swing_points(close, window=5)
        high_highs,  _          = _find_swing_points(high,  window=4)
        _,           low_lows   = _find_swing_points(low,   window=4)

        supports    = _cluster_levels(close_lows + low_lows,  tolerance=0.025)
        resistances = _cluster_levels(close_highs + high_highs, tolerance=0.025)

        # Split by current price
        supports    = [z for z in supports    if z["price"] < current_price * 1.01]
        resistances = [z for z in resistances if z["price"] > current_price * 0.99]

        nearest_sup = max(supports,    key=lambda z: z["price"]) if supports    else None
        nearest_res = min(resistances, key=lambda z: z["price"]) if resistances else None

        return {
            "supports":           supports[:4],
            "resistances":        resistances[:3],
            "nearest_support":    nearest_sup,
            "nearest_resistance": nearest_res,
        }
    except Exception:
        return {"supports": [], "resistances": [],
                "nearest_support": None, "nearest_resistance": None}


# ── Volume Profile ────────────────────────────────────────────────────────────

def volume_profile(weekly_hist, n_bins=36, lookback_weeks=52):
    """
    Compute volume profile: Point of Control (POC), Value Area High/Low.
    Assigns each week's volume to its typical-price bin, then finds the
    bin with most volume (POC) and expands to capture 70% of total volume.
    """
    try:
        data   = weekly_hist.iloc[-lookback_weeks:].copy()
        high   = data["High"].squeeze().values.astype(float)
        low    = data["Low"].squeeze().values.astype(float)
        close  = data["Close"].squeeze().values.astype(float)
        volume = data["Volume"].squeeze().values.astype(float)

        typical   = (high + low + close) / 3
        price_min = typical.min()
        price_max = typical.max()
        if price_max <= price_min:
            return {}

        bins        = np.linspace(price_min, price_max, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        vol_hist    = np.zeros(n_bins)

        for tp, v in zip(typical, volume):
            idx = min(int((tp - price_min) / (price_max - price_min) * n_bins), n_bins - 1)
            vol_hist[idx] += v

        poc_idx   = int(np.argmax(vol_hist))
        poc_price = float(bin_centers[poc_idx])

        # Expand outward from POC until 70% of volume is captured
        total_vol  = vol_hist.sum()
        target_vol = total_vol * 0.70
        va_lo      = poc_idx
        va_hi      = poc_idx
        va_vol     = vol_hist[poc_idx]

        while va_vol < target_vol and (va_lo > 0 or va_hi < n_bins - 1):
            vol_below = vol_hist[va_lo - 1] if va_lo > 0 else 0.0
            vol_above = vol_hist[va_hi + 1] if va_hi < n_bins - 1 else 0.0
            if vol_above >= vol_below:
                va_hi += 1
                va_vol += vol_hist[va_hi]
            else:
                va_lo -= 1
                va_vol += vol_hist[va_lo]

        return {
            "poc": round(poc_price, 2),
            "vah": round(float(bin_centers[va_hi]), 2),
            "val": round(float(bin_centers[va_lo]), 2),
        }
    except Exception:
        return {}


# ── Weekly Technical Score ────────────────────────────────────────────────────

def weekly_technical_score(weekly_hist, current_price, sr_data, vp_data):
    """
    Score the technical setup on the weekly chart (0–5).
    Returns (score, setup_label, detail_dict).
    """
    try:
        close = weekly_hist["Close"].squeeze()
        score = 0.0
        d     = {}

        # Weekly RSI
        rsi = _rsi(close, period=14)
        d["weekly_rsi"] = round(rsi, 1)
        if 40 <= rsi <= 60:      score += 1.0   # neutral / healthy
        elif 30 <= rsi < 40:     score += 0.5   # mildly oversold, bounce candidate
        elif 60 < rsi <= 70:     score += 0.5   # bullish but stretched

        # Weekly MA50 — price position
        ma50      = close.rolling(50).mean()
        ma50_last = float(ma50.iloc[-1])
        above_ma  = current_price > ma50_last
        d["above_weekly_ma50"] = above_ma
        d["ma50_val"]          = round(ma50_last, 2)
        if above_ma:
            score += 1.0

        # Weekly MA50 — slope (5-week look-back)
        if len(ma50.dropna()) >= 5:
            slope = (float(ma50.iloc[-1]) - float(ma50.iloc[-5])) / float(ma50.iloc[-5]) * 100
            d["ma50_slope"] = round(slope, 2)
            if slope > 0.5:    score += 0.5
            elif slope > 0.0:  score += 0.25
        else:
            d["ma50_slope"] = None

        # Distance from nearest weekly support zone
        ns = sr_data.get("nearest_support")
        if ns and current_price > 0:
            pct = (current_price - ns["price"]) / current_price * 100
            d["pct_from_support"] = round(pct, 1)
            if pct <= 3:     score += 1.5   # at support
            elif pct <= 7:   score += 1.0   # near support
            elif pct <= 12:  score += 0.5   # approaching support
        else:
            d["pct_from_support"] = None

        # Volume Profile
        if vp_data:
            poc = vp_data.get("poc", 0)
            vah = vp_data.get("vah", 0)
            val = vp_data.get("val", 0)
            d.update({"poc": poc, "vah": vah, "val": val})
            if val <= current_price <= vah:
                score += 0.5               # inside value area
                if current_price >= poc:
                    score += 0.25          # above POC → more bullish
            elif current_price > vah:
                score += 0.25              # above value area → breakout

        # Weekly ADX (trend strength)
        adx, pdi, mdi = _adx(weekly_hist)
        d["adx"]          = round(adx, 1)
        d["bullish_trend"] = pdi > mdi
        if adx > 25 and pdi > mdi:    score += 0.5
        elif adx > 18 and pdi > mdi:  score += 0.25

        # Setup label
        pct_sup    = d.get("pct_from_support") or 100
        bull_trend = d.get("bullish_trend", False)

        if score >= 4.0 and pct_sup <= 5 and above_ma and bull_trend:
            setup = "STRONG BUY SETUP"
        elif score >= 3.0 and pct_sup <= 8 and above_ma:
            setup = "BUY SETUP"
        elif score >= 2.5 and rsi < 40:
            setup = "OVERSOLD — BOUNCE WATCH"
        elif score >= 2.5:
            setup = "WATCH"
        elif pct_sup > 15 or not above_ma:
            setup = "EXTENDED — WAIT FOR PULLBACK"
        else:
            setup = "NEUTRAL"

        return round(min(score, 5.0), 2), setup, d

    except Exception:
        return 0.0, "ERROR", {}


# ── Quarterly Fundamental Score ───────────────────────────────────────────────

def quarterly_fundamental_score(ticker_obj, info):
    """
    Score fundamentals using quarterly financials + EPS beat history (0–5).
    Returns (score, detail_dict).
    """
    score = 0.0
    d     = {}

    # 1. EPS beats in the last 4 quarters
    try:
        eh = ticker_obj.earnings_history
        if eh is not None and not eh.empty:
            beats = 0
            for _, row in eh.tail(4).iterrows():
                est    = row.get("epsEstimate")
                actual = row.get("epsActual")
                if (est is not None and actual is not None
                        and not pd.isna(est) and not pd.isna(actual)):
                    if actual > est:
                        beats += 1
            d["eps_beats_4q"] = beats
            score += min(beats * 0.5, 2.0)
    except Exception:
        d["eps_beats_4q"] = None

    # 2. Revenue growth YoY (latest quarter vs same quarter last year)
    rev_growth = None
    try:
        qf = None
        for attr in ("quarterly_income_stmt", "quarterly_financials"):
            try:
                candidate = getattr(ticker_obj, attr)
                if candidate is not None and not candidate.empty:
                    qf = candidate
                    break
            except Exception:
                pass

        if qf is not None and not qf.empty:
            rev_row = None
            for key in ("Total Revenue", "TotalRevenue", "Revenue"):
                if key in qf.index:
                    rev_row = qf.loc[key]
                    break

            if rev_row is not None and len(rev_row) >= 5:
                q_latest = float(rev_row.iloc[0])
                q_yoy    = float(rev_row.iloc[4])
                if q_yoy > 0 and not np.isnan(q_latest) and not np.isnan(q_yoy):
                    rev_growth = (q_latest - q_yoy) / abs(q_yoy)
                    # QoQ acceleration check
                    if len(rev_row) >= 4:
                        q1, q2, q3 = float(rev_row.iloc[0]), float(rev_row.iloc[1]), float(rev_row.iloc[2])
                        d["rev_qoq_trend"] = "accelerating" if q1 > q2 > q3 else "mixed"
    except Exception:
        pass

    # Fallback to info dict
    if rev_growth is None:
        rev_growth = info.get("revenueGrowth") or 0.0

    d["rev_growth_yoy"] = round(rev_growth * 100, 1)
    if rev_growth > 0.25:    score += 1.5
    elif rev_growth > 0.10:  score += 1.0
    elif rev_growth > 0.05:  score += 0.5

    # 3. Earnings growth YoY
    eg = info.get("earningsGrowth") or 0.0
    d["earnings_growth"] = round(eg * 100, 1)
    if eg > 0.30:    score += 0.75
    elif eg > 0.15:  score += 0.50
    elif eg > 0.05:  score += 0.25

    # 4. Gross margin (profitability quality)
    gm = info.get("grossMargins") or 0.0
    d["gross_margin"] = round(gm * 100, 1)
    if gm > 0.60:    score += 0.75
    elif gm > 0.40:  score += 0.50
    elif gm > 0.25:  score += 0.25

    return round(min(score, 5.0), 2), d


# ── Sector Sentiment ──────────────────────────────────────────────────────────

def sector_sentiment(etf_ticker, pct_above_ma50, pct_bullish_rsi):
    """
    Compute sector sentiment from ETF technicals + stock breadth.
    Returns dict with label, score, and detail dict.
    """
    try:
        etf_hist = get_weekly_history(etf_ticker, period="2y")
        if etf_hist is None:
            return {"label": "UNKNOWN", "score": 0, "details": {}}

        etf_close = etf_hist["Close"].squeeze()
        etf_rsi   = _rsi(etf_close, period=14)
        ma50_last = float(etf_close.rolling(50).mean().iloc[-1])
        etf_price = float(etf_close.iloc[-1])
        adx, pdi, mdi = _adx(etf_hist)
        mom13w = float(etf_close.iloc[-1] / etf_close.iloc[-13] - 1) * 100 \
                 if len(etf_close) >= 13 else 0.0

        score = 0
        if etf_rsi > 60:      score += 2
        elif etf_rsi > 50:    score += 1
        elif etf_rsi < 40:    score -= 2
        elif etf_rsi < 50:    score -= 1

        score += 1 if etf_price > ma50_last else -1

        if adx > 20:
            score += 1 if pdi > mdi else -1

        if pct_above_ma50 >= 0.70:    score += 2
        elif pct_above_ma50 >= 0.55:  score += 1
        elif pct_above_ma50 <= 0.30:  score -= 2
        elif pct_above_ma50 <= 0.45:  score -= 1

        if mom13w > 5:    score += 1
        elif mom13w < -5: score -= 1

        if score >= 5:     label = "STRONGLY BULLISH"
        elif score >= 2:   label = "BULLISH"
        elif score >= 0:   label = "NEUTRAL / MIXED"
        elif score >= -2:  label = "BEARISH"
        else:              label = "STRONGLY BEARISH"

        return {
            "label": label,
            "score": score,
            "details": {
                "etf_rsi":             round(etf_rsi, 1),
                "etf_above_ma50":      etf_price > ma50_last,
                "etf_momentum_13w":    round(mom13w, 1),
                "breadth_above_ma50":  round(pct_above_ma50 * 100, 1),
                "breadth_bullish_rsi": round(pct_bullish_rsi * 100, 1),
                "etf_adx":             round(adx, 1),
                "etf_trend":           "Uptrend" if pdi > mdi else "Downtrend",
            },
        }
    except Exception:
        return {"label": "UNKNOWN", "score": 0, "details": {}}


# ── Output ────────────────────────────────────────────────────────────────────

def _score_bar(value, max_val=5, width=10):
    filled = int(round(float(value) / max_val * width))
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def print_sentiment_header(etf_ticker, sector_name, sentiment):
    label = sentiment.get("label", "UNKNOWN")
    score = sentiment.get("score", 0)
    d     = sentiment.get("details", {})
    icons = {
        "STRONGLY BULLISH": "▲▲",
        "BULLISH":          "▲ ",
        "NEUTRAL / MIXED":  "◆ ",
        "BEARISH":          "▼ ",
        "STRONGLY BEARISH": "▼▼",
    }
    icon = icons.get(label, "◆ ")

    print(f"\n{'═' * W}")
    print(f"  SECTOR SCANNER  ·  {sector_name} ({etf_ticker})  ·  {datetime.now().strftime('%Y-%m-%d')}")
    print(f"{'─' * W}")
    print(f"  SENTIMENT:  {icon} {label}  (score {score:+d})")
    print(f"{'─' * W}")
    print(f"  ETF RSI (weekly): {d.get('etf_rsi', '?')}   │   "
          f"Above MA50: {'Yes' if d.get('etf_above_ma50') else 'No'}   │   "
          f"13w Momentum: {d.get('etf_momentum_13w', 0):+.1f}%")
    print(f"  ETF Trend: {d.get('etf_trend', '?')} (ADX {d.get('etf_adx', '?')})   │   "
          f"Breadth >MA50: {d.get('breadth_above_ma50', '?')}%   │   "
          f"Bullish RSI: {d.get('breadth_bullish_rsi', '?')}%")
    print(f"{'═' * W}")


def print_summary_table(candidates, n_total):
    top = candidates[:8]
    print(f"\n  TOP {len(top)} of {len(candidates)} screened  "
          f"({n_total} holdings total)")
    print(f"  {'─' * 70}")
    print(f"  {'#':<3} {'Ticker':<7} {'Name':<22} {'Price':>8}  "
          f"{'Tech':>4}  {'Fund':>4}  {'Rating':>6}  Setup")
    print(f"  {'─' * 70}")
    for rank, c in enumerate(top, 1):
        price_s = f"${c['price']:.2f}" if c.get("price") else "  N/A"
        setup   = c.get("setup_label", "")[:22]
        print(f"  {rank:<3} {c['ticker']:<7} {c['name'][:21]:<22} {price_s:>8}  "
              f"{c['tech_score']:>3.1f}  {c['fund_score']:>3.1f}  "
              f"{c['rating']:>5.1f}/10  {setup}")
    print(f"  {'─' * 70}")


def print_stock_detail(rank, c):
    td    = c.get("tech_details", {})
    fd    = c.get("fund_details", {})
    sr    = c.get("sr_data", {})
    price = c.get("price") or 0
    sep   = "─" * (W - 2)

    print(f"\n  RANK #{rank}  ·  {c['ticker']} — {c['name']}")
    print(f"  {'Overall Rating':18} {c['rating']}/10  "
          f"(Technical: {c['tech_score']}/5  |  Fundamental: {c['fund_score']}/5)")
    t_bar = _score_bar(c["tech_score"])
    f_bar = _score_bar(c["fund_score"])
    print(f"  T {t_bar}  F {f_bar}  Setup: {c.get('setup_label', '')}")
    print(f"  {sep}")

    # ── Technical (Weekly Chart) ──────────────────────────────────────────────
    print("  TECHNICAL  (weekly chart)")

    rsi = td.get("weekly_rsi")
    if rsi is not None:
        if rsi < 30:     rsi_note = "OVERSOLD — reversal zone"
        elif rsi <= 40:  rsi_note = "mildly oversold — bounce candidate"
        elif rsi <= 55:  rsi_note = "healthy / neutral"
        elif rsi <= 65:  rsi_note = "bullish momentum"
        elif rsi <= 75:  rsi_note = "getting overbought"
        else:            rsi_note = "OVERBOUGHT — elevated risk"
        print(f"    RSI (14w):        {rsi:.1f}  →  {rsi_note}")

    ma50_val = td.get("ma50_val")
    above    = td.get("above_weekly_ma50")
    if above is not None and ma50_val is not None:
        pos  = "ABOVE" if above else "BELOW"
        note = "bullish structure" if above else "bearish structure"
        print(f"    50-Week MA:       {pos} ${ma50_val:,.2f}  →  {note}")

    slope = td.get("ma50_slope")
    if slope is not None:
        sdir = "rising" if slope > 0 else "falling"
        print(f"    MA50 slope (5w):  {slope:+.2f}%  →  {sdir}")

    adx = td.get("adx")
    if adx is not None:
        tdir = "Uptrend" if td.get("bullish_trend") else "Downtrend"
        tstr = "strong" if adx > 25 else "developing" if adx > 18 else "weak / ranging"
        print(f"    Trend (ADX):      {adx:.1f} — {tdir} ({tstr})")

    # Volume Profile
    poc = td.get("poc")
    if poc and price:
        vah     = td.get("vah", 0)
        val     = td.get("val", 0)
        poc_pct = (price - poc) / price * 100
        if val <= price <= vah:
            vp_note = "inside Value Area"
        elif price > vah:
            vp_note = "above Value Area (breakout zone)"
        else:
            vp_note = "below Value Area (bearish)"
        print(f"    Volume Profile:   POC ${poc:.2f} ({poc_pct:+.1f}% vs price)  "
              f"|  VA ${val:.2f}–${vah:.2f}")
        print(f"                      → {vp_note}")

    # Support & Resistance zones
    ns  = sr.get("nearest_support")
    nr  = sr.get("nearest_resistance")

    if ns and price:
        pct   = (price - ns["price"]) / price * 100
        stars = "★" * min(ns["touches"], 3)
        print(f"    Support:          ${ns['low']:.2f}–${ns['high']:.2f}  "
              f"({pct:.1f}% below)  {stars}  ({ns['touches']} touches)")

    if nr and price:
        pct   = (nr["price"] - price) / price * 100
        stars = "★" * min(nr["touches"], 3)
        print(f"    Resistance:       ${nr['low']:.2f}–${nr['high']:.2f}  "
              f"({pct:.1f}% above)  {stars}  ({nr['touches']} touches)")

    # Entry setup with risk/reward
    if ns and price:
        stop = round(ns["low"] * 0.985, 2)
        risk_pct = (price - stop) / price * 100
        if nr and price:
            target    = round(nr["price"], 2)
            upside    = (target - price) / price * 100
            risk_dist = price - stop
            rr        = round((target - price) / risk_dist, 1) if risk_dist > 0 else 0
            print(f"\n    ► ENTRY ZONE:    ${ns['low']:.2f}–${ns['high']:.2f}")
            print(f"      Stop loss:     ${stop:.2f}  (−{risk_pct:.1f}%)")
            print(f"      Target:        ${target:.2f}  (+{upside:.1f}%)  |  Risk/Reward: 1:{rr}")
        else:
            print(f"\n    ► ENTRY ZONE:    ${ns['low']:.2f}–${ns['high']:.2f}")
            print(f"      Stop loss:     ${stop:.2f}  (−{risk_pct:.1f}%)")

    # ── Fundamental (Quarterly) ───────────────────────────────────────────────
    print(f"\n  FUNDAMENTAL  (quarterly data)")

    beats = fd.get("eps_beats_4q")
    if beats is not None:
        bar  = "●" * beats + "○" * (4 - beats)
        note = "consistent beats" if beats >= 3 else "mixed results" if beats >= 2 else "mostly missed"
        print(f"    EPS Beats:        {bar}  ({beats}/4 quarters)  →  {note}")

    rev_g = fd.get("rev_growth_yoy")
    if rev_g is not None:
        if rev_g > 25:    note = "high-growth"
        elif rev_g > 10:  note = "solid expansion"
        elif rev_g > 0:   note = "moderate growth"
        else:             note = "declining revenue"
        print(f"    Revenue (YoY):    {rev_g:+.1f}%  →  {note}")

    rev_trend = fd.get("rev_qoq_trend")
    if rev_trend:
        print(f"    Rev. QoQ trend:   {rev_trend}")

    eg = fd.get("earnings_growth")
    if eg is not None:
        if eg > 30:    note = "accelerating earnings"
        elif eg > 10:  note = "growing earnings"
        elif eg >= 0:  note = "flat earnings"
        else:          note = "earnings declining"
        print(f"    Earnings (YoY):   {eg:+.1f}%  →  {note}")

    gm = fd.get("gross_margin")
    if gm is not None:
        if gm > 60:   note = "premium margins"
        elif gm > 40: note = "solid margins"
        elif gm > 25: note = "average margins"
        else:         note = "thin margins"
        print(f"    Gross Margin:     {gm:.1f}%  →  {note}")

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Weekly Sector Stock Scanner — Technical + Quarterly Fundamental",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sector names (case-insensitive):
  technology   healthcare    energy        financials    industrials
  consumer_discretionary     consumer_staples            utilities
  real_estate  materials     communication

ETF tickers also accepted directly: XLK XLV XLE XLF XLI XLY XLP XLU XLRE XLB XLC
        """,
    )
    parser.add_argument("sector", nargs="?",
                        help="Sector name or ETF ticker (e.g. technology, XLK)")
    parser.add_argument("--top", type=int, default=8, metavar="N",
                        help="Stocks to show in detail (default: 8)")
    args = parser.parse_args()

    sector_input = args.sector or input("Enter sector name or ETF ticker: ").strip()
    etf_ticker, sector_name = resolve_sector(sector_input)

    print(f"\nFetching {sector_name} ({etf_ticker}) holdings …")
    tickers = get_sector_holdings(etf_ticker, max_stocks=25)

    if not tickers:
        print(f"\n  Could not fetch holdings for '{etf_ticker}'.")
        print("  Try: XLK  XLV  XLE  XLF  XLI  XLY  XLP  XLU  XLRE  XLB  XLC")
        sys.exit(1)

    print(f"Found {len(tickers)} holdings. Analyzing weekly charts …\n")

    candidates   = []
    n_above_ma   = 0
    n_bull_rsi   = 0
    n_valid_brd  = 0

    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i:>2}/{len(tickers)}] {ticker:<8}", end="\r")
        try:
            t     = yf.Ticker(ticker)
            info  = t.info
            name  = info.get("shortName", ticker)
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            if not price:
                continue

            weekly_hist = get_weekly_history(ticker)
            if weekly_hist is None:
                continue

            # Breadth counters
            close_w  = weekly_hist["Close"].squeeze()
            ma50_w   = close_w.rolling(50).mean()
            rsi_w    = _rsi(close_w, period=14)
            ma50_val = float(ma50_w.iloc[-1])

            if not pd.isna(ma50_val):
                n_valid_brd += 1
                if price > ma50_val:   n_above_ma += 1
                if rsi_w > 50:         n_bull_rsi += 1

            # Per-stock analysis
            sr_data             = support_resistance_weekly(weekly_hist, price)
            vp_data             = volume_profile(weekly_hist)
            t_score, setup, td  = weekly_technical_score(weekly_hist, price, sr_data, vp_data)
            f_score, fd         = quarterly_fundamental_score(t, info)
            rating              = round(max(1.0, min(10.0, t_score + f_score)), 1)

            candidates.append({
                "ticker":       ticker,
                "name":         name,
                "price":        price,
                "rating":       rating,
                "tech_score":   round(t_score, 1),
                "fund_score":   round(f_score, 1),
                "setup_label":  setup,
                "tech_details": td,
                "fund_details": fd,
                "sr_data":      sr_data,
                "vp_data":      vp_data,
            })
        except Exception:
            continue

    print(" " * 50)

    if not candidates:
        print("No valid stocks found.")
        sys.exit(1)

    pct_above_ma = n_above_ma / max(n_valid_brd, 1)
    pct_bull_rsi = n_bull_rsi / max(n_valid_brd, 1)
    sentiment    = sector_sentiment(etf_ticker, pct_above_ma, pct_bull_rsi)

    candidates.sort(key=lambda x: x["rating"], reverse=True)

    print_sentiment_header(etf_ticker, sector_name, sentiment)
    print_summary_table(candidates, len(tickers))

    n_detail = min(args.top, len(candidates))
    print(f"\n\n{'═' * W}")
    print(f"  DETAILED ANALYSIS  —  Top {n_detail} stocks")
    print(f"  Weekly chart · Support/Resistance · Volume Profile · Quarterly data")
    print(f"{'═' * W}")

    for rank, c in enumerate(candidates[:n_detail], 1):
        print_stock_detail(rank, c)

    print(f"{'═' * W}")
    print("  Not financial advice. Always do your own research.")
    print(f"{'═' * W}\n")


if __name__ == "__main__":
    main()
