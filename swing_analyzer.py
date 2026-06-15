#!/usr/bin/env python3
"""
Swing Trading Analyzer — Multi-timeframe technical analysis
Data source: Yahoo Finance (free)
Usage:
  python3 swing_analyzer.py AAPL          # single stock deep analysis
  python3 swing_analyzer.py QQQ --etf     # screen ETF holdings
"""

import yfinance as yf
import numpy as np
import pandas as pd
import sys
import argparse
from datetime import datetime


# ── Data Fetching ─────────────────────────────────────────────────────────────

def get_history(ticker, period="2y", interval="1d"):
    hist = yf.download(ticker, period=period, interval=interval,
                       progress=False, auto_adjust=True)
    if hist.empty or len(hist) < 20:
        return None
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    return hist


def get_etf_holdings(etf_ticker):
    etf = yf.Ticker(etf_ticker)
    try:
        df = etf.funds_data.top_holdings
        if df is not None and not df.empty:
            tickers = [t for t in df.index if isinstance(t, str) and len(t) <= 6]
            if tickers:
                return tickers[:30]
    except Exception:
        pass
    try:
        raw = etf.info.get("holdings", [])
        tickers = [h.get("symbol") for h in raw if h.get("symbol")]
        if tickers:
            return tickers[:30]
    except Exception:
        pass
    return []


# ── Indicators ────────────────────────────────────────────────────────────────

def _ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def _rsi_series(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _rsi(series, period=14):
    val = _rsi_series(series, period).iloc[-1]
    return round(float(val), 1) if not np.isnan(val) else None


def _stoch_rsi(series, rsi_period=14, stoch_period=14, smooth_k=3, smooth_d=3):
    rsi = _rsi_series(series, rsi_period)
    lo = rsi.rolling(stoch_period).min()
    hi = rsi.rolling(stoch_period).max()
    raw = (rsi - lo) / (hi - lo).replace(0, np.nan) * 100
    k = raw.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    kv, dv = float(k.iloc[-1]), float(d.iloc[-1])
    if np.isnan(kv) or np.isnan(dv):
        return None, None
    return round(kv, 1), round(dv, 1)


def _macd(series, fast=12, slow=26, signal=9):
    ml = _ema(series, fast) - _ema(series, slow)
    sl = ml.ewm(span=signal, adjust=False).mean()
    hist = ml - sl
    h0, h1 = float(hist.iloc[-1]), float(hist.iloc[-2])
    m0, s0 = float(ml.iloc[-1]), float(sl.iloc[-1])
    return {
        "macd":          round(m0, 4),
        "signal":        round(s0, 4),
        "histogram":     round(h0, 4),
        "bullish_cross": m0 > s0,
        "hist_rising":   h0 > h1,
        "above_zero":    m0 > 0,
    }


def _bollinger(series, period=20, std_dev=2):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    up  = mid + std_dev * std
    lo  = mid - std_dev * std
    p, u, m, l = float(series.iloc[-1]), float(up.iloc[-1]), float(mid.iloc[-1]), float(lo.iloc[-1])
    b_pct = (p - l) / (u - l) if (u - l) != 0 else 0.5
    bw    = (u - l) / m if m != 0 else 0
    avg_bw = float(((up - lo) / mid).rolling(50).mean().iloc[-1])
    return {
        "upper":   round(u, 2),
        "middle":  round(m, 2),
        "lower":   round(l, 2),
        "b_pct":   round(b_pct, 2),
        "squeeze": bw < avg_bw * 0.75 if not np.isnan(avg_bw) else False,
    }


def _obv(close, volume):
    direction  = np.sign(close.diff()).fillna(0)
    obv_series = (direction * volume).cumsum()
    obv_ema    = obv_series.ewm(span=20, adjust=False).mean()
    rising     = float(obv_series.iloc[-1]) > float(obv_ema.iloc[-1])
    slope_raw  = float(obv_series.iloc[-1]) - float(obv_series.iloc[-20])
    base       = abs(float(obv_series.iloc[-20])) + 1
    return {
        "trend":  "Rising" if rising else "Falling",
        "rising": rising,
        "slope":  round(slope_raw / base * 100, 1),
    }


def _adx(hist, period=14):
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
    adx_val = float(dx.rolling(period).mean().iloc[-1])
    return round(adx_val, 1), round(float(pdi.iloc[-1]), 1), round(float(mdi.iloc[-1]), 1)


def _vol_analysis(hist):
    vol    = hist["Volume"].squeeze()
    avg20  = float(vol.rolling(20).mean().iloc[-1])
    last   = float(vol.iloc[-1])
    ratio  = last / avg20 if avg20 > 0 else 1.0
    return {"last": int(last), "avg_20": int(avg20), "ratio": round(ratio, 2)}


# ── Market Structure ──────────────────────────────────────────────────────────

def detect_structure(close, window=10):
    prices = close.values
    highs, lows = [], []
    for i in range(window, len(prices) - window):
        seg = prices[i - window: i + window + 1]
        if prices[i] == seg.max():
            highs.append(float(prices[i]))
        if prices[i] == seg.min():
            lows.append(float(prices[i]))

    if len(highs) < 2 or len(lows) < 2:
        return {"label": "Insufficient data", "bias": "neutral"}

    rh, rl = highs[-3:], lows[-3:]
    hh = len(rh) >= 2 and all(rh[i] > rh[i-1] for i in range(1, len(rh)))
    hl = len(rl) >= 2 and all(rl[i] > rl[i-1] for i in range(1, len(rl)))
    lh = len(rh) >= 2 and all(rh[i] < rh[i-1] for i in range(1, len(rh)))
    ll = len(rl) >= 2 and all(rl[i] < rl[i-1] for i in range(1, len(rl)))

    if hh and hl:
        return {"label": "Higher Highs + Higher Lows", "bias": "bullish"}
    if lh and ll:
        return {"label": "Lower Highs + Lower Lows", "bias": "bearish"}
    if hh and ll:
        return {"label": "Expanding Range", "bias": "neutral"}
    if lh and hl:
        return {"label": "Contracting Range / Squeeze", "bias": "neutral"}
    return {"label": "Mixed / Ranging", "bias": "neutral"}


# ── Zone Detection ────────────────────────────────────────────────────────────

def _swing_pivots(close, volume, window=8, kind="low"):
    prices = close.values
    vols   = volume.values
    pivots = []
    for i in range(window, len(prices) - window):
        seg = prices[i - window: i + window + 1]
        if kind == "low"  and prices[i] == seg.min():
            pivots.append({"price": float(prices[i]), "volume": float(vols[i]), "idx": i})
        if kind == "high" and prices[i] == seg.max():
            pivots.append({"price": float(prices[i]), "volume": float(vols[i]), "idx": i})
    return pivots


def _cluster_zones(pivots, n_bars, tolerance=0.025):
    if not pivots:
        return []
    sorted_p = sorted(pivots, key=lambda x: x["price"])
    used, zones = set(), []
    for i in range(len(sorted_p)):
        if i in used:
            continue
        cluster = [sorted_p[i]]
        used.add(i)
        for j in range(i + 1, len(sorted_p)):
            if j in used:
                continue
            if abs(sorted_p[j]["price"] - sorted_p[i]["price"]) / sorted_p[i]["price"] < tolerance:
                cluster.append(sorted_p[j])
                used.add(j)
        prices  = [c["price"] for c in cluster]
        vols    = [c["volume"] for c in cluster]
        idxs    = [c["idx"]   for c in cluster]
        lo_     = round(min(prices) * 0.998, 2)
        hi_     = round(max(prices) * 1.002, 2)
        recency = max(idxs) / max(n_bars, 1)
        strength = len(cluster) * (sum(vols) / 1e6) * (0.3 + 0.7 * recency)
        zones.append({
            "low":     lo_,
            "high":    hi_,
            "mid":     round((lo_ + hi_) / 2, 2),
            "touches": len(cluster),
            "strength": round(strength, 1),
        })
    return sorted(zones, key=lambda z: z["strength"], reverse=True)


def detect_demand_zones(hist, window=8):
    close  = hist["Close"].squeeze()
    volume = hist["Volume"].squeeze()
    pivots = _swing_pivots(close, volume, window=window, kind="low")
    return _cluster_zones(pivots, len(close))


def detect_supply_zones(hist, window=8):
    close  = hist["Close"].squeeze()
    volume = hist["Volume"].squeeze()
    pivots = _swing_pivots(close, volume, window=window, kind="high")
    return _cluster_zones(pivots, len(close))


# ── Fibonacci ─────────────────────────────────────────────────────────────────

def find_major_swing(hist, lookback=252):
    high  = hist["High"].squeeze().iloc[-lookback:]
    low   = hist["Low"].squeeze().iloc[-lookback:]
    return round(float(high.max()), 2), round(float(low.min()), 2)


def fibonacci_levels(swing_high, swing_low):
    diff = swing_high - swing_low
    retrace = {
        "23.6%": round(swing_high - 0.236 * diff, 2),
        "38.2%": round(swing_high - 0.382 * diff, 2),
        "50.0%": round(swing_high - 0.500 * diff, 2),
        "61.8%": round(swing_high - 0.618 * diff, 2),
        "78.6%": round(swing_high - 0.786 * diff, 2),
    }
    extension = {
        "100%  (prev. high)": round(swing_high, 2),
        "127.2%":             round(swing_high + 0.272 * diff, 2),
        "161.8%":             round(swing_high + 0.618 * diff, 2),
        "200%":               round(swing_high + 1.000 * diff, 2),
    }
    return retrace, extension


# ── Full Timeframe Analysis ───────────────────────────────────────────────────

def analyze_timeframe(hist):
    close  = hist["Close"].squeeze()
    volume = hist["Volume"].squeeze()

    adx, pdi, mdi = _adx(hist)
    bullish = pdi > mdi
    if adx > 25:    strength = "Strong"
    elif adx > 18:  strength = "Developing"
    else:           strength = "Weak"
    trend_label = f"{strength} {'Uptrend' if bullish else 'Downtrend'}" if adx > 18 else "Ranging / Consolidating"

    price = float(close.iloc[-1])
    ema20  = round(float(_ema(close, 20).iloc[-1]),  2)
    ema50  = round(float(_ema(close, 50).iloc[-1]),  2) if len(close) >= 50  else None
    ema200 = round(float(_ema(close, 200).iloc[-1]), 2) if len(close) >= 200 else None

    try:
        bb = _bollinger(close)
    except Exception:
        bb = {}

    try:
        stk, std = _stoch_rsi(close)
    except Exception:
        stk, std = None, None

    return {
        "price":     round(price, 2),
        "rsi":       _rsi(close),
        "stoch_k":   stk,
        "stoch_d":   std,
        "macd":      _macd(close),
        "bb":        bb,
        "obv":       _obv(close, volume),
        "vol":       _vol_analysis(hist),
        "adx":       adx,
        "pdi":       pdi,
        "mdi":       mdi,
        "trend":     trend_label,
        "bullish":   bullish,
        "ema20":     ema20,
        "ema50":     ema50,
        "ema200":    ema200,
        "structure": detect_structure(close, window=max(5, min(10, len(close) // 20))),
    }


# ── Trading Plan Builder ──────────────────────────────────────────────────────

def build_plan(price, daily, weekly, demand_zones, supply_zones, retrace, extension):
    confluence = 0.0
    factors, risks = [], []

    # 1. Weekly trend
    if weekly.get("bullish"):
        confluence += 1
        factors.append(f"Weekly {weekly.get('trend','uptrend')} confirmed")
    else:
        risks.append(f"Weekly trend is {weekly.get('trend','unknown')} — headwind for longs")

    # 2. Daily structure
    struct = daily.get("structure", {})
    if struct.get("bias") == "bullish":
        confluence += 1
        factors.append(f"Daily structure bullish ({struct.get('label','')})")
    elif struct.get("bias") == "bearish":
        risks.append(f"Daily structure bearish ({struct.get('label','')})")

    # 3. Demand zone proximity
    at_zone   = [z for z in demand_zones if price <= z["high"] * 1.03]
    near_zone = [z for z in demand_zones if price <= z["high"] * 1.12]
    best_demand = None
    if at_zone:
        best_demand = at_zone[0]
        confluence += 1.5
        pct = (price - best_demand["high"]) / price * 100
        factors.append(f"Price AT demand zone ({pct:+.1f}% from zone)")
    elif near_zone:
        best_demand = near_zone[0]
        pct = (price - best_demand["high"]) / price * 100
        factors.append(f"Demand zone {abs(pct):.1f}% below — wait for pullback")
    else:
        risks.append("Price extended above all demand zones")

    # 4. RSI + Stochastic
    rsi = daily.get("rsi") or 50
    stk = daily.get("stoch_k") or 50
    if rsi < 35 or stk < 20:
        confluence += 1
        factors.append(f"Oversold: RSI {rsi}, StochRSI {stk}")
    elif rsi < 55:
        confluence += 0.5
        factors.append(f"RSI neutral ({rsi}) — room to run")
    elif rsi > 70 or stk > 80:
        risks.append(f"Overbought: RSI {rsi}, StochRSI {stk} — risky to buy")

    # 5. MACD
    macd = daily.get("macd", {})
    if macd.get("bullish_cross") and not macd.get("above_zero"):
        confluence += 1
        factors.append("MACD bullish cross below zero (high-quality signal)")
    elif macd.get("bullish_cross"):
        confluence += 0.5
        factors.append("MACD bullish cross (above zero)")
    elif macd.get("hist_rising") and (macd.get("histogram") or 0) < 0:
        confluence += 0.5
        factors.append("MACD histogram narrowing — selling momentum fading")
    elif not macd.get("bullish_cross"):
        risks.append("MACD bearish — no cross yet")

    # 6. OBV
    obv = daily.get("obv", {})
    if obv.get("rising"):
        confluence += 1
        factors.append("OBV rising — smart money accumulating")
    else:
        risks.append("OBV falling — distribution in play")

    total = round(confluence)

    # Signal text
    if at_zone and total >= 4:
        signal = "STRONG BUY — High-confluence setup at demand zone"
    elif at_zone and total >= 2:
        signal = "BUY — At demand zone"
    elif near_zone and total >= 3:
        signal = "WATCH — Wait for pullback to demand zone"
    elif near_zone:
        signal = "WAIT — Set alert at demand zone"
    elif not best_demand:
        signal = "AVOID — No demand zone / price too extended"
    else:
        signal = "EXTENDED — High risk, wait for deeper pullback"

    # Entry / stop
    if best_demand:
        el, eh, em = best_demand["low"], best_demand["high"], best_demand["mid"]
        stop = round(el * 0.984, 2)
        rps  = em - stop
    else:
        el = eh = em = None
        stop = round(price * 0.95, 2)
        rps  = price * 0.05

    risk_pct = round((em - stop) / em * 100, 1) if em else 0

    # Targets: supply zones first, then fib extensions
    targets = []
    ref = eh or price
    for i, sz in enumerate([z for z in supply_zones if z["low"] > ref * 1.01][:2]):
        gain = sz["mid"] - ref
        rr   = gain / rps if rps > 0 else 0
        targets.append({"label": f"Supply Zone {i+1}", "price": sz["mid"],
                        "pct": round(gain / ref * 100, 1), "rr": round(rr, 1)})

    for name, lvl in sorted(extension.items(), key=lambda x: x[1]):
        if lvl > ref * 1.01 and len(targets) < 3:
            gain = lvl - (em or price)
            rr   = gain / rps if rps > 0 else 0
            targets.append({"label": f"Fib {name}", "price": lvl,
                            "pct": round(gain / (em or price) * 100, 1), "rr": round(rr, 1)})

    targets = sorted(targets, key=lambda x: x["price"])[:3]

    return {
        "signal":     signal,
        "confluence": total,
        "factors":    factors,
        "risks":      risks,
        "entry_low":  el,
        "entry_high": eh,
        "entry_mid":  em,
        "stop":       stop,
        "risk_pct":   risk_pct,
        "rps":        rps,
        "targets":    targets,
    }


# ── Report Printer ────────────────────────────────────────────────────────────

W = 70

def print_report(ticker, info, price, daily, weekly, demand_z, supply_z,
                 swing_high, swing_low, retrace, extension, plan):
    name     = (info or {}).get("shortName", ticker)
    sector   = (info or {}).get("sector", "")
    mktcap   = (info or {}).get("marketCap", 0)
    cap_str  = f"${mktcap/1e9:.1f}B" if mktcap > 1e9 else (f"${mktcap/1e6:.0f}M" if mktcap else "N/A")
    date_str = datetime.now().strftime("%Y-%m-%d")

    SEP = "─" * 65

    print(f"\n{'═'*W}")
    print(f"  SWING TRADING ANALYSIS  ·  {ticker}  ·  {date_str}")
    print(f"  {name}" + (f"  ·  {sector}" if sector else ""))
    print(f"  Current Price: ${price:.2f}  |  Market Cap: {cap_str}")
    print(f"{'═'*W}")

    # ── Market Structure ──────────────────────────────────────────────────────
    print(f"\n  {SEP}")
    print(f"  MARKET STRUCTURE")
    print(f"  {SEP}")
    ws = weekly.get("structure", {}) if weekly else {}
    ds = daily.get("structure", {})
    wa = weekly if weekly else {}
    print(f"  Weekly:  {wa.get('trend','N/A'):<30}  (ADX {wa.get('adx','?')}  +DI {wa.get('pdi','?')} / -DI {wa.get('mdi','?')})")
    print(f"           {ws.get('label','N/A')}")
    print(f"  Daily:   {daily.get('trend','N/A'):<30}  (ADX {daily.get('adx','?')}  +DI {daily.get('pdi','?')} / -DI {daily.get('mdi','?')})")
    print(f"           {ds.get('label','N/A')}")

    # ── Moving Averages ───────────────────────────────────────────────────────
    print(f"\n  {SEP}")
    print(f"  MOVING AVERAGES  (Daily EMA)")
    print(f"  {SEP}")
    for period, key in [(20, "ema20"), (50, "ema50"), (200, "ema200")]:
        val = daily.get(key)
        if val is None:
            continue
        pos  = "above ✓" if price > val else "below ✗"
        diff = (price - val) / val * 100
        line = f"  EMA {period:<3}:  ${val:.2f}  →  Price {pos}  ({diff:+.1f}%)"
        if key == "ema200" and daily.get("ema50"):
            gc = "Golden Cross ✓" if daily["ema50"] > val else "Death Cross ✗"
            line += f"  [{gc}]"
        print(line)

    # Weekly EMAs
    if weekly:
        print(f"  ─  ─  ─  Weekly EMAs  ─  ─  ─")
        for period, key in [(20, "ema20"), (50, "ema50")]:
            val = weekly.get(key)
            if val is None:
                continue
            pos  = "above ✓" if price > val else "below ✗"
            diff = (price - val) / val * 100
            print(f"  EMA {period:<3}w: ${val:.2f}  →  Price {pos}  ({diff:+.1f}%)")

    # ── Oscillators ───────────────────────────────────────────────────────────
    print(f"\n  {SEP}")
    print(f"  OSCILLATORS  (Daily)")
    print(f"  {SEP}")

    rsi = daily.get("rsi")
    if rsi is not None:
        if rsi < 30:    desc = "Oversold  ← potential reversal"
        elif rsi < 40:  desc = "Mildly oversold"
        elif rsi < 60:  desc = "Neutral"
        elif rsi < 70:  desc = "Mildly overbought"
        else:           desc = "Overbought — caution"
        print(f"  RSI (14):    {rsi:5.1f}  →  {desc}")

    stk, std_ = daily.get("stoch_k"), daily.get("stoch_d")
    if stk is not None and std_ is not None:
        if stk < 20:    desc = "Oversold"
        elif stk < 50:  desc = "Neutral-low"
        elif stk < 80:  desc = "Neutral-high"
        else:           desc = "Overbought"
        cross = "K > D ↑ (bullish)" if stk > std_ else "K < D ↓ (bearish)"
        print(f"  StochRSI:    K={stk}  D={std_}  →  {desc}  |  {cross}")

    macd = daily.get("macd", {})
    if macd:
        cross = "Bullish ✓" if macd.get("bullish_cross") else "Bearish ✗"
        side  = "above zero" if macd.get("above_zero") else "below zero"
        mom   = "gaining ↑" if macd.get("hist_rising") else "losing ↓"
        h     = macd.get("histogram", 0)
        print(f"  MACD:        {cross} ({side})  |  Momentum {mom}  (hist {h:+.4f})")

    bb = daily.get("bb", {})
    if bb:
        b = bb.get("b_pct", 0.5)
        if b < 0.2:    bd = "Near lower band — oversold / high vol"
        elif b < 0.4:  bd = "Lower half of band"
        elif b < 0.6:  bd = "Mid-band"
        elif b < 0.8:  bd = "Upper half of band"
        else:          bd = "Near upper band — overbought / extended"
        sq = "  ⚡ SQUEEZE — breakout likely imminent" if bb.get("squeeze") else ""
        print(f"  Bollinger:   %B={b:.2f}  →  {bd}{sq}")
        print(f"               ${bb['lower']:.2f} — ${bb['middle']:.2f} — ${bb['upper']:.2f}")

    # ── Volume ────────────────────────────────────────────────────────────────
    print(f"\n  {SEP}")
    print(f"  VOLUME ANALYSIS")
    print(f"  {SEP}")
    obv = daily.get("obv", {})
    vol = daily.get("vol", {})
    print(f"  OBV trend:   {obv.get('trend','N/A')}  (slope {obv.get('slope',0):+.1f}%)  →  {'Accumulation ✓' if obv.get('rising') else 'Distribution ✗'}")
    ratio = vol.get("ratio", 1)
    if ratio > 1.5:    vd = "High volume — strong conviction"
    elif ratio > 1.1:  vd = "Above average"
    elif ratio > 0.7:  vd = "Average"
    else:              vd = "Low volume — weak conviction"
    avg = vol.get("avg_20", 0)
    print(f"  Volume:      {ratio:.1f}x 20-day avg  →  {vd}")
    if avg:
        print(f"               20d avg: {avg:,}")

    # ── Fibonacci ─────────────────────────────────────────────────────────────
    print(f"\n  {SEP}")
    print(f"  FIBONACCI LEVELS  (from ${swing_low:.2f} low → ${swing_high:.2f} high, 52-week)")
    print(f"  {SEP}")
    print(f"  Retracements (support / entry zones):")
    for lvl, val in retrace.items():
        pct = (price - val) / price * 100
        mark = "  ◄ AT PRICE" if abs(pct) < 2 else ("  ◄ nearby" if abs(pct) < 5 else "")
        print(f"    {lvl}: ${val:.2f}  ({pct:+.1f}%){mark}")
    print(f"  Extensions (profit targets):")
    for lvl, val in sorted(extension.items(), key=lambda x: x[1]):
        pct = (val - price) / price * 100
        mark = "  ◄ AT PRICE" if abs(pct) < 2 else ""
        print(f"    {lvl}: ${val:.2f}  ({pct:+.1f}%){mark}")

    # ── Demand Zones ──────────────────────────────────────────────────────────
    print(f"\n  {SEP}")
    print(f"  DEMAND ZONES  (Support — Buy Areas)")
    print(f"  {SEP}")
    d_below = [z for z in demand_z if z["high"] < price * 1.05][:4]
    if d_below:
        for i, z in enumerate(d_below):
            pct   = (price - z["high"]) / price * 100
            stars = "★★★" if z["touches"] >= 3 else ("★★" if z["touches"] == 2 else "★ ")
            near  = "  ← AT ZONE" if pct < 3 else ("  ← nearby" if pct < 8 else "")
            print(f"  {i+1}. {stars}  ${z['low']:.2f} – ${z['high']:.2f}  "
                  f"({z['touches']} touches)  [{pct:.1f}% away]{near}")
    else:
        print("  None found below current price.")

    # ── Supply Zones ──────────────────────────────────────────────────────────
    print(f"\n  {SEP}")
    print(f"  SUPPLY ZONES  (Resistance — Take Profit Areas)")
    print(f"  {SEP}")
    s_above = [z for z in supply_z if z["low"] > price * 0.97][:4]
    if s_above:
        for i, z in enumerate(s_above):
            pct   = (z["low"] - price) / price * 100
            stars = "★★★" if z["touches"] >= 3 else ("★★" if z["touches"] == 2 else "★ ")
            near  = "  ← AT ZONE" if pct < 3 else ("  ← nearby" if pct < 8 else "")
            print(f"  {i+1}. {stars}  ${z['low']:.2f} – ${z['high']:.2f}  "
                  f"({z['touches']} touches)  [{pct:+.1f}% above]{near}")
    else:
        print("  None found above current price.")

    # ── Trading Plan ──────────────────────────────────────────────────────────
    print(f"\n  {'═'*65}")
    print(f"  TRADING PLAN")
    print(f"  {'═'*65}")

    conf     = plan.get("confluence", 0)
    conf_bar = "■" * conf + "□" * max(0, 6 - conf)
    print(f"\n  Signal:      {plan.get('signal','')}")
    print(f"  Confluence:  [{conf_bar}]  {conf}/6")

    if plan.get("factors"):
        print(f"\n  Bullish factors:")
        for f in plan["factors"]:
            print(f"    ✓  {f}")
    if plan.get("risks"):
        print(f"  Risk factors:")
        for r in plan["risks"]:
            print(f"    ✗  {r}")

    el = plan.get("entry_low")
    eh = plan.get("entry_high")
    em = plan.get("entry_mid")
    stop = plan.get("stop")
    rps  = plan.get("rps", 0)

    if el and eh and stop:
        print(f"\n  ┌{'─'*60}┐")
        print(f"  │  ENTRY ZONE:  ${el:.2f} – ${eh:.2f}  (mid ${em:.2f})")
        print(f"  │  STOP LOSS:   ${stop:.2f}  (-{plan.get('risk_pct','?')}% from entry mid)")
        for i, t in enumerate(plan.get("targets", []), 1):
            print(f"  │  TARGET {i}:    ${t['price']:.2f}  ({t['pct']:+.1f}%)  "
                  f"R:R {t['rr']:.1f}:1  [{t['label']}]")
        if rps and rps > 0:
            account  = 10_000
            risk_amt = account * 0.01
            shares   = max(1, int(risk_amt / rps))
            pos_val  = round(shares * (em or price))
            print(f"  │")
            print(f"  │  POSITION SIZING  (1% risk on ${account:,} account):")
            print(f"  │  Risk/share: ${rps:.2f}  →  {shares} shares ≈ ${pos_val:,} position")
        print(f"  └{'─'*60}┘")
    else:
        print(f"\n  No actionable entry zone identified at current price.")

    print(f"\n{'═'*W}")
    print(f"  Not financial advice. Always do your own research.")
    print(f"{'═'*W}\n")


# ── Single Ticker Analysis ────────────────────────────────────────────────────

def analyze_ticker(ticker):
    print(f"\n  Fetching data for {ticker} …")

    hist_d = get_history(ticker, period="2y",  interval="1d")
    hist_w = get_history(ticker, period="5y",  interval="1wk")

    if hist_d is None:
        print(f"  ERROR: Not enough daily data for {ticker}.")
        return False

    try:
        info  = yf.Ticker(ticker).info
    except Exception:
        info  = {}

    price = float(hist_d["Close"].squeeze().iloc[-1])
    live  = (info or {}).get("currentPrice") or (info or {}).get("regularMarketPrice")
    if live:
        price = float(live)

    print(f"  Computing indicators …")
    daily   = analyze_timeframe(hist_d)
    weekly  = analyze_timeframe(hist_w) if hist_w is not None else {}

    demand_z = detect_demand_zones(hist_d, window=8)
    supply_z = detect_supply_zones(hist_d, window=8)

    s_high, s_low   = find_major_swing(hist_d, lookback=min(252, len(hist_d)))
    retrace, extend = fibonacci_levels(s_high, s_low)

    plan = build_plan(price, daily, weekly, demand_z, supply_z, retrace, extend)

    print_report(ticker, info, price, daily, weekly, demand_z, supply_z,
                 s_high, s_low, retrace, extend, plan)
    return True


# ── ETF Screen ────────────────────────────────────────────────────────────────

def screen_etf(etf_ticker):
    print(f"\n  Fetching {etf_ticker} holdings …")
    tickers = get_etf_holdings(etf_ticker)
    if not tickers:
        print(f"  Could not fetch holdings for '{etf_ticker}'.")
        return

    print(f"  Found {len(tickers)} holdings. Scanning for swing setups …\n")
    results = []
    for i, t in enumerate(tickers, 1):
        print(f"  [{i:>2}/{len(tickers)}] {t:<8}", end="\r", flush=True)
        try:
            h = get_history(t, period="2y", interval="1d")
            if h is None:
                continue
            hw = get_history(t, period="5y", interval="1wk")
            info  = yf.Ticker(t).info
            price = float(h["Close"].squeeze().iloc[-1])
            lp    = (info or {}).get("currentPrice") or (info or {}).get("regularMarketPrice")
            if lp:
                price = float(lp)
            daily  = analyze_timeframe(h)
            weekly = analyze_timeframe(hw) if hw else {}
            dz = detect_demand_zones(h)
            sz = detect_supply_zones(h)
            sh, sl  = find_major_swing(h, min(252, len(h)))
            r, ex   = fibonacci_levels(sh, sl)
            pl      = build_plan(price, daily, weekly, dz, sz, r, ex)
            name    = (info or {}).get("shortName", t)
            results.append({
                "ticker": t, "name": name, "price": price,
                "conf": pl["confluence"],
                "signal": pl["signal"],
                "trend": daily.get("trend", ""),
                "rsi": daily.get("rsi"),
            })
        except Exception:
            continue

    print(" " * 60)
    results.sort(key=lambda x: x["conf"], reverse=True)
    top = results[:10]

    print(f"\n{'═'*W}")
    print(f"  ETF: {etf_ticker}  —  TOP SWING SETUPS  ({datetime.now().strftime('%Y-%m-%d')})")
    print(f"  Scanned {len(results)} of {len(tickers)} holdings")
    print(f"{'═'*W}")
    print(f"\n  {'#':<3} {'Ticker':<7} {'Name':<22} {'Price':>8}  {'RSI':>5}  {'Conf':>5}  Signal")
    print(f"  {'─'*68}")
    for rank, c in enumerate(top, 1):
        rsi_s = f"{c['rsi']:.0f}" if c.get("rsi") else " N/A"
        sig   = c["signal"][:32]
        print(f"  {rank:<3} {c['ticker']:<7} {c['name'][:21]:<22} ${c['price']:>7.2f}  {rsi_s:>5}  {c['conf']:>4}/6  {sig}")

    print()
    choice = input("  Enter ticker for full analysis (or Enter to exit): ").strip().upper()
    if choice:
        analyze_ticker(choice)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Swing Trading Analyzer — Multi-timeframe technical analysis"
    )
    parser.add_argument("ticker", nargs="?", help="Stock or ETF ticker (e.g. AAPL, QQQ)")
    parser.add_argument("--etf", action="store_true", help="Screen ETF holdings for swing setups")
    args = parser.parse_args()

    ticker = (args.ticker or input("  Enter ticker: ").strip()).upper()

    if args.etf:
        screen_etf(ticker)
    else:
        analyze_ticker(ticker)


if __name__ == "__main__":
    main()
