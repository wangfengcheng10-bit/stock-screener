#!/usr/bin/env python3
"""
app.py — Sector Scanner Web Application
Run with: python3 app.py
Open:     http://localhost:5000
"""

from flask import Flask, render_template, jsonify, request
import yfinance as yf
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sector_screener import (
    resolve_sector,
    get_sector_holdings,
    get_weekly_history,
    _rsi,
    _adx,
    support_resistance_weekly,
    volume_profile,
    weekly_technical_score,
    quarterly_fundamental_score,
    sector_sentiment,
)

app = Flask(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rsi_series(series, period=14):
    """Full RSI series for chart plotting (not just last value)."""
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def get_daily_history(ticker, period="2y"):
    hist = yf.download(ticker, period=period, interval="1d",
                       progress=False, auto_adjust=True)
    if hist.empty or len(hist) < 30:
        return None
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    return hist


def hist_to_candles(hist):
    """Convert OHLCV DataFrame → list for Lightweight Charts."""
    out = []
    for ts, row in hist.iterrows():
        try:
            out.append({
                "time":   pd.Timestamp(ts).strftime("%Y-%m-%d"),
                "open":   round(float(row["Open"]),  2),
                "high":   round(float(row["High"]),  2),
                "low":    round(float(row["Low"]),   2),
                "close":  round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
            })
        except Exception:
            continue
    return out


def build_indicators(hist, price, weekly_hist=None):
    """
    Compute chart overlays: MA50, RSI series, volume bars, SR zones, volume profile.
    Support/resistance always uses the weekly hist if provided (matches the screener).
    """
    close = hist["Close"].squeeze()

    # MA50
    ma50      = close.rolling(50).mean()
    ma50_data = [
        {"time": pd.Timestamp(ts).strftime("%Y-%m-%d"), "value": round(float(v), 2)}
        for ts, v in ma50.items() if not pd.isna(v)
    ]

    # RSI(14)
    rsi_s    = _rsi_series(close)
    rsi_data = [
        {"time": pd.Timestamp(ts).strftime("%Y-%m-%d"), "value": round(float(v), 2)}
        for ts, v in rsi_s.items() if not pd.isna(v)
    ]

    # Volume bars (coloured by direction)
    vol_data = []
    for ts, row in hist.iterrows():
        try:
            up = float(row["Close"]) >= float(row["Open"])
            vol_data.append({
                "time":  pd.Timestamp(ts).strftime("%Y-%m-%d"),
                "value": int(row["Volume"]),
                "color": "#26a69a55" if up else "#ef535055",
            })
        except Exception:
            continue

    # SR and VP on weekly for daily view, or on hist directly for weekly view
    analysis = weekly_hist if weekly_hist is not None else hist
    sr  = support_resistance_weekly(analysis, price)
    vp  = volume_profile(analysis)

    # Fibonacci retracement from 52-week high → low
    lookback = min(len(close), 252)
    hi52 = float(close.iloc[-lookback:].max())
    lo52 = float(close.iloc[-lookback:].min())
    span = hi52 - lo52
    fibs = {
        "0":    round(hi52, 2),
        "23.6": round(hi52 - 0.236 * span, 2),
        "38.2": round(hi52 - 0.382 * span, 2),
        "50.0": round(hi52 - 0.500 * span, 2),
        "61.8": round(hi52 - 0.618 * span, 2),
        "78.6": round(hi52 - 0.786 * span, 2),
        "100":  round(lo52, 2),
    } if span > 0 else {}

    def _zone(z):
        if z is None:
            return None
        return {k: z.get(k) for k in ("price", "low", "high", "touches", "strength")}

    return {
        "ma50":   ma50_data,
        "rsi":    rsi_data,
        "volume": vol_data,
        "fibs":   fibs,
        "sr": {
            "supports":           [_zone(z) for z in sr.get("supports",    [])[:4]],
            "resistances":        [_zone(z) for z in sr.get("resistances", [])[:3]],
            "nearest_support":    _zone(sr.get("nearest_support")),
            "nearest_resistance": _zone(sr.get("nearest_resistance")),
        },
        "vp": vp or {},
    }


# ── Per-ticker scan (called concurrently) ─────────────────────────────────────

def _analyse_ticker(ticker):
    t     = yf.Ticker(ticker)
    info  = t.info
    name  = info.get("shortName", ticker)
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if not price:
        return None

    prev_close = info.get("regularMarketPreviousClose") or info.get("previousClose")
    change_pct = round((price - prev_close) / prev_close * 100, 2) \
                 if prev_close and prev_close > 0 else None

    weekly_hist = get_weekly_history(ticker)
    if weekly_hist is None:
        return None

    close_w  = weekly_hist["Close"].squeeze()
    ma50_w   = close_w.rolling(50).mean()
    rsi_w    = _rsi(close_w, period=14)
    ma50_val = float(ma50_w.iloc[-1])

    sr_data             = support_resistance_weekly(weekly_hist, price)
    vp_data             = volume_profile(weekly_hist)
    t_score, setup, td  = weekly_technical_score(weekly_hist, price, sr_data, vp_data)
    f_score, fd         = quarterly_fundamental_score(t, info)
    rating              = round(max(1.0, min(10.0, t_score + f_score)), 1)

    def _z(z):
        if z is None:
            return None
        return {k: z.get(k) for k in ("price", "low", "high", "touches")}

    above_ma = (not pd.isna(ma50_val)) and (price > ma50_val)
    bull_rsi = rsi_w > 50

    return {
        "ticker":       ticker,
        "name":         name,
        "price":        price,
        "change_pct":   change_pct,
        "rating":       rating,
        "tech_score":   round(t_score, 1),
        "fund_score":   round(f_score, 1),
        "setup_label":  setup,
        "tech_details": td,
        "fund_details": fd,
        "sr": {
            "supports":           [_z(z) for z in sr_data.get("supports",    [])[:4]],
            "resistances":        [_z(z) for z in sr_data.get("resistances", [])[:3]],
            "nearest_support":    _z(sr_data.get("nearest_support")),
            "nearest_resistance": _z(sr_data.get("nearest_resistance")),
        },
        "vp": vp_data or {},
        "info": {
            "sector":      info.get("sector", ""),
            "industry":    info.get("industry", ""),
            "description": (info.get("longBusinessSummary") or "")[:400],
            "market_cap":  info.get("marketCap"),
            "pe":          info.get("trailingPE"),
            "forward_pe":  info.get("forwardPE"),
            "peg":         info.get("pegRatio"),
            "roe":         info.get("returnOnEquity"),
            "debt_equity": info.get("debtToEquity"),
            "analyst":     info.get("recommendationKey", ""),
        },
        "_breadth": {"above_ma": above_ma, "bull_rsi": bull_rsi},
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/scan", methods=["POST"])
def scan_sector():
    body      = request.get_json(force=True)
    etf_input = (body.get("etf") or "").strip()
    if not etf_input:
        return jsonify({"error": "No ETF ticker provided"}), 400

    etf_ticker, sector_name = resolve_sector(etf_input)
    tickers = get_sector_holdings(etf_ticker, max_stocks=25)
    if not tickers:
        return jsonify({"error": f"No holdings found for '{etf_ticker}'"}), 404

    candidates  = []
    n_above_ma  = n_bull_rsi = n_valid = 0

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_analyse_ticker, t): t for t in tickers}
        for fut in as_completed(futures):
            try:
                result = fut.result()
                if result is None:
                    continue
                b = result.pop("_breadth", {})
                n_valid += 1
                if b.get("above_ma"):  n_above_ma += 1
                if b.get("bull_rsi"):  n_bull_rsi += 1
                candidates.append(result)
            except Exception:
                continue

    sent = sector_sentiment(
        etf_ticker,
        n_above_ma / max(n_valid, 1),
        n_bull_rsi / max(n_valid, 1),
    )
    candidates.sort(key=lambda x: x["rating"], reverse=True)

    return jsonify({
        "sector": {"name": sector_name, "etf": etf_ticker, "sentiment": sent},
        "stocks": candidates,
    })


@app.route("/api/history/<ticker>")
def get_history(ticker):
    try:
        t     = yf.Ticker(ticker)
        info  = t.info
        price = info.get("currentPrice") or info.get("regularMarketPrice") or 0.0

        daily_hist  = get_daily_history(ticker, period="2y")
        weekly_hist = get_weekly_history(ticker, period="3y")

        result = {}
        if daily_hist is not None:
            result["daily"]            = hist_to_candles(daily_hist)
            result["daily_indicators"] = build_indicators(
                daily_hist, price, weekly_hist=weekly_hist
            )
        if weekly_hist is not None:
            result["weekly"]            = hist_to_candles(weekly_hist)
            result["weekly_indicators"] = build_indicators(weekly_hist, price)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Sector Scanner Web App")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()
    print(f"\n  Sector Scanner  →  http://localhost:{args.port}\n")
    app.run(debug=False, host=args.host, port=args.port)
