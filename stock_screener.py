import yfinance as yf
import numpy as np
import pandas as pd
import sys
import argparse
from datetime import datetime


# ── ETF Holdings ──────────────────────────────────────────────────────────────

def get_etf_holdings(etf_ticker):
    etf = yf.Ticker(etf_ticker)
    try:
        holdings_df = etf.funds_data.top_holdings
        if holdings_df is not None and not holdings_df.empty:
            tickers = [t for t in holdings_df.index if isinstance(t, str) and len(t) <= 6]
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


# ── Price History ─────────────────────────────────────────────────────────────

def get_history(ticker):
    hist = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
    if hist.empty or len(hist) < 60:
        return None
    return hist


# ── Indicators ───────────────────────────────────────────────────────────────

def _rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])


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
    adx  = float(dx.rolling(period).mean().iloc[-1])
    return adx, float(pdi.iloc[-1]), float(mdi.iloc[-1])


# ── Technical Score ───────────────────────────────────────────────────────────

def technical_score(hist):
    """Returns (score 0-5, detail dict). Expects pre-downloaded hist."""
    try:
        close = hist["Close"].squeeze()
        score = 0.0
        d     = {}

        rsi = _rsi(close)
        d["rsi"] = round(rsi, 1)
        if 40 <= rsi <= 65:                          score += 1.5
        elif (30 <= rsi < 40) or (65 < rsi <= 75):  score += 0.75

        ma50  = float(close.rolling(50).mean().iloc[-1])
        price = float(close.iloc[-1])
        d["above_ma50"] = price > ma50
        if price > ma50:  score += 1.0

        if len(close) >= 200:
            ma200 = float(close.rolling(200).mean().iloc[-1])
            d["golden_cross"] = ma50 > ma200
            if ma50 > ma200:  score += 1.0

        if len(close) >= 63:
            mom = float(close.iloc[-1] / close.iloc[-63] - 1)
            d["momentum_3m"] = round(mom * 100, 1)
            if mom > 0.08:   score += 1.5
            elif mom > 0.02: score += 0.75

        return round(min(score, 5.0), 2), d
    except Exception:
        return 0.0, {}


# ── Fundamental Score ─────────────────────────────────────────────────────────

def fundamental_score(info):
    """Returns (score 0-5, detail dict)."""
    score = 0.0
    d     = {}

    rev = info.get("revenueGrowth") or 0
    d["revenue_growth"] = round(rev * 100, 1)
    if rev > 0.25:   score += 1.5
    elif rev > 0.10: score += 1.0
    elif rev > 0.05: score += 0.5

    gm = info.get("grossMargins") or 0
    d["gross_margin"] = round(gm * 100, 1)
    if gm > 0.60:   score += 1.5
    elif gm > 0.40: score += 1.0
    elif gm > 0.25: score += 0.5

    roe = info.get("returnOnEquity") or 0
    d["roe"] = round(roe * 100, 1)
    if roe > 0.30:   score += 1.0
    elif roe > 0.15: score += 0.5

    peg = info.get("pegRatio")
    d["peg"] = peg
    if peg and 0 < peg < 1.0:  score += 1.5
    elif peg and peg < 1.5:    score += 1.0
    elif peg and peg < 2.0:    score += 0.5

    eg = info.get("earningsGrowth") or 0
    d["earnings_growth"] = round(eg * 100, 1)
    if eg > 0.20:  score += 0.5

    return round(min(score, 5.0), 2), d


# ── Combined Rating ───────────────────────────────────────────────────────────

def combined_rating(t_score, f_score):
    return round(max(1.0, min(10.0, t_score + f_score)), 1)


# ── Trend Analysis ────────────────────────────────────────────────────────────

def analyze_trend(hist):
    """
    Returns a dict describing the current trend direction and strength.
    Uses ADX (trend strength), ±DI (direction), and MA slope.
    """
    try:
        close = hist["Close"].squeeze()
        adx, pdi, mdi = _adx(hist)

        bullish = pdi > mdi
        if adx > 25:
            strength = "Strong"
        elif adx > 18:
            strength = "Developing"
        else:
            strength = "Weak / Ranging"

        direction = "Uptrend" if bullish else "Downtrend"
        label = f"{strength} {direction}" if adx > 18 else "Ranging / Consolidating"

        # MA slope (rising or falling over last 20 bars)
        ma50     = close.rolling(50).mean()
        ma_slope = float(ma50.iloc[-1] - ma50.iloc[-20]) / float(ma50.iloc[-20]) * 100

        return {
            "label":     label,
            "adx":       round(adx, 1),
            "pdi":       round(pdi, 1),
            "mdi":       round(mdi, 1),
            "bullish":   bullish,
            "ma_slope":  round(ma_slope, 2),
        }
    except Exception:
        return {}


# ── Demand Zone Detection ─────────────────────────────────────────────────────

def _swing_lows(close, volume, window=10):
    """Find local price minima (swing lows) with their volume."""
    prices = close.values
    vols   = volume.values
    lows   = []
    for i in range(window, len(prices) - window):
        if prices[i] == prices[i - window : i + window + 1].min():
            lows.append({"price": float(prices[i]), "volume": float(vols[i]), "idx": i})
    return lows


def _cluster_zones(swing_lows, tolerance=0.025):
    """Group swing lows within tolerance% of each other into demand zones."""
    lows  = sorted(swing_lows, key=lambda x: x["price"])
    used  = set()
    zones = []
    n     = len(lows)

    for i in range(n):
        if i in used:
            continue
        cluster = [lows[i]]
        used.add(i)
        for j in range(i + 1, n):
            if j in used:
                continue
            if abs(lows[j]["price"] - lows[i]["price"]) / lows[i]["price"] < tolerance:
                cluster.append(lows[j])
                used.add(j)

        prices    = [c["price"] for c in cluster]
        vols      = [c["volume"] for c in cluster]
        idxs      = [c["idx"] for c in cluster]
        zone_low  = round(min(prices) * 0.998, 2)
        zone_high = round(max(prices) * 1.002, 2)
        recency   = max(idxs) / max(n, 1)           # 0–1, higher = more recent
        strength  = len(cluster) * (sum(vols) / 1e6) * (0.3 + 0.7 * recency)

        zones.append({
            "low":      zone_low,
            "high":     zone_high,
            "touches":  len(cluster),
            "strength": round(strength, 1),
        })

    return sorted(zones, key=lambda z: z["strength"], reverse=True)


def _fibonacci_levels(recent_high, recent_low):
    """Return 38.2%, 50%, 61.8% fib retracement from a swing high to swing low."""
    diff = recent_high - recent_low
    return {
        "23.6%": round(recent_high - 0.236 * diff, 2),
        "38.2%": round(recent_high - 0.382 * diff, 2),
        "50.0%": round(recent_high - 0.500 * diff, 2),
        "61.8%": round(recent_high - 0.618 * diff, 2),
    }


def demand_zone_analysis(hist, current_price):
    """
    Detect demand zones, compute Fibonacci levels, and suggest an entry.
    Returns a dict with zones, fibs, and entry recommendation.
    """
    try:
        close  = hist["Close"].squeeze()
        volume = hist["Volume"].squeeze()

        # ── Swing lows → clustered zones ──
        raw_lows = _swing_lows(close, volume, window=8)
        all_zones = _cluster_zones(raw_lows)

        # Only zones below current price (support, not resistance)
        zones_below = [z for z in all_zones if z["high"] < current_price * 1.01][:4]

        # ── Fibonacci from 52-week high → most recent swing low ──
        recent = close.iloc[-252:]
        hi52   = float(recent.max())
        lo52   = float(recent.min())
        fibs   = _fibonacci_levels(hi52, lo52)

        # ── Entry logic ──
        at_zone   = [z for z in zones_below if current_price <= z["high"] * 1.03]
        near_zone = [z for z in zones_below if current_price <= z["high"] * 1.12]

        if at_zone:
            zone = at_zone[0]
            pct  = round((current_price - zone["high"]) / current_price * 100, 1)
            stop = round(zone["low"] * 0.985, 2)
            entry = {
                "signal":      "AT ZONE — entry active",
                "zone":        zone,
                "entry_range": f"${zone['low']:.2f} – ${zone['high']:.2f}",
                "stop":        f"${stop:.2f}  (below zone, ~{round((current_price - stop)/current_price*100,1)}% risk)",
                "note":        f"Price is within 3% of demand zone ({pct:+.1f}%) — buyers expected here",
            }
        elif near_zone:
            zone = near_zone[0]
            pct  = round((current_price - zone["high"]) / current_price * 100, 1)
            stop = round(zone["low"] * 0.985, 2)
            entry = {
                "signal":      "WAIT FOR PULLBACK",
                "zone":        zone,
                "entry_range": f"${zone['low']:.2f} – ${zone['high']:.2f}",
                "stop":        f"${stop:.2f}  (~{round((zone['high'] - stop)/zone['high']*100,1)}% below zone)",
                "note":        f"Nearest demand zone is {pct:.1f}% below current price — set alert",
            }
        elif zones_below:
            zone = zones_below[0]
            pct  = round((current_price - zone["high"]) / current_price * 100, 1)
            stop = round(zone["low"] * 0.985, 2)
            entry = {
                "signal":      "EXTENDED — deep pullback needed",
                "zone":        zone,
                "entry_range": f"${zone['low']:.2f} – ${zone['high']:.2f}",
                "stop":        f"${stop:.2f}",
                "note":        f"Price is {pct:.1f}% above nearest zone — elevated risk to chase here",
            }
        else:
            entry = {
                "signal":      "NO CLEAR ZONE — insufficient history",
                "zone":        None,
                "entry_range": "N/A",
                "stop":        "N/A",
                "note":        "Not enough price history to identify reliable demand zones",
            }

        return {
            "zones": zones_below,
            "fibs":  fibs,
            "entry": entry,
            "hi52":  round(hi52, 2),
            "lo52":  round(lo52, 2),
        }

    except Exception:
        return {"zones": [], "fibs": {}, "entry": {"signal": "Error", "note": ""}, "hi52": 0, "lo52": 0}


# ── Report ────────────────────────────────────────────────────────────────────

def print_detail(rank, c):
    td   = c["tech_details"]
    fd   = c["fund_details"]
    tr   = c["trend"]
    dza  = c["demand_zone"]
    sep  = "─" * 65

    print(f"\n  RANK #{rank}  ·  {c['ticker']} — {c['name']}")
    print(f"  {'OVERALL RATING':16} {c['rating']}/10  "
          f"(Technical: {c['tech_score']}/5  |  Fundamental: {c['fund_score']}/5)")
    print(f"  {sep}")

    # ── Technical ──
    print("  TECHNICAL")
    rsi = td.get("rsi")
    if rsi is not None:
        if rsi < 30:      desc = "oversold — potential bounce"
        elif rsi <= 40:   desc = "mildly oversold"
        elif rsi <= 65:   desc = "healthy bullish zone"
        elif rsi <= 75:   desc = "mildly overbought"
        else:             desc = "overbought — caution"
        print(f"    RSI {rsi}  →  {desc}")

    above = td.get("above_ma50")
    if above is not None:
        label = "above" if above else "below"
        note  = "bullish short-term structure" if above else "trading below trend — weak near-term"
        print(f"    Price {label} 50-day MA  →  {note}")

    gc = td.get("golden_cross")
    if gc is not None:
        if gc:  print("    50 MA > 200 MA (Golden Cross)  →  long-term uptrend intact")
        else:   print("    50 MA < 200 MA  →  long-term trend under pressure")

    mom = td.get("momentum_3m")
    if mom is not None:
        if mom > 8:     note = "strong bullish momentum"
        elif mom > 2:   note = "moderate positive momentum"
        elif mom >= 0:  note = "sideways — no clear push"
        else:           note = "negative momentum, watch for reversal"
        print(f"    3-month return {mom:+.1f}%  →  {note}")

    # ── Fundamental ──
    print("  FUNDAMENTAL")
    rg = fd.get("revenue_growth")
    if rg is not None:
        if rg > 25:    note = "high-growth business"
        elif rg > 10:  note = "solid revenue expansion"
        elif rg > 5:   note = "moderate growth"
        else:          note = "slow / stalling top-line"
        print(f"    Revenue growth {rg:+.1f}%  →  {note}")

    gm = fd.get("gross_margin")
    if gm is not None:
        if gm > 60:   note = "premium margins — strong pricing power"
        elif gm > 40: note = "solid margins"
        elif gm > 25: note = "average margins"
        else:         note = "thin margins — cost-sensitive"
        print(f"    Gross margin {gm:.1f}%  →  {note}")

    roe = fd.get("roe")
    if roe is not None:
        if roe > 30:   note = "exceptional capital efficiency"
        elif roe > 15: note = "solid returns on equity"
        else:          note = "below-average capital returns"
        print(f"    ROE {roe:.1f}%  →  {note}")

    peg = fd.get("peg")
    if peg:
        if peg < 1.0:   note = "undervalued relative to growth"
        elif peg < 1.5: note = "fairly valued"
        elif peg < 2.0: note = "slight growth premium"
        else:           note = "expensive versus growth rate"
        print(f"    PEG {peg:.2f}  →  {note}")

    eg = fd.get("earnings_growth")
    if eg is not None:
        if eg > 20:    note = "accelerating profitability"
        elif eg > 10:  note = "growing earnings"
        elif eg >= 0:  note = "flat earnings"
        else:          note = "earnings declining — watch closely"
        print(f"    Earnings growth {eg:+.1f}%  →  {note}")

    # ── Trend ──
    print("  TREND")
    if tr:
        slope_dir = "rising" if tr.get("ma_slope", 0) > 0 else "falling"
        print(f"    {tr.get('label','N/A')}  (ADX {tr.get('adx','?')}  |  +DI {tr.get('pdi','?')}  /  -DI {tr.get('mdi','?')})")
        print(f"    50-day MA slope: {tr.get('ma_slope',0):+.2f}%  ({slope_dir} over last 20 sessions)")

    # ── Demand Zones & Entry ──
    print("  DEMAND ZONES & ENTRY")
    price = c["price"] or 0

    zones = dza.get("zones", [])
    fibs  = dza.get("fibs", {})
    entry = dza.get("entry", {})
    hi52  = dza.get("hi52", 0)
    lo52  = dza.get("lo52", 0)

    if price:
        print(f"    52-week range: ${lo52:.2f} – ${hi52:.2f}  |  Current: ${price:.2f}")

    if zones:
        print(f"    Key demand zones (strongest first):")
        for z in zones[:3]:
            pct_away = (price - z["high"]) / price * 100 if price else 0
            strength_label = "★★★" if z["touches"] >= 3 else "★★" if z["touches"] == 2 else "★"
            print(f"      {strength_label}  ${z['low']:.2f} – ${z['high']:.2f}  "
                  f"({z['touches']} touches)  [{pct_away:.1f}% away]")

    if fibs:
        print(f"    Fibonacci retracements (from ${hi52:.2f} high):")
        for level, price_lvl in fibs.items():
            pct_away = (price - price_lvl) / price * 100 if price else 0
            marker = " ◄ near current price" if abs(pct_away) < 3 else ""
            print(f"      {level}: ${price_lvl:.2f}  ({pct_away:+.1f}%){marker}")

    if entry:
        print(f"\n    ► ENTRY SIGNAL:  {entry.get('signal','')}")
        if entry.get("entry_range") and entry.get("entry_range") != "N/A":
            print(f"      Entry zone:    {entry['entry_range']}")
            print(f"      Stop loss:     {entry['stop']}")
        print(f"      {entry.get('note','')}")


# ── Fundamental Comparison ──────────────────────────────────────────────────────

def compare_fundamentals(tickers):
    """Side-by-side fundamental comparison of two or more tickers."""
    rows = []
    for ticker in tickers:
        print(f"  Fetching {ticker} …", end="\r")
        try:
            info = yf.Ticker(ticker).info
            score, d = fundamental_score(info)
            rows.append({
                "ticker": ticker,
                "name":   info.get("shortName", ticker),
                "price":  info.get("currentPrice") or info.get("regularMarketPrice"),
                "score":  round(score, 2),
                "d":      d,
            })
        except Exception:
            print(f"  Could not fetch data for {ticker} — skipping.")

    print(" " * 40)
    if not rows:
        print("No fundamental data could be fetched for the given tickers.")
        sys.exit(1)

    # Metrics to display: (label, key, suffix, "higher"/"lower" is better)
    metrics = [
        ("Revenue growth", "revenue_growth", "%",  "higher"),
        ("Gross margin",   "gross_margin",   "%",  "higher"),
        ("ROE",            "roe",            "%",  "higher"),
        ("PEG",            "peg",            "",   "lower"),
        ("Earnings growth","earnings_growth","%",  "higher"),
    ]

    name_w = 17
    col_w  = max(12, max(len(r["ticker"]) for r in rows) + 2)
    W      = name_w + col_w * len(rows) + 2

    print(f"\n{'═'*W}")
    print(f"  FUNDAMENTAL COMPARISON  ({datetime.now().strftime('%Y-%m-%d')})")
    print(f"{'═'*W}")

    # Header rows
    header = f"  {'':<{name_w}}" + "".join(f"{r['ticker']:>{col_w}}" for r in rows)
    print(f"\n{header}")
    print(f"  {'':<{name_w}}" + "".join(f"{(r['name'][:col_w-2]):>{col_w}}" for r in rows))
    print(f"  {'─'*(W-2)}")

    price_line = f"  {'Price':<{name_w}}" + "".join(
        f"{('$%.2f' % r['price']) if r['price'] else 'N/A':>{col_w}}" for r in rows)
    print(price_line)
    print(f"  {'─'*(W-2)}")

    # One line per metric, marking the winner with ◄
    for label, key, suffix, better in metrics:
        vals = [r["d"].get(key) for r in rows]
        present = [v for v in vals if v is not None]
        best = None
        if present:
            best = max(present) if better == "higher" else min([v for v in present if v and v > 0] or present)
        cells = []
        for v in vals:
            if v is None:
                cells.append(f"{'—':>{col_w}}")
            else:
                txt = f"{v:.2f}{suffix}" if key == "peg" else f"{v:+.1f}{suffix}"
                mark = " ◄" if (best is not None and v == best) else ""
                cells.append(f"{txt + mark:>{col_w}}")
        print(f"  {label:<{name_w}}" + "".join(cells))

    print(f"  {'─'*(W-2)}")
    score_line = f"  {'SCORE (0–5)':<{name_w}}" + "".join(
        f"{('%.2f' % r['score']):>{col_w}}" for r in rows)
    print(score_line)

    # Verdict
    top = max(rows, key=lambda r: r["score"])
    ties = [r for r in rows if r["score"] == top["score"]]
    print(f"\n{'═'*W}")
    if len(ties) > 1:
        names = ", ".join(r["ticker"] for r in ties)
        print(f"  VERDICT: tie on fundamental score ({top['score']}/5) — {names}")
    else:
        print(f"  VERDICT: {top['ticker']} has the stronger fundamentals "
              f"({top['score']}/5)")
    print("  ◄ marks the best value for each metric (PEG: lower is better).")
    print(f"{'═'*W}")
    print("  Not financial advice. Always do your own research.")
    print(f"{'═'*W}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ETF Stock Screener — Top 7 Ranked")
    parser.add_argument("etf", nargs="?", help="ETF ticker (e.g. QQQ, XLK, IBB)")
    parser.add_argument("-c", "--compare", nargs="+", metavar="TICKER",
                        help="Compare fundamentals of two or more stocks "
                             "(e.g. --compare MU SNDK)")
    args = parser.parse_args()

    if args.compare:
        tickers = [t.upper() for t in args.compare]
        if len(tickers) < 2:
            print("  --compare needs at least two tickers (e.g. --compare MU SNDK).")
            sys.exit(1)
        compare_fundamentals(tickers)
        return

    etf_ticker = (args.etf or input("Enter ETF ticker: ").strip()).upper()

    print(f"\nFetching holdings for {etf_ticker} …")
    tickers = get_etf_holdings(etf_ticker)

    if not tickers:
        print(f"\n  Could not fetch holdings for '{etf_ticker}'.")
        print("  Check that the ticker is a valid ETF (e.g. QQQ, XLK, IBB, ARKK).")
        sys.exit(1)

    print(f"Found {len(tickers)} holdings. Analyzing …\n")

    candidates = []
    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i:>2}/{len(tickers)}] {ticker:<8}", end="\r")
        try:
            t    = yf.Ticker(ticker)
            info = t.info
            name  = info.get("shortName", ticker)
            price = info.get("currentPrice") or info.get("regularMarketPrice")

            hist = get_history(ticker)
            if hist is None:
                continue

            f_score, fd = fundamental_score(info)
            t_score, td = technical_score(hist)
            rating       = combined_rating(t_score, f_score)
            trend        = analyze_trend(hist)
            dza          = demand_zone_analysis(hist, price) if price else {}

            candidates.append({
                "ticker":      ticker,
                "name":        name,
                "price":       price,
                "rating":      rating,
                "tech_score":  round(t_score, 1),
                "fund_score":  round(f_score, 1),
                "tech_details": td,
                "fund_details": fd,
                "trend":       trend,
                "demand_zone": dza,
            })
        except Exception:
            continue

    print(" " * 50)

    if not candidates:
        print("No valid stocks found in this ETF's holdings.")
        sys.exit(1)

    candidates.sort(key=lambda x: x["rating"], reverse=True)
    top7 = candidates[:7]

    # ── Summary table ──
    W = 67
    print(f"\n{'═'*W}")
    print(f"  ETF: {etf_ticker}  —  TOP 7 STOCKS  ({datetime.now().strftime('%Y-%m-%d')})")
    print(f"  Screened {len(candidates)} of {len(tickers)} holdings")
    print(f"{'═'*W}")
    print(f"\n  {'#':<3} {'Ticker':<8} {'Name':<24} {'Price':>8}  {'Rating':>6}  {'Trend'}")
    print(f"  {'─'*65}")
    for rank, c in enumerate(top7, 1):
        price_str  = f"${c['price']:.2f}" if c["price"] else "  N/A"
        trend_lbl  = c["trend"].get("label", "")[:22] if c["trend"] else ""
        print(f"  {rank:<3} {c['ticker']:<8} {c['name'][:23]:<24} {price_str:>8}  {c['rating']:>5}/10  {trend_lbl}")

    # ── Detailed analysis ──
    print(f"\n\n{'═'*W}")
    print(f"  DETAILED ANALYSIS  (Technical · Fundamental · Trend · Entry)")
    print(f"{'═'*W}")
    for rank, c in enumerate(top7, 1):
        print_detail(rank, c)

    print(f"\n{'═'*W}")
    print("  Not financial advice. Always do your own research.")
    print(f"{'═'*W}\n")


if __name__ == "__main__":
    main()
