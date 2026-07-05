# Can You Beat the S&P 500? A Survey of the Empirical Evidence

This is a summary of the most-cited, peer-reviewed and industry-standard empirical
studies on whether swing trading, active stock-picking, or professionally managed
funds actually beat a simple S&P 500 index over time — and how common success
actually is in practice. It's meant as background reading for anyone using the
tools in this repo (`stock_screener.py`, `ai_infra_agent.py`) to set realistic
expectations: these tools help find *setups*, they don't repeal the base rates
below.

## TL;DR

| Approach | What the data shows | How often it "works" |
|---|---|---|
| Professional active fund managers | Most underperform their benchmark, especially over 10–20 yrs | ~8–20% beat the S&P 500 over 15–20 years |
| Hedge funds | Underperform net of fees vs. a plain index fund | Buffett's bet: index won 7.1%/yr vs 2.2%/yr over 10 yrs |
| Retail swing/position traders | A small minority outperform; most underperform buy-and-hold | Roughly consistent with "most people lose to the index" |
| Retail day traders | Overwhelming majority lose money | ~97% lose money (Brazil); <1% show skill (Taiwan) |
| Systematic momentum/trend factors | Real, replicable edge exists in academic samples | Averages ~1%/month gross in original sample, but decays ~50% after publication/fees |
| Average individual investor / behavior gap | Underperforms even the funds they're invested in, due to timing | ~1–8 percentage points/year behind the index depending on the year |
| Long-term buy-and-hold | Matches the index by definition; beaten mainly by impatience | The main risk is missing a handful of best days, not stock selection |

Below is the detail, organized by category, with sources.

---

## 1. Professional active managers rarely beat the index

**S&P Dow Jones Indices SPIVA Scorecard** (the standard, most rigorous
industry benchmark study, run annually since 2002) tracks actual fund returns
against their stated benchmarks, correcting for survivorship bias.

- Full-year 2024: **65%** of active large-cap U.S. equity funds underperformed
  the S&P 500.
- Full-year 2025: underperformance worsened to **79%** — the fourth-worst year
  in the report's 25-year history — despite mid-year 2025 numbers looking
  better (54% underperforming through H1).
- Over **15 years**, there is no equity or fixed-income category in the SPIVA
  universe where a majority of active managers beat their benchmark.
- Over **20 years**, roughly **92%** of domestic active funds underperformed.

The **SPIVA Persistence Scorecard** goes further: even the active managers who
*do* beat the market in a given period rarely repeat it — top-quartile
performance is close to random from one period to the next, which matches
academic findings (see Carhart, below).

**Carhart (1997), "On Persistence in Mutual Fund Performance,"** *Journal of
Finance* — the classic academic paper. Using a survivorship-bias-free sample,
Carhart found:
- Apparent "hot hand" fund performance is almost entirely explained by common
  risk factors (market, size, value) plus the one-year momentum effect, not
  manager skill.
- The only *persistent* effect is at the bottom: bad funds stay bad (mostly
  because of high expenses).
- Conclusion: little to no evidence of skilled active management once fees
  and factor exposure are accounted for.

## 2. Hedge funds: same story, worse fees

**Warren Buffett's 2007 bet** (formalized via Long Bets/Protégé Partners) is
the most famous real-money test: $1M wagered that a low-cost S&P 500 index
fund would beat a hand-picked basket of five funds-of-hedge-funds over
2008–2017, net of all fees.
- Result: the index fund returned **~125.8% cumulative (~7.1%/yr)**; the
  hedge fund basket returned **~36% cumulative (~2.2%/yr)**. Not close.
- Buffett's stated thesis going in: high fees ("helpers") mathematically
  guarantee that active managers underperform in aggregate, since the market
  return is a zero-sum game before costs and a negative-sum game after them.

## 3. Retail investors who trade actively underperform buy-and-hold

**Barber & Odean (2000), "Trading Is Hazardous to Your Wealth,"** *Journal of
Finance* — analyzed 66,465 discount-brokerage households, 1991–1996:
- Households in the **top quintile of turnover** (most active traders) earned
  **11.4%/yr net**, vs. a market return of **17.9%/yr** — a ~6.5 point/year
  gap, driven by trading costs and poor selection.
- The *average* household still underperformed the market (16.4%/yr) despite
  turning over 75% of its portfolio per year.
- Their explanation: **overconfidence** drives excess trading, and excess
  trading is the single best predictor of underperformance.

## 4. Day trading specifically: the base rate is brutal

- **Barber, Lee, Liu & Odean, "The Cross-Section of Speculator Skill:
  Evidence from Day Trading"** (Taiwan futures data) — fewer than **1%** of
  day traders were able to predictably and reliably earn positive
  net-of-fee abnormal returns. A slightly larger group (~10-15%) broke even
  or profited before costs, but transaction costs erased it for almost all.
- **Brazilian equity futures study (2013–2015 cohort, published in the
  Review of Finance-adjacent literature)** — of individuals who began day
  trading and persisted for at least 300 days, **97% lost money**, and fewer
  than 1% earned more than a Brazilian minimum wage from it.
- Net takeaway: day trading is close to a negative-sum game after costs, and
  persistent, reliable skill is empirically rare, not just "hard."

## 5. Options / attention-driven retail trading (the "Robinhood era")

**Barber, Huang, Odean & Schwarz, "Attention Induced Trading and Returns:
Evidence from Robinhood Users"** (*Journal of Finance*, 2021) and related
work on retail options flow:
- Commission-free trading and app design (push notifications, gamified UX)
  measurably increase trading in attention-grabbing, high-volatility stocks.
- Herding into the same names (the GameStop/meme-stock pattern) predicts
  *lower* subsequent returns for the herd, even though the stocks themselves
  can spike sharply in the short term.
- Retail options traders, especially around earnings announcements, tend to
  demand liquidity without an information edge and show a persistent net
  loss pattern — consistent with the day-trading findings above.

## 6. Is there a real, exploitable edge anywhere? Yes — but it decays

**Jegadeesh & Titman (1993), "Returns to Buying Winners and Selling
Losers,"** *Journal of Finance* — the seminal momentum/swing paper:
- Buying past 3–12 month winners and shorting past 3–12 month losers produced
  positive returns in every one of 16 formation/holding period combinations
  tested, 1965–1989.
- The strongest combination (12-month formation, 3-month holding) generated
  about **1.31%/month** for the winner-minus-loser spread — a genuinely large
  effect in academic terms, roughly 12%/year for a long-short portfolio.
- This is real, peer-reviewed evidence that price momentum (the basis for
  much of swing trading) is a statistically robust anomaly, not pure noise.

**But: McLean & Pontiff (2016), "Does Academic Research Destroy Stock Return
Predictability?"**, *Journal of Finance*:
- Studied 97 known return-predicting signals (including momentum-style
  factors). Average returns were **26% lower out-of-sample** than in the
  original published sample (an upper bound for pure data-mining/overfitting
  bias), and **58% lower after publication** — implying roughly a further
  **32%** decline attributable to real-world arbitrage capital piling into
  the signal once it became public.
- Interpretation for a retail swing trader: a well-known, published
  edge (moving-average crossovers, classic momentum, etc.) is real in
  academic history, but has been substantially arbitraged away, and gross
  edges before costs get much thinner once slippage/fees/taxes are applied.

**Park & Irwin (2007), "What Do We Know About the Profitability of Technical
Analysis?"**, *Journal of Economic Surveys* (meta-analysis of 95 studies):
- 56 of 95 studies found statistically positive results for technical
  trading rules; 20 found negative results; 19 mixed.
- Positive results were concentrated in FX and futures markets (lower costs,
  more trending behavior), and mostly pre-1990s data; profitability in pure
  equity markets was weaker and largely explained by short-term momentum
  rather than classic chart patterns.
- The authors flag serious methodological issues across the literature:
  data snooping, ignoring transaction costs, and survivorship bias in which
  rules got tested/published.

## 7. The average investor underperforms even their own funds — the "behavior gap"

**DALBAR Quantitative Analysis of Investor Behavior (QAIB)** — measures what
investors actually earn (via fund flow timing) vs. the index/funds they're
invested in:
- **2024**: average equity investor earned **16.54%** vs. the S&P 500's
  **25.02%** — an **848 basis point** gap, driven almost entirely by
  mistimed buying/selling around volatility, not fund selection.
- **2025**: the gap narrowed to **~72 basis points** (17.16% vs. 17.88%),
  one of the smallest gaps since 1985 — showing the gap varies a lot year to
  year but rarely favors the investor.
- **20-year trailing (through 2024)**: average investor **9.24%/yr** vs.
  S&P 500 **10.35%/yr** — roughly a full percentage point/year lost to
  behavior, compounding to a large gap over decades.

## 8. The cost of trying to time it: missing a handful of days dominates outcomes

Multiple studies (Franklin Templeton, AQR "So What If You Miss the Market's N
Best Days," Wells Fargo Investment Institute) using S&P 500 daily data reach
the same conclusion:
- Missing the best **30 days** over a 30-year window (1995–2025) cuts the
  annualized return from **8.4% to 2.1%**.
- Missing the best **50 days** turns a positive multi-decade return
  **negative (-0.6%/yr)**.
- Best and worst days cluster together, usually inside bear markets/high
  volatility windows (e.g., 3 of the 30 best and 5 of the 30 worst days in
  the last 50 years all fell within an 8-day span in March 2020) — meaning
  "get out to avoid the crash, get back in after" is nearly impossible to
  execute in practice, because the recovery days arrive embedded in the
  crash itself.

---

## How to read this evidence set together

1. **The base rate is stacked against beating the index.** Across
   professional managers (SPIVA, Carhart), hedge funds (Buffett bet), retail
   swing traders (Barber & Odean), and day traders (Barber/Lee/Liu/Odean,
   Brazil study), the consistent finding is that a large majority
   underperform a simple S&P 500 index fund net of costs, and the fraction
   that beats it *persistently* (not just in one lucky period) is small —
   roughly single digits to low double-digit percent depending on the
   population and time horizon.
2. **Real, academically documented edges exist (momentum being the best
   evidenced one), but they are not free money.** Effect sizes shrink by
   roughly half after publication/arbitrage, and further after realistic
   transaction costs, slippage, and taxes are applied — which is why an
   edge that looks like 12%/year in a backtest often does not survive
   contact with live trading at retail scale.
3. **Most of the retail underperformance isn't stock selection — it's
   behavior.** DALBAR's investor gap and the "missing the best days"
   literature both point to *when* people buy and sell (panic selling,
   chasing performance, sitting in cash after a drawdown) as a bigger drag
   than which stocks they pick.
4. **Practical implication for tools like this repo's screener/agent:**
   demand-zone entries, trend scoring, and position sizing (what
   `stock_screener.py` and `ai_infra_agent.py` do) are attempts to
   systematize exactly the kind of momentum/mean-reversion signal that
   Jegadeesh & Titman and the technical-analysis literature show has *some*
   real basis — but the evidence above says the realistic edge is modest,
   erodes with popularity, and is easily wiped out by overtrading,
   undiscipline around stops/targets, or panic-driven timing. Backtested or
   scored setups should be sized and risk-managed accordingly, not treated
   as a guarantee of outperformance.

---

## Sources

- [SPIVA U.S. Year-End 2025 — S&P Dow Jones Indices](https://www.spglobal.com/spdji/en/spiva/article/spiva-us/)
- [SPIVA U.S. Persistence Scorecard Year-End 2025 — S&P Dow Jones Indices](https://www.spglobal.com/spdji/en/spiva/article/us-persistence-scorecard/)
- [SPIVA U.S. Scorecard Mid-Year 2025 (PDF)](https://d1e00ek4ebabms.cloudfront.net/production/uploaded-files/spiva-us-mid-year-2025-bc7a7f61-4b27-48b0-b20a-856cc87521d0.pdf)
- Carhart, M. (1997). "On Persistence in Mutual Fund Performance." *Journal of Finance*, 52(1), 57–82. [Wiley](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1540-6261.1997.tb03808.x)
- [Warren Buffett Wins $1M Bet — AEI](https://www.aei.org/carpe-diem/warren-buffett-wins-1m-bet-made-a-decade-ago-that-the-sp-500-stock-index-would-outperform-hedge-funds-and-it-wasnt-even-close/)
- [Betting with Buffett: Seven Lean Years Later — CFA Institute](https://blogs.cfainstitute.org/investor/2015/02/12/betting-with-buffett-seven-lean-years-later/)
- Barber, B. & Odean, T. (2000). "Trading Is Hazardous to Your Wealth: The Common Stock Investment Performance of Individual Investors." *Journal of Finance*, 55(2), 773–806. [Berkeley PDF](https://faculty.haas.berkeley.edu/odean/papers%20current%20versions/individual_investor_performance_final.pdf)
- Barber, B., Lee, Y-T., Liu, Y-J. & Odean, T. (2014). "The Cross-Section of Speculator Skill: Evidence from Day Trading." *Journal of Financial Markets*, 18. [Berkeley PDF](https://faculty.haas.berkeley.edu/odean/papers/day%20traders/The%20Cross-Section%20of%20Speculator%20Skill.pdf)
- [I Reviewed Every Major Day Trading Study from the Last 25 Years](https://medium.com/@faisal_haroon/i-reviewed-every-major-day-trading-study-from-the-last-25-years-the-data-is-devastating-4b116273b956) (secondary summary of the Brazil day-trading cohort study)
- Barber, B., Huang, X., Odean, T. & Schwarz, C. (2021). "Attention Induced Trading and Returns: Evidence from Robinhood Users." [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3715077)
- Jegadeesh, N. & Titman, S. (1993). "Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency." *Journal of Finance*, 48(1), 65–91. [Overview](https://foxholm.com/q/research/jegadeesh-titman-momentum/)
- McLean, R.D. & Pontiff, J. (2016). "Does Academic Research Destroy Stock Return Predictability?" *Journal of Finance*, 71(1), 5–32. [PDF](https://www.fmg.ac.uk/sites/default/files/2020-08/Jeffrey-Pontiff.pdf)
- Park, C-H. & Irwin, S. (2007). "What Do We Know About the Profitability of Technical Analysis?" *Journal of Economic Surveys*, 21(4), 786–826. [PDF](https://farmdoc.illinois.edu/assets/marketing/agmas/AgMAS04_04.pdf)
- [DALBAR QAIB](https://www.dalbar.com/qaib/) and [How The Average Investor's Returns Compare To The Market — Forbes](https://www.forbes.com/sites/wesmoss/2026/01/27/how-the-average-investors-returns-compare-to-the-market/)
- [DALBAR's 2026 QAIB Report — Morningstar](https://www.morningstar.com/news/pr-newswire/20260417ne37232/dalbars-2026-qaib-report-shows-narrower-investor-gap-amid-a-complex-and-volatile-market-year)
- [Investors Missed the Best of 2024's Market Gains — DALBAR/PR Newswire](https://www.prnewswire.com/news-releases/investors-missed-the-best-of-2024s-market-gains-latest-dalbar-investor-behavior-report-finds-302416023.html)
- [(So) What If You Miss the Market's N Best Days? — AQR](https://www.aqr.com/Insights/Perspectives/So-What-If-You-Miss-the-Markets-N-Best-Days)
- [The Cost of Missing the 10 Best Days in the Stock Market — FMP Wealth Advisers](https://fmpwa.com/the-cost-of-missing-the-10-best-days-in-the-stock-market/)
