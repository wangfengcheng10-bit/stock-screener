from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd
import yfinance as yf

from screener.models import AnalystEstimate, BalanceSheetPeriod, CashFlowPeriod, FiscalPeriod, FundamentalSnapshot, Holding, IncomeStatementPeriod
from screener.providers.base import FundamentalsProvider, HoldingsProvider


def _period_from_column(col) -> FiscalPeriod:
    ts = pd.Timestamp(col)
    return FiscalPeriod(fiscal_year=ts.year, fiscal_quarter=(ts.month - 1) // 3 + 1, period_end=ts.date())


def _row(df: pd.DataFrame, *labels: str) -> Optional[pd.Series]:
    for label in labels:
        if label in df.index:
            return df.loc[label]
    return None


def _safe(row: Optional[pd.Series], col) -> Optional[float]:
    if row is None or col not in row.index:
        return None
    val = row[col]
    return float(val) if pd.notna(val) else None


class YFinanceFundamentalsProvider(FundamentalsProvider):
    """Free fallback. Thinner coverage than FMP: no estimate-revision history,
    minimal earnings-surprise depth, and less consistent field naming across tickers."""

    name = "yfinance"

    async def get_snapshot(self, ticker: str) -> FundamentalSnapshot:
        try:
            t = yf.Ticker(ticker)
            info = t.info
            q_fin, a_fin = t.quarterly_financials, t.financials
            q_bs, q_cf = t.quarterly_balance_sheet, t.quarterly_cashflow
        except Exception as exc:
            return FundamentalSnapshot(ticker=ticker, fetch_failed=True, fetch_error=str(exc))

        if q_fin is None or q_fin.empty:
            return FundamentalSnapshot(ticker=ticker, fetch_failed=True, fetch_error="no quarterly financials returned")

        def _build_income(df: pd.DataFrame) -> list[IncomeStatementPeriod]:
            revenue = _row(df, "Total Revenue", "TotalRevenue")
            gross = _row(df, "Gross Profit", "GrossProfit")
            op_income = _row(df, "Operating Income", "OperatingIncome")
            net_income = _row(df, "Net Income", "NetIncome")
            ebit = _row(df, "EBIT")
            ebitda = _row(df, "EBITDA")
            interest = _row(df, "Interest Expense", "InterestExpense")
            shares = _row(df, "Diluted Average Shares", "DilutedAverageShares")
            eps = _row(df, "Diluted EPS", "DilutedEPS")
            return [
                IncomeStatementPeriod(
                    period=_period_from_column(col),
                    revenue=_safe(revenue, col),
                    gross_profit=_safe(gross, col),
                    operating_income=_safe(op_income, col),
                    net_income=_safe(net_income, col),
                    ebit=_safe(ebit, col),
                    ebitda=_safe(ebitda, col),
                    interest_expense=_safe(interest, col),
                    diluted_shares_outstanding=_safe(shares, col),
                    diluted_eps=_safe(eps, col),
                )
                for col in df.columns
            ]

        def _build_balance(df: pd.DataFrame) -> list[BalanceSheetPeriod]:
            debt = _row(df, "Total Debt", "TotalDebt")
            cash = _row(df, "Cash And Cash Equivalents", "CashAndCashEquivalents")
            equity = _row(df, "Stockholders Equity", "StockholdersEquity")
            assets = _row(df, "Total Assets", "TotalAssets")
            cur_assets = _row(df, "Current Assets", "CurrentAssets")
            cur_liab = _row(df, "Current Liabilities", "CurrentLiabilities")
            return [
                BalanceSheetPeriod(
                    period=_period_from_column(col),
                    total_debt=_safe(debt, col),
                    cash_and_equivalents=_safe(cash, col),
                    total_equity=_safe(equity, col),
                    total_assets=_safe(assets, col),
                    current_assets=_safe(cur_assets, col),
                    current_liabilities=_safe(cur_liab, col),
                )
                for col in df.columns
            ]

        def _build_cashflow(df: pd.DataFrame) -> list[CashFlowPeriod]:
            ocf = _row(df, "Operating Cash Flow", "OperatingCashFlow", "Total Cash From Operating Activities")
            capex = _row(df, "Capital Expenditure", "CapitalExpenditure")
            fcf = _row(df, "Free Cash Flow", "FreeCashFlow")
            out = []
            for col in df.columns:
                ocf_v, capex_v, fcf_v = _safe(ocf, col), _safe(capex, col), _safe(fcf, col)
                if fcf_v is None and ocf_v is not None and capex_v is not None:
                    fcf_v = ocf_v + capex_v  # yfinance reports capex as a negative number
                out.append(CashFlowPeriod(period=_period_from_column(col), operating_cash_flow=ocf_v, capex=capex_v, free_cash_flow=fcf_v))
            return out

        income_quarterly = _build_income(q_fin)
        income_annual = _build_income(a_fin) if a_fin is not None and not a_fin.empty else []
        balance_sheet = _build_balance(q_bs) if q_bs is not None and not q_bs.empty else []
        cash_flow = _build_cashflow(q_cf) if q_cf is not None and not q_cf.empty else []

        revenue_ttm = sum(p.revenue or 0 for p in income_quarterly[:4]) if len(income_quarterly) >= 4 else None

        analyst_estimates = []
        fwd_growth = info.get("revenueGrowth")
        if fwd_growth is not None and revenue_ttm is not None:
            analyst_estimates.append(
                AnalystEstimate(
                    fiscal_year=date.today().year + 1,
                    consensus_revenue=revenue_ttm * (1 + fwd_growth),
                    consensus_eps=info.get("forwardEps"),
                    as_of=date.today(),
                )
            )

        return FundamentalSnapshot(
            ticker=ticker,
            sector=info.get("sector"),
            industry=info.get("industry"),
            market_cap=info.get("marketCap"),
            currency=info.get("currency") or "USD",
            pre_revenue=bool(revenue_ttm is not None and revenue_ttm < 1_000_000),
            income_quarterly=income_quarterly,
            income_annual=income_annual,
            balance_sheet=balance_sheet,
            cash_flow=cash_flow,
            analyst_estimates=analyst_estimates,
            estimate_revisions=[],
            earnings_surprises=[],
        )


class YFinanceHoldingsProvider(HoldingsProvider):
    """Intentionally NOT part of the default holdings cascade: yfinance only exposes
    an ETF's top ~10-30 holdings, and a partial universe corrupts every percentile
    computed downstream. Kept only so the interface exists for explicit, opt-in debugging."""

    name = "yfinance"

    async def get_holdings(self, etf_ticker: str) -> list[Holding]:
        raise NotImplementedError(
            "yfinance only exposes partial ETF holdings and must not be used for universe "
            "resolution. Use IssuerHoldingsProvider or FMPHoldingsProvider instead."
        )
