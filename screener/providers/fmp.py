from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from screener.models import (
    AnalystEstimate,
    AssetClass,
    BalanceSheetPeriod,
    CashFlowPeriod,
    EarningsSurprise,
    FiscalPeriod,
    FundamentalSnapshot,
    Holding,
    IncomeStatementPeriod,
)
from screener.providers.base import FundamentalsProvider, HoldingsProvider, is_retryable_http_error

BASE_URL = "https://financialmodelingprep.com/api/v3"
BASE_URL_V4 = "https://financialmodelingprep.com/api/v4"


class _FMPClientMixin:
    def __init__(self, api_key: str, client: Optional[httpx.AsyncClient] = None, timeout: float = 15.0):
        if not api_key:
            raise ValueError("FMP provider requires an api_key (set FMP_API_KEY)")
        self._api_key = api_key
        self._client = client
        self._timeout = timeout

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential_jitter(initial=1, max=20),
        retry=retry_if_exception(is_retryable_http_error),
    )
    async def _get(self, url: str, params: Optional[dict] = None) -> httpx.Response:
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=self._timeout)
        try:
            resp = await client.get(url, params={**(params or {}), "apikey": self._api_key})
            resp.raise_for_status()
            return resp
        finally:
            if owns_client:
                await client.aclose()


class FMPHoldingsProvider(_FMPClientMixin, HoldingsProvider):
    name = "fmp"

    async def get_holdings(self, etf_ticker: str) -> list[Holding]:
        resp = await self._get(f"{BASE_URL_V4}/etf-holdings", params={"symbol": etf_ticker, "date": date.today().isoformat()})
        holdings = []
        for row in resp.json():
            ticker = (row.get("asset") or "").strip()
            if not ticker:
                continue
            holdings.append(
                Holding(
                    ticker=ticker,
                    name=row.get("name", ticker),
                    weight_pct=float(row.get("weightPercentage", 0) or 0),
                    sector=row.get("sector"),
                    asset_class=AssetClass.EQUITY,
                    currency=row.get("currency") or "USD",
                    source="fmp",
                )
            )
        return holdings


def _period(item: dict, is_ttm: bool = False) -> FiscalPeriod:
    d = item.get("date")
    period_end = datetime.strptime(d, "%Y-%m-%d").date() if d else date.today()
    period_str = str(item.get("period", ""))
    fiscal_quarter = int(period_str[1:]) if period_str.startswith("Q") and period_str[1:].isdigit() else None
    return FiscalPeriod(fiscal_year=period_end.year, fiscal_quarter=fiscal_quarter, period_end=period_end, is_ttm=is_ttm)


class FMPFundamentalsProvider(_FMPClientMixin, FundamentalsProvider):
    name = "fmp"

    async def get_snapshot(self, ticker: str) -> FundamentalSnapshot:
        try:
            income_q = await self._get(f"{BASE_URL}/income-statement/{ticker}", params={"period": "quarter", "limit": 8})
            income_a = await self._get(f"{BASE_URL}/income-statement/{ticker}", params={"period": "annual", "limit": 5})
            balance = await self._get(f"{BASE_URL}/balance-sheet-statement/{ticker}", params={"period": "quarter", "limit": 8})
            cashflow = await self._get(f"{BASE_URL}/cash-flow-statement/{ticker}", params={"period": "quarter", "limit": 8})
            cashflow_a = await self._get(f"{BASE_URL}/cash-flow-statement/{ticker}", params={"period": "annual", "limit": 5})
            estimates = await self._get(f"{BASE_URL}/analyst-estimates/{ticker}", params={"period": "annual", "limit": 2})
            surprises = await self._get(f"{BASE_URL}/earnings-surprises/{ticker}", params={"limit": 4})
            profile = await self._get(f"{BASE_URL}/profile/{ticker}")
        except httpx.HTTPStatusError as exc:
            return FundamentalSnapshot(ticker=ticker, fetch_failed=True, fetch_error=str(exc))

        profile_data = (profile.json() or [{}])[0]

        income_quarterly = [
            IncomeStatementPeriod(
                period=_period(row),
                revenue=row.get("revenue"),
                gross_profit=row.get("grossProfit"),
                operating_income=row.get("operatingIncome"),
                net_income=row.get("netIncome"),
                ebit=row.get("operatingIncome"),
                ebitda=row.get("ebitda"),
                interest_expense=row.get("interestExpense"),
                diluted_shares_outstanding=row.get("weightedAverageShsOutDil"),
                diluted_eps=row.get("epsdiluted"),
            )
            for row in income_q.json()
        ]
        income_annual = [
            IncomeStatementPeriod(
                period=_period(row),
                revenue=row.get("revenue"),
                gross_profit=row.get("grossProfit"),
                operating_income=row.get("operatingIncome"),
                net_income=row.get("netIncome"),
                ebit=row.get("operatingIncome"),
                ebitda=row.get("ebitda"),
                interest_expense=row.get("interestExpense"),
                diluted_shares_outstanding=row.get("weightedAverageShsOutDil"),
                diluted_eps=row.get("epsdiluted"),
            )
            for row in income_a.json()
        ]
        balance_sheet = [
            BalanceSheetPeriod(
                period=_period(row),
                total_debt=row.get("totalDebt"),
                cash_and_equivalents=row.get("cashAndCashEquivalents"),
                total_equity=row.get("totalStockholdersEquity"),
                total_assets=row.get("totalAssets"),
                current_assets=row.get("totalCurrentAssets"),
                current_liabilities=row.get("totalCurrentLiabilities"),
            )
            for row in balance.json()
        ]
        cash_flow = [
            CashFlowPeriod(
                period=_period(row),
                operating_cash_flow=row.get("operatingCashFlow"),
                capex=row.get("capitalExpenditure"),
                free_cash_flow=row.get("freeCashFlow"),
            )
            for row in cashflow.json()
        ]
        cash_flow_annual = [
            CashFlowPeriod(
                period=_period(row),
                operating_cash_flow=row.get("operatingCashFlow"),
                capex=row.get("capitalExpenditure"),
                free_cash_flow=row.get("freeCashFlow"),
            )
            for row in cashflow_a.json()
        ]
        analyst_estimates = [
            AnalystEstimate(
                fiscal_year=datetime.strptime(row["date"], "%Y-%m-%d").year,
                consensus_revenue=row.get("estimatedRevenueAvg"),
                consensus_eps=row.get("estimatedEpsAvg"),
                as_of=date.today(),
            )
            for row in estimates.json()
            if row.get("date")
        ]
        earnings_surprises = [
            EarningsSurprise(
                period=FiscalPeriod(
                    fiscal_year=datetime.strptime(row["date"], "%Y-%m-%d").year,
                    period_end=datetime.strptime(row["date"], "%Y-%m-%d").date(),
                ),
                actual_eps=row.get("actualEarningResult"),
                estimated_eps=row.get("estimatedEarning"),
                surprise_pct=(
                    (row["actualEarningResult"] - row["estimatedEarning"]) / abs(row["estimatedEarning"]) * 100
                    if row.get("actualEarningResult") is not None and row.get("estimatedEarning")
                    else None
                ),
            )
            for row in surprises.json()
            if row.get("date")
        ]

        revenue_ttm = sum(p.revenue or 0 for p in income_quarterly[:4]) if len(income_quarterly) >= 4 else None

        return FundamentalSnapshot(
            ticker=ticker,
            sector=profile_data.get("sector"),
            industry=profile_data.get("industry"),
            market_cap=profile_data.get("mktCap"),
            currency=profile_data.get("currency") or "USD",
            pre_revenue=bool(revenue_ttm is not None and revenue_ttm < 1_000_000),
            income_quarterly=income_quarterly,
            income_annual=income_annual,
            balance_sheet=balance_sheet,
            cash_flow=cash_flow,
            cash_flow_annual=cash_flow_annual,
            analyst_estimates=analyst_estimates,
            estimate_revisions=[],  # FMP v3 has no direct revision-history endpoint; wire in when a source is picked
            earnings_surprises=earnings_surprises,
        )
