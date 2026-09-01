from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AssetClass(str, Enum):
    EQUITY = "Equity"
    CASH = "Cash"
    FUTURE = "Future"
    FX_FORWARD = "FX Forward"
    BOND = "Bond"
    OTHER = "Other"


class Holding(BaseModel):
    ticker: str
    name: str
    weight_pct: float
    sector: Optional[str] = None
    asset_class: AssetClass = AssetClass.EQUITY
    currency: str = "USD"
    source: str = Field(description="provider that resolved this holding, e.g. 'ssga', 'invesco', 'fmp'")


class FiscalPeriod(BaseModel):
    fiscal_year: int
    fiscal_quarter: Optional[int] = None  # None for annual periods
    period_end: date
    is_ttm: bool = False


class IncomeStatementPeriod(BaseModel):
    period: FiscalPeriod
    revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None
    net_income: Optional[float] = None
    ebit: Optional[float] = None
    ebitda: Optional[float] = None
    interest_expense: Optional[float] = None
    diluted_shares_outstanding: Optional[float] = None
    diluted_eps: Optional[float] = None


class BalanceSheetPeriod(BaseModel):
    period: FiscalPeriod
    total_debt: Optional[float] = None
    cash_and_equivalents: Optional[float] = None
    total_equity: Optional[float] = None
    total_assets: Optional[float] = None
    current_assets: Optional[float] = None
    current_liabilities: Optional[float] = None


class CashFlowPeriod(BaseModel):
    period: FiscalPeriod
    operating_cash_flow: Optional[float] = None
    capex: Optional[float] = None
    free_cash_flow: Optional[float] = None


class AnalystEstimate(BaseModel):
    fiscal_year: int
    consensus_revenue: Optional[float] = None
    consensus_eps: Optional[float] = None
    as_of: date


class EstimateRevision(BaseModel):
    as_of: date
    consensus_next_fy_eps: Optional[float] = None


class EarningsSurprise(BaseModel):
    period: FiscalPeriod
    actual_eps: Optional[float] = None
    estimated_eps: Optional[float] = None
    surprise_pct: Optional[float] = None


class FundamentalSnapshot(BaseModel):
    ticker: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    currency: str = "USD"
    fx_rate_to_usd: float = 1.0
    pre_revenue: bool = False

    income_quarterly: list[IncomeStatementPeriod] = Field(default_factory=list)  # up to 8 quarters
    income_annual: list[IncomeStatementPeriod] = Field(default_factory=list)  # up to 5 years
    balance_sheet: list[BalanceSheetPeriod] = Field(default_factory=list)
    cash_flow: list[CashFlowPeriod] = Field(default_factory=list)  # up to 8 quarters
    cash_flow_annual: list[CashFlowPeriod] = Field(default_factory=list)  # up to 5 years
    analyst_estimates: list[AnalystEstimate] = Field(default_factory=list)
    estimate_revisions: list[EstimateRevision] = Field(default_factory=list)
    earnings_surprises: list[EarningsSurprise] = Field(default_factory=list)

    fetch_failed: bool = False
    fetch_error: Optional[str] = None
    as_of: date = Field(default_factory=date.today)


class MetricValue(BaseModel):
    name: str
    raw_value: Optional[float] = None
    winsorized_value: Optional[float] = None
    percentile: Optional[float] = None
    higher_is_better: bool = True
    is_missing: bool = False


class MetricSet(BaseModel):
    ticker: str
    metrics: dict[str, MetricValue] = Field(default_factory=dict)


class PillarScore(BaseModel):
    name: str
    score: float
    metrics_available: int
    metrics_total: int


class ScoreCard(BaseModel):
    ticker: str
    name: str
    sector: Optional[str] = None
    etf_weight_pct: float
    composite_score: float
    grade: str
    pillar_scores: dict[str, PillarScore]
    data_coverage_pct: float
    low_confidence: bool
    as_of: date
