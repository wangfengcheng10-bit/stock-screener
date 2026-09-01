from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable, Optional

from screener.models import FundamentalSnapshot

# US statutory rate, used as a NOPAT proxy for ROIC since neither provider
# exposes an effective tax rate line item.
DEFAULT_TAX_RATE = 0.21

PILLAR_GROWTH = "growth"
PILLAR_PROFITABILITY = "profitability"
PILLAR_BALANCE_SHEET = "balance_sheet"
PILLAR_CASH_FLOW = "cash_flow"
PILLAR_OUTLOOK = "outlook"


def _sorted_desc(periods: list) -> list:
    return sorted(periods, key=lambda p: p.period.period_end, reverse=True)


def _ttm(periods: list, field: str, offset: int = 0) -> Optional[float]:
    window = periods[offset : offset + 4]
    if len(window) < 4:
        return None
    values = [getattr(p, field) for p in window]
    if any(v is None for v in values):
        return None
    return sum(values)


def _yoy_growth_series(quarters: list) -> list[float]:
    """YoY growth for each leading quarter that has a same-quarter comparison
    4 slots back. With the 8 quarters this pipeline fetches, that caps out at
    4 points, not the 8 the spec's growth-consistency formula implies -- an
    8-point series would need 12 quarters of history, which the fetch stage
    (providers/fmp.py, providers/yfinance_provider.py) doesn't pull."""
    out = []
    for i in range(min(4, max(0, len(quarters) - 4))):
        cur, prior = quarters[i].revenue, quarters[i + 4].revenue
        if cur is None or not prior:
            continue
        out.append(cur / prior - 1)
    return out


# ---------------------------------------------------------------------------
# Pillar A -- Revenue Growth
# ---------------------------------------------------------------------------


def rev_growth_ttm_yoy(s: FundamentalSnapshot) -> Optional[float]:
    q = _sorted_desc(s.income_quarterly)
    ttm, ttm_prior = _ttm(q, "revenue", 0), _ttm(q, "revenue", 4)
    if ttm is None or not ttm_prior:
        return None
    return ttm / ttm_prior - 1


def rev_cagr_3y(s: FundamentalSnapshot) -> Optional[float]:
    a = _sorted_desc(s.income_annual)
    if len(a) < 4:
        return None
    rev_t, rev_t3 = a[0].revenue, a[3].revenue
    if not rev_t or not rev_t3 or rev_t3 <= 0:
        return None
    return (rev_t / rev_t3) ** (1 / 3) - 1


def growth_acceleration(s: FundamentalSnapshot) -> Optional[float]:
    series = _yoy_growth_series(_sorted_desc(s.income_quarterly))
    if len(series) < 4:
        return None
    return series[0] - series[3]


def growth_consistency(s: FundamentalSnapshot) -> Optional[float]:
    series = _yoy_growth_series(_sorted_desc(s.income_quarterly))
    if len(series) < 2:
        return None
    return -statistics.pstdev(series)


# ---------------------------------------------------------------------------
# Pillar B -- Profitability & Net Margin
# ---------------------------------------------------------------------------


def net_margin_ttm(s: FundamentalSnapshot) -> Optional[float]:
    q = _sorted_desc(s.income_quarterly)
    ni, rev = _ttm(q, "net_income"), _ttm(q, "revenue")
    if ni is None or not rev:
        return None
    return ni / rev


def net_margin_trend(s: FundamentalSnapshot) -> Optional[float]:
    """Current TTM net margin minus the average annual net margin over the
    trailing (up to) 3 fiscal years, in basis points."""
    current = net_margin_ttm(s)
    annual = _sorted_desc(s.income_annual)[:3]
    if current is None or not annual:
        return None
    margins = [p.net_income / p.revenue for p in annual if p.net_income is not None and p.revenue]
    if not margins:
        return None
    return (current - sum(margins) / len(margins)) * 10_000


def operating_margin_ttm(s: FundamentalSnapshot) -> Optional[float]:
    q = _sorted_desc(s.income_quarterly)
    op, rev = _ttm(q, "operating_income"), _ttm(q, "revenue")
    if op is None or not rev:
        return None
    return op / rev


def gross_margin_ttm(s: FundamentalSnapshot) -> Optional[float]:
    q = _sorted_desc(s.income_quarterly)
    gp, rev = _ttm(q, "gross_profit"), _ttm(q, "revenue")
    if gp is None or not rev:
        return None
    return gp / rev


def roic(s: FundamentalSnapshot) -> Optional[float]:
    ebit = _ttm(_sorted_desc(s.income_quarterly), "ebit")
    bs = _sorted_desc(s.balance_sheet)
    if ebit is None or not bs:
        return None
    latest = bs[0]
    if latest.total_debt is None or latest.total_equity is None or latest.cash_and_equivalents is None:
        return None
    invested_capital = latest.total_debt + latest.total_equity - latest.cash_and_equivalents
    if invested_capital <= 0:
        return None
    return (ebit * (1 - DEFAULT_TAX_RATE)) / invested_capital


# ---------------------------------------------------------------------------
# Pillar C -- Balance Sheet Health
# ---------------------------------------------------------------------------


def net_debt_ebitda(s: FundamentalSnapshot) -> Optional[float]:
    bs = _sorted_desc(s.balance_sheet)
    ebitda = _ttm(_sorted_desc(s.income_quarterly), "ebitda")
    if not bs or not ebitda:
        return None
    latest = bs[0]
    if latest.total_debt is None or latest.cash_and_equivalents is None:
        return None
    return (latest.total_debt - latest.cash_and_equivalents) / ebitda


def debt_equity(s: FundamentalSnapshot) -> Optional[float]:
    bs = _sorted_desc(s.balance_sheet)
    if not bs or bs[0].total_debt is None or not bs[0].total_equity:
        return None
    return bs[0].total_debt / bs[0].total_equity


def current_ratio(s: FundamentalSnapshot) -> Optional[float]:
    bs = _sorted_desc(s.balance_sheet)
    if not bs or bs[0].current_assets is None or not bs[0].current_liabilities:
        return None
    return bs[0].current_assets / bs[0].current_liabilities


def interest_coverage(s: FundamentalSnapshot) -> Optional[float]:
    q = _sorted_desc(s.income_quarterly)
    ebit, interest = _ttm(q, "ebit"), _ttm(q, "interest_expense")
    if ebit is None or not interest:
        return None
    return ebit / abs(interest)


def dilution(s: FundamentalSnapshot) -> Optional[float]:
    a = _sorted_desc(s.income_annual)
    if len(a) < 4:
        return None
    latest, past = a[0].diluted_shares_outstanding, a[3].diluted_shares_outstanding
    if not latest or not past:
        return None
    return -1 * (latest / past - 1)


# ---------------------------------------------------------------------------
# Pillar D -- Cash Flow
# ---------------------------------------------------------------------------


def fcf_margin(s: FundamentalSnapshot) -> Optional[float]:
    cf = _sorted_desc(s.cash_flow)
    ocf, capex = _ttm(cf, "operating_cash_flow"), _ttm(cf, "capex")
    rev = _ttm(_sorted_desc(s.income_quarterly), "revenue")
    if ocf is None or capex is None or not rev:
        return None
    return (ocf + capex) / rev  # capex is stored as a negative outflow


def fcf_conversion(s: FundamentalSnapshot) -> Optional[float]:
    cf = _sorted_desc(s.cash_flow)
    ocf, capex = _ttm(cf, "operating_cash_flow"), _ttm(cf, "capex")
    ni = _ttm(_sorted_desc(s.income_quarterly), "net_income")
    if ocf is None or capex is None or not ni:
        return None
    return (ocf + capex) / ni


def ocf_cagr_3y(s: FundamentalSnapshot) -> Optional[float]:
    a = _sorted_desc(s.cash_flow_annual)
    if len(a) < 4:
        return None
    ocf_t, ocf_t3 = a[0].operating_cash_flow, a[3].operating_cash_flow
    # A CAGR off a zero or negative base is not a growth rate -- report it
    # missing rather than emit a number that ranks as if it were one.
    if ocf_t is None or not ocf_t3 or ocf_t3 <= 0 or ocf_t <= 0:
        return None
    return (ocf_t / ocf_t3) ** (1 / 3) - 1


def capex_intensity(s: FundamentalSnapshot) -> Optional[float]:
    capex = _ttm(_sorted_desc(s.cash_flow), "capex")
    rev = _ttm(_sorted_desc(s.income_quarterly), "revenue")
    if capex is None or not rev:
        return None
    return -1 * (abs(capex) / rev)


def fcf_consistency(s: FundamentalSnapshot) -> Optional[float]:
    """Count of positive-FCF years in the last 5."""
    annual = _sorted_desc(s.cash_flow_annual)[:5]
    fcf_values = []
    for period in annual:
        if period.free_cash_flow is not None:
            fcf_values.append(period.free_cash_flow)
        elif period.operating_cash_flow is not None and period.capex is not None:
            fcf_values.append(period.operating_cash_flow + period.capex)
    if not fcf_values:
        return None
    return float(sum(1 for v in fcf_values if v > 0))


# ---------------------------------------------------------------------------
# Pillar E -- Forward Outlook
# ---------------------------------------------------------------------------


def forward_revenue_growth(s: FundamentalSnapshot) -> Optional[float]:
    a = _sorted_desc(s.income_annual)
    if not a or not a[0].revenue:
        return None
    next_est = next((e for e in s.analyst_estimates if e.fiscal_year == a[0].period.fiscal_year + 1 and e.consensus_revenue), None)
    if next_est is None:
        return None
    return next_est.consensus_revenue / a[0].revenue - 1


def forward_eps_growth(s: FundamentalSnapshot) -> Optional[float]:
    a = _sorted_desc(s.income_annual)
    if not a or not a[0].diluted_eps:
        return None
    next_est = next((e for e in s.analyst_estimates if e.fiscal_year == a[0].period.fiscal_year + 1 and e.consensus_eps), None)
    if next_est is None:
        return None
    return next_est.consensus_eps / a[0].diluted_eps - 1


def estimate_revision_momentum(s: FundamentalSnapshot) -> Optional[float]:
    # Needs a time series of estimate snapshots 90 days apart. No provider
    # currently populates FundamentalSnapshot.estimate_revisions (see the
    # empty list left in providers/fmp.py and providers/yfinance_provider.py).
    return None


def earnings_surprise_history(s: FundamentalSnapshot) -> Optional[float]:
    surprises = [e.surprise_pct for e in s.earnings_surprises[:4] if e.surprise_pct is not None]
    if not surprises:
        return None
    return sum(surprises) / len(surprises)


# ---------------------------------------------------------------------------
# Context-only valuation metrics (reported, never scored)
# ---------------------------------------------------------------------------


def pe_ttm(s: FundamentalSnapshot) -> Optional[float]:
    ni = _ttm(_sorted_desc(s.income_quarterly), "net_income")
    if not ni or ni <= 0 or not s.market_cap:
        return None
    return s.market_cap / ni


def ev_ebitda(s: FundamentalSnapshot) -> Optional[float]:
    ebitda = _ttm(_sorted_desc(s.income_quarterly), "ebitda")
    bs = _sorted_desc(s.balance_sheet)
    if not ebitda or ebitda <= 0 or not bs or not s.market_cap:
        return None
    latest = bs[0]
    if latest.total_debt is None or latest.cash_and_equivalents is None:
        return None
    return (s.market_cap + latest.total_debt - latest.cash_and_equivalents) / ebitda


# ---------------------------------------------------------------------------
# Sector-substitute metrics (Financials)
# ---------------------------------------------------------------------------


def roe(s: FundamentalSnapshot) -> Optional[float]:
    ni = _ttm(_sorted_desc(s.income_quarterly), "net_income")
    bs = _sorted_desc(s.balance_sheet)
    if ni is None or not bs or not bs[0].total_equity:
        return None
    return ni / bs[0].total_equity


def capital_adequacy_proxy(s: FundamentalSnapshot) -> Optional[float]:
    bs = _sorted_desc(s.balance_sheet)
    if not bs or not bs[0].total_assets or bs[0].total_equity is None:
        return None
    return bs[0].total_equity / bs[0].total_assets


@dataclass(frozen=True)
class MetricDefinition:
    key: str
    pillar: str
    higher_is_better: bool
    compute: Callable[[FundamentalSnapshot], Optional[float]]
    scored: bool = True


METRIC_REGISTRY: list[MetricDefinition] = [
    MetricDefinition("rev_growth_ttm_yoy", PILLAR_GROWTH, True, rev_growth_ttm_yoy),
    MetricDefinition("rev_cagr_3y", PILLAR_GROWTH, True, rev_cagr_3y),
    MetricDefinition("growth_acceleration", PILLAR_GROWTH, True, growth_acceleration),
    MetricDefinition("growth_consistency", PILLAR_GROWTH, True, growth_consistency),
    MetricDefinition("net_margin_ttm", PILLAR_PROFITABILITY, True, net_margin_ttm),
    MetricDefinition("net_margin_trend", PILLAR_PROFITABILITY, True, net_margin_trend),
    MetricDefinition("operating_margin_ttm", PILLAR_PROFITABILITY, True, operating_margin_ttm),
    MetricDefinition("gross_margin_ttm", PILLAR_PROFITABILITY, True, gross_margin_ttm),
    MetricDefinition("roic", PILLAR_PROFITABILITY, True, roic),
    MetricDefinition("net_debt_ebitda", PILLAR_BALANCE_SHEET, False, net_debt_ebitda),
    MetricDefinition("debt_equity", PILLAR_BALANCE_SHEET, False, debt_equity),
    MetricDefinition("current_ratio", PILLAR_BALANCE_SHEET, True, current_ratio),
    MetricDefinition("interest_coverage", PILLAR_BALANCE_SHEET, True, interest_coverage),
    MetricDefinition("dilution", PILLAR_BALANCE_SHEET, True, dilution),
    MetricDefinition("fcf_margin", PILLAR_CASH_FLOW, True, fcf_margin),
    MetricDefinition("fcf_conversion", PILLAR_CASH_FLOW, True, fcf_conversion),
    MetricDefinition("ocf_cagr_3y", PILLAR_CASH_FLOW, True, ocf_cagr_3y),
    MetricDefinition("capex_intensity", PILLAR_CASH_FLOW, True, capex_intensity),
    MetricDefinition("fcf_consistency", PILLAR_CASH_FLOW, True, fcf_consistency),
    MetricDefinition("forward_revenue_growth", PILLAR_OUTLOOK, True, forward_revenue_growth),
    MetricDefinition("forward_eps_growth", PILLAR_OUTLOOK, True, forward_eps_growth),
    MetricDefinition("estimate_revision_momentum", PILLAR_OUTLOOK, True, estimate_revision_momentum),
    MetricDefinition("earnings_surprise_history", PILLAR_OUTLOOK, True, earnings_surprise_history),
]

CONTEXT_METRICS: list[MetricDefinition] = [
    MetricDefinition("pe_ttm", "context", True, pe_ttm, scored=False),
    MetricDefinition("ev_ebitda", "context", False, ev_ebitda, scored=False),
]

METRIC_REGISTRY_BY_KEY: dict[str, MetricDefinition] = {m.key: m for m in METRIC_REGISTRY}
