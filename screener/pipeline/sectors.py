from __future__ import annotations

from enum import Enum

from screener.pipeline.metrics import (
    PILLAR_BALANCE_SHEET,
    PILLAR_PROFITABILITY,
    METRIC_REGISTRY,
    METRIC_REGISTRY_BY_KEY,
    MetricDefinition,
    capital_adequacy_proxy,
    roe,
)


class SectorGroup(str, Enum):
    FINANCIALS = "Financials"
    REITS = "Real Estate"
    STANDARD = "Standard"


FINANCIALS_SECTOR_NAMES = {"Financial Services", "Financials", "Banks", "Insurance", "Diversified Financials"}
REIT_SECTOR_NAMES = {"Real Estate", "REIT", "Equity Real Estate Investment Trusts (REITs)"}


def classify_sector_group(sector: str | None) -> SectorGroup:
    if sector in FINANCIALS_SECTOR_NAMES:
        return SectorGroup.FINANCIALS
    if sector in REIT_SECTOR_NAMES:
        return SectorGroup.REITS
    return SectorGroup.STANDARD


# Metrics excluded per sector group because they're structurally meaningless:
# banks/insurers don't report a gross margin or capex the way an industrial
# company does, and Debt/Equity or a current ratio means something different
# for a balance sheet that IS the business.
EXCLUDED_METRICS_BY_GROUP: dict[SectorGroup, set[str]] = {
    SectorGroup.FINANCIALS: {
        "gross_margin_ttm",
        "operating_margin_ttm",
        "roic",
        "net_debt_ebitda",
        "debt_equity",
        "current_ratio",
        "interest_coverage",
        "fcf_margin",
        "fcf_conversion",
        "ocf_cagr_3y",
        "capex_intensity",
        "fcf_consistency",
    },
    # A true REIT treatment substitutes FFO/AFFO for net income in margin
    # metrics; neither FMP's free tier nor yfinance expose an FFO line item,
    # so those metrics are excluded rather than computed against the wrong
    # earnings figure. Net-debt/EBITDA and cash-flow metrics stay -- they
    # don't depend on net income and are valid for REITs.
    SectorGroup.REITS: {"net_margin_ttm", "net_margin_trend"},
    SectorGroup.STANDARD: set(),
}

SUBSTITUTE_METRICS_BY_GROUP: dict[SectorGroup, list[MetricDefinition]] = {
    SectorGroup.FINANCIALS: [
        MetricDefinition("roe", PILLAR_PROFITABILITY, True, roe),
        MetricDefinition("capital_adequacy_proxy", PILLAR_BALANCE_SHEET, True, capital_adequacy_proxy),
    ],
    SectorGroup.REITS: [],
    SectorGroup.STANDARD: [],
}


def applicable_metrics(group: SectorGroup) -> list[MetricDefinition]:
    excluded = EXCLUDED_METRICS_BY_GROUP.get(group, set())
    base = [m for m in METRIC_REGISTRY if m.key not in excluded]
    return base + SUBSTITUTE_METRICS_BY_GROUP.get(group, [])


def all_registered_metric_keys() -> set[str]:
    keys = set(METRIC_REGISTRY_BY_KEY.keys())
    for substitutes in SUBSTITUTE_METRICS_BY_GROUP.values():
        keys.update(m.key for m in substitutes)
    return keys
