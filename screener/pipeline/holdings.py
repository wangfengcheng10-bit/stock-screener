from __future__ import annotations

import logging
from dataclasses import dataclass, field

from screener.models import AssetClass, Holding
from screener.providers.base import HoldingsProvider

logger = logging.getLogger(__name__)

# Share-class / listing normalization: map raw issuer tickers to the fundamentals
# provider's convention (dot vs dash class suffixes).
TICKER_NORMALIZATION: dict[str, str] = {
    "BRK.B": "BRK-B",
    "BRK.A": "BRK-A",
    "BF.B": "BF-B",
    "BF.A": "BF-A",
}

# Multi-class siblings: when both appear in the same universe, keep the higher-weight
# ticker and drop the other (recorded, not silently discarded).
MULTI_CLASS_SIBLINGS: list[tuple[str, str]] = [
    ("GOOG", "GOOGL"),
    ("FOX", "FOXA"),
    ("NWS", "NWSA"),
]


@dataclass
class HoldingsResolutionReport:
    etf_ticker: str
    source_used: str
    holdings_found: int
    equities_retained: int
    dropped_non_equity: int
    dropped_duplicate_class: list[tuple[str, str]] = field(default_factory=list)
    weight_coverage_pct: float = 0.0


async def resolve_holdings(etf_ticker: str, cascade: list[HoldingsProvider]) -> tuple[list[Holding], HoldingsResolutionReport]:
    """Try each provider in order; use the first that returns a non-empty list.
    Fails loudly (raises) if the whole cascade is exhausted — never silently
    falls back to a partial universe."""
    raw_holdings: list[Holding] | None = None
    source_used: str | None = None
    errors: list[str] = []

    for provider in cascade:
        try:
            holdings = await provider.get_holdings(etf_ticker)
        except Exception as exc:
            errors.append(f"{provider.name}: {exc}")
            logger.warning("Holdings provider %s failed for %s: %s", provider.name, etf_ticker, exc)
            continue
        if holdings:
            raw_holdings, source_used = holdings, provider.name
            logger.info("Resolved %s holdings for %s via %s", len(holdings), etf_ticker, provider.name)
            break
        errors.append(f"{provider.name}: returned no holdings")

    if raw_holdings is None:
        raise RuntimeError(
            f"Could not resolve holdings for {etf_ticker} from any provider in the cascade. Attempts: {'; '.join(errors)}"
        )

    holdings_found = len(raw_holdings)
    equities = [h for h in raw_holdings if h.asset_class == AssetClass.EQUITY]
    dropped_non_equity = holdings_found - len(equities)

    for h in equities:
        h.ticker = TICKER_NORMALIZATION.get(h.ticker, h.ticker)

    by_ticker: dict[str, Holding] = {h.ticker: h for h in equities}

    dropped_duplicates: list[tuple[str, str]] = []
    for primary, sibling in MULTI_CLASS_SIBLINGS:
        if primary in by_ticker and sibling in by_ticker:
            keep, drop = (primary, sibling) if by_ticker[primary].weight_pct >= by_ticker[sibling].weight_pct else (sibling, primary)
            del by_ticker[drop]
            dropped_duplicates.append((keep, drop))

    final_holdings = list(by_ticker.values())
    weight_coverage = sum(h.weight_pct for h in final_holdings)

    report = HoldingsResolutionReport(
        etf_ticker=etf_ticker,
        source_used=source_used,
        holdings_found=holdings_found,
        equities_retained=len(final_holdings),
        dropped_non_equity=dropped_non_equity,
        dropped_duplicate_class=dropped_duplicates,
        weight_coverage_pct=round(weight_coverage, 2),
    )
    return final_holdings, report
