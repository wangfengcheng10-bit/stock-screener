from __future__ import annotations

import asyncio
import logging
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from screener.cache.store import CacheStore
from screener.config import ScreenerSettings
from screener.pipeline.fetch import fetch_all_fundamentals
from screener.pipeline.holdings import resolve_holdings
from screener.pipeline.metrics import METRIC_REGISTRY_BY_KEY
from screener.pipeline.scoring import build_metric_set, build_universe_data, score_universe
from screener.pipeline.sectors import SUBSTITUTE_METRICS_BY_GROUP
from screener.providers.base import FundamentalsProvider, HoldingsProvider
from screener.providers.fmp import FMPFundamentalsProvider, FMPHoldingsProvider
from screener.providers.issuer_holdings import IssuerHoldingsProvider
from screener.providers.yfinance_provider import YFinanceFundamentalsProvider
from screener.report.export import export, scorecards_to_dataframe
from screener.report.table import render_scorecard_table

app = typer.Typer(add_completion=False)
console = Console()


def _build_holdings_cascade(settings: ScreenerSettings) -> list[HoldingsProvider]:
    cascade: list[HoldingsProvider] = [IssuerHoldingsProvider()]
    if settings.fmp_api_key:
        cascade.append(FMPHoldingsProvider(api_key=settings.fmp_api_key))
    return cascade


def _build_fundamentals_provider(settings: ScreenerSettings) -> FundamentalsProvider:
    if settings.fmp_api_key:
        return FMPFundamentalsProvider(api_key=settings.fmp_api_key)
    console.print("[yellow]No FMP_API_KEY set — falling back to yfinance (thinner fundamentals coverage).[/yellow]")
    return YFinanceFundamentalsProvider()


def _metric_defs_by_key() -> dict:
    defs = dict(METRIC_REGISTRY_BY_KEY)
    for group_defs in SUBSTITUTE_METRICS_BY_GROUP.values():
        for m in group_defs:
            defs.setdefault(m.key, m)
    return defs


async def _resolve_and_fetch(
    etf: str,
    settings: ScreenerSettings,
    cache: CacheStore,
    no_cache: bool,
    min_market_cap: Optional[float],
):
    cascade = _build_holdings_cascade(settings)
    holdings, report = await resolve_holdings(etf.upper(), cascade)

    console.print(f"\n[bold]Holdings resolution — {etf.upper()}[/bold]")
    console.print(f"  Source used:        {report.source_used}")
    console.print(f"  Holdings found:      {report.holdings_found}")
    console.print(f"  Equities retained:   {report.equities_retained}")
    console.print(f"  Dropped non-equity:  {report.dropped_non_equity}")
    console.print(f"  Dropped duplicates:  {report.dropped_duplicate_class}")
    console.print(f"  Weight coverage:     {report.weight_coverage_pct}%\n")

    fundamentals_provider = _build_fundamentals_provider(settings)
    fetch_result = await fetch_all_fundamentals(
        [h.ticker for h in holdings],
        fundamentals_provider,
        cache=cache,
        max_concurrent=settings.max_concurrent_requests,
        use_cache=not no_cache,
    )

    holdings_by_ticker = {h.ticker: h for h in holdings}
    snapshots = fetch_result.snapshots
    if min_market_cap is not None:
        snapshots = {t: s for t, s in snapshots.items() if s.market_cap is not None and s.market_cap >= min_market_cap}

    return holdings_by_ticker, snapshots, fetch_result, report


@app.command()
def run(
    etf: str = typer.Argument(..., help="ETF ticker, e.g. QQQ"),
    top: int = typer.Option(25, help="Number of rows to display"),
    weights: str = typer.Option("config/weights.yaml", help="Path to pillar weights YAML"),
    sector_neutral: bool = typer.Option(False, "--sector-neutral", help="Percentile-rank within each GICS sector instead of the whole universe"),
    include_prerevenue: bool = typer.Option(False, "--include-prerevenue", help="Score pre-revenue names instead of listing them separately"),
    min_market_cap: Optional[float] = typer.Option(None, help="Drop holdings below this market cap (USD) before scoring"),
    no_cache: bool = typer.Option(False, "--no-cache"),
    out: Optional[str] = typer.Option(None, help="Export path: .csv, .parquet, or .xlsx"),
) -> None:
    """Resolve holdings, fetch fundamentals, score the universe, and print a ranked table."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = ScreenerSettings.from_yaml(weights, include_prerevenue=include_prerevenue, sector_neutral=sector_neutral)
    cache = CacheStore(settings.cache_dir, ttl_hours=settings.cache_ttl_hours)

    async def _run() -> None:
        holdings_by_ticker, snapshots, fetch_result, _ = await _resolve_and_fetch(etf, settings, cache, no_cache, min_market_cap)

        universe = build_universe_data(snapshots, include_prerevenue=settings.include_prerevenue)
        sector_by_ticker = {t: s.sector for t, s in snapshots.items()} if settings.sector_neutral else None
        pillar_weights = settings.pillar_weights.model_dump()

        scorecards = score_universe(
            universe,
            holdings_by_ticker,
            pillar_weights,
            settings.winsorize_lower_pct,
            settings.winsorize_upper_pct,
            settings.min_data_coverage_pct,
            sector_by_ticker,
        )

        console.print(
            f"[bold]Run summary[/bold]  ETF={etf.upper()}  holdings_resolved={len(holdings_by_ticker)}  "
            f"scored={len(scorecards)}  fetch_failed={len(fetch_result.failures)}  "
            f"unscored_prerevenue={len(universe.unscored_prerevenue)}\n"
        )

        ranked = sorted(scorecards.values(), key=lambda sc: (sc.low_confidence, -sc.composite_score))[:top]
        render_scorecard_table(console, ranked, title=f"{etf.upper()} — ranked by composite score")

        if universe.unscored_prerevenue:
            console.print(f"\n[yellow]Unscored (pre-revenue): {', '.join(universe.unscored_prerevenue)}[/yellow]")
        if fetch_result.failures:
            console.print(f"[yellow]Unscored (fetch failed): {', '.join(f.ticker for f in fetch_result.failures)}[/yellow]")

        if out:
            df = scorecards_to_dataframe(list(scorecards.values()), universe.raw_df)
            export(df, out)
            console.print(f"\nExported {len(df)} rows to {out}")

    asyncio.run(_run())


@app.command()
def explain(
    ticker: str,
    etf: str = typer.Option(..., help="ETF universe to compute percentiles within"),
    weights: str = typer.Option("config/weights.yaml", help="Path to pillar weights YAML"),
    no_cache: bool = typer.Option(False, "--no-cache"),
) -> None:
    """Per-metric breakdown for one holding: raw value, percentile, and pillar contribution."""
    logging.basicConfig(level=logging.WARNING)
    settings = ScreenerSettings.from_yaml(weights)
    cache = CacheStore(settings.cache_dir, ttl_hours=settings.cache_ttl_hours)
    ticker = ticker.upper()

    async def _run() -> None:
        holdings_by_ticker, snapshots, _, _ = await _resolve_and_fetch(etf, settings, cache, no_cache, None)
        if ticker not in snapshots:
            console.print(f"[red]{ticker} not found in {etf.upper()}'s scored universe (missing data or not a constituent).[/red]")
            raise typer.Exit(1)

        from screener.pipeline.scoring import compute_percentiles

        universe = build_universe_data(snapshots, include_prerevenue=True)
        pct_df = compute_percentiles(universe.raw_df, settings.winsorize_lower_pct, settings.winsorize_upper_pct)
        metric_set = build_metric_set(ticker, universe, pct_df, _metric_defs_by_key())

        table = Table(title=f"{ticker} — metric breakdown within {etf.upper()}")
        table.add_column("Metric")
        table.add_column("Pillar")
        table.add_column("Raw", justify="right")
        table.add_column("Percentile", justify="right")
        table.add_column("Missing", justify="center")
        for key, mv in metric_set.metrics.items():
            pillar = _metric_defs_by_key()[key].pillar
            raw_str = f"{mv.raw_value:.4f}" if mv.raw_value is not None else "-"
            table.add_row(key, pillar, raw_str, f"{mv.percentile:.1f}", "yes" if mv.is_missing else "")
        console.print(table)

    asyncio.run(_run())


if __name__ == "__main__":
    app()
