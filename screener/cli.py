from __future__ import annotations

import asyncio
import logging

import typer
from rich.console import Console
from rich.table import Table

from screener.cache.store import CacheStore
from screener.config import ScreenerSettings
from screener.pipeline.fetch import fetch_all_fundamentals
from screener.pipeline.holdings import resolve_holdings
from screener.providers.base import FundamentalsProvider, HoldingsProvider
from screener.providers.fmp import FMPFundamentalsProvider, FMPHoldingsProvider
from screener.providers.issuer_holdings import IssuerHoldingsProvider
from screener.providers.yfinance_provider import YFinanceFundamentalsProvider

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


@app.command()
def run(
    etf: str = typer.Argument(..., help="ETF ticker, e.g. QQQ"),
    top: int = typer.Option(25, help="Number of rows to display"),
    weights: str = typer.Option("config/weights.yaml", help="Path to pillar weights YAML"),
    no_cache: bool = typer.Option(False, "--no-cache"),
) -> None:
    """Resolve holdings + fetch fundamentals for an ETF universe.

    Scoring (Stage 4) is not wired up yet — this command demonstrates the
    holdings-resolution and fundamentals-fetch vertical slice.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = ScreenerSettings.from_yaml(weights)
    cache = CacheStore(settings.cache_dir, ttl_hours=settings.cache_ttl_hours)

    async def _run() -> None:
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

        table = Table(title=f"{etf.upper()} constituents — fundamentals fetched")
        table.add_column("Ticker")
        table.add_column("Name")
        table.add_column("Weight %", justify="right")
        table.add_column("Fundamentals", justify="center")
        for h in sorted(holdings, key=lambda x: -x.weight_pct)[:top]:
            status = "[green]OK[/green]" if h.ticker in fetch_result.snapshots else "[red]FAILED[/red]"
            table.add_row(h.ticker, h.name, f"{h.weight_pct:.2f}", status)
        console.print(table)

        if fetch_result.failures:
            console.print(f"\n[yellow]{len(fetch_result.failures)} tickers failed fundamentals fetch:[/yellow]")
            for failure in fetch_result.failures:
                console.print(f"  {failure.ticker}: {failure.reason}")

        console.print("\n[bold yellow]Scoring engine (Stage 4) not yet implemented.[/bold yellow]")

    asyncio.run(_run())


@app.command()
def explain(ticker: str, etf: str = typer.Option(..., help="ETF universe to compute percentiles within")) -> None:
    """Per-metric breakdown for one holding. Requires the scoring engine (not yet built)."""
    console.print("[red]Not implemented yet — scoring engine (Stage 4) has not been built.[/red]")
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
