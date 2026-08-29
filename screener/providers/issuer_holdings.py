from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Callable

import httpx
import pandas as pd
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from screener.models import AssetClass, Holding
from screener.providers.base import HoldingsProvider, is_retryable_http_error


@dataclass
class IssuerEndpoint:
    issuer: str
    url: str
    parser: Callable[[bytes], pd.DataFrame]


def _parse_ishares_csv(content: bytes) -> pd.DataFrame:
    text = content.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    header_idx = next((i for i, line in enumerate(lines) if line.startswith("Ticker")), None)
    if header_idx is None:
        raise ValueError("iShares CSV format has changed: no line starting with 'Ticker' found — registry needs a refresh")
    df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    return df.rename(
        columns={"Ticker": "ticker", "Name": "name", "Weight (%)": "weight_pct", "Sector": "sector", "Asset Class": "asset_class"}
    )


def _parse_ssga_xlsx(content: bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(content), skiprows=4)
    return df.rename(columns={"Ticker": "ticker", "Name": "name", "Weight": "weight_pct", "Sector": "sector"})


def _parse_invesco_csv(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content))
    return df.rename(columns={"Holding Ticker": "ticker", "Name": "name", "Weight": "weight_pct", "Sector": "sector"})


def _parse_vaneck_csv(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content))
    return df.rename(columns={"Ticker": "ticker", "Holding Name": "name", "Weightings": "weight_pct", "Sector": "sector"})


def _parse_vanguard_json(content: bytes) -> pd.DataFrame:
    data = json.loads(content)
    rows = data.get("holdings", data) if isinstance(data, dict) else data
    df = pd.DataFrame(rows)
    return df.rename(columns={"ticker": "ticker", "shortName": "name", "percentWeight": "weight_pct", "sector": "sector"})


# These are the issuers' documented/observed public download endpoints as of the
# last time this registry was verified. Issuer sites restructure these paths
# periodically (iShares in particular embeds a numeric fund id per product) —
# treat a 404 here as "registry needs a refresh," not "issuer stopped publishing."
ISSUER_REGISTRY: dict[str, IssuerEndpoint] = {
    "SPY": IssuerEndpoint(
        issuer="ssga",
        url="https://www.ssga.com/us/en/individual/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx",
        parser=_parse_ssga_xlsx,
    ),
    "XLF": IssuerEndpoint(
        issuer="ssga",
        url="https://www.ssga.com/us/en/individual/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xlf.xlsx",
        parser=_parse_ssga_xlsx,
    ),
    "QQQ": IssuerEndpoint(
        issuer="invesco",
        url="https://www.invesco.com/us/financial-products/etfs/holdings/main/holdings/0?action=download&ticker=QQQ",
        parser=_parse_invesco_csv,
    ),
    "SMH": IssuerEndpoint(
        issuer="vaneck",
        url="https://www.vaneck.com/us/en/investments/semiconductor-etf-smh/holdings/export/",
        parser=_parse_vaneck_csv,
    ),
    "VOO": IssuerEndpoint(
        issuer="vanguard",
        url="https://investor.vanguard.com/investment-products/etfs/profile/api/voo/portfolio-holding/stock",
        parser=_parse_vanguard_json,
    ),
}


class IssuerHoldingsProvider(HoldingsProvider):
    name = "issuer"

    def __init__(self, client: httpx.AsyncClient | None = None, timeout: float = 15.0):
        self._client = client
        self._timeout = timeout

    async def get_holdings(self, etf_ticker: str) -> list[Holding]:
        endpoint = ISSUER_REGISTRY.get(etf_ticker.upper())
        if endpoint is None:
            raise KeyError(f"No issuer endpoint registered for {etf_ticker!r}")

        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=self._timeout, follow_redirects=True)
        try:
            resp = await self._get_with_retry(client, endpoint.url)
            df = endpoint.parser(resp.content)
        finally:
            if owns_client:
                await client.aclose()

        return self._rows_to_holdings(df, endpoint.issuer)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=10),
        retry=retry_if_exception(is_retryable_http_error),
    )
    async def _get_with_retry(self, client: httpx.AsyncClient, url: str) -> httpx.Response:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp

    @staticmethod
    def _rows_to_holdings(df: pd.DataFrame, issuer: str) -> list[Holding]:
        holdings = []
        for _, row in df.iterrows():
            ticker = str(row.get("ticker", "")).strip()
            if not ticker or ticker.lower() in ("nan", "none", "-", "cash"):
                continue
            try:
                weight_pct = float(str(row.get("weight_pct")).replace("%", "").replace(",", ""))
            except (TypeError, ValueError):
                continue
            asset_class_raw = str(row.get("asset_class", "Equity")).strip().lower()
            asset_class = AssetClass.EQUITY if asset_class_raw in ("equity", "common stock", "nan", "") else AssetClass.OTHER
            sector = row.get("sector")
            holdings.append(
                Holding(
                    ticker=ticker,
                    name=str(row.get("name", ticker)).strip(),
                    weight_pct=weight_pct,
                    sector=str(sector) if pd.notna(sector) else None,
                    asset_class=asset_class,
                    source=issuer,
                )
            )
        return holdings
