from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class PillarWeights(BaseModel):
    growth: float
    profitability: float
    balance_sheet: float
    cash_flow: float
    outlook: float

    @field_validator("*")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("pillar weight must be >= 0")
        return v

    def validate_sum(self) -> None:
        total = self.growth + self.profitability + self.balance_sheet + self.cash_flow + self.outlook
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"pillar weights must sum to 1.0, got {total:.4f}")


DEFAULT_PILLAR_WEIGHTS = dict(growth=0.20, profitability=0.25, balance_sheet=0.20, cash_flow=0.20, outlook=0.15)


class ScreenerSettings(BaseModel):
    fmp_api_key: Optional[str] = Field(default_factory=lambda: os.environ.get("FMP_API_KEY"))
    cache_dir: Path = Path(".cache/screener")
    cache_ttl_hours: int = 24
    max_concurrent_requests: int = 8
    winsorize_lower_pct: float = 0.05
    winsorize_upper_pct: float = 0.95
    min_data_coverage_pct: float = 60.0
    prerevenue_floor_usd: float = 1_000_000.0
    include_prerevenue: bool = False
    sector_neutral: bool = False
    pillar_weights: PillarWeights = Field(default_factory=lambda: PillarWeights(**DEFAULT_PILLAR_WEIGHTS))

    @classmethod
    def from_yaml(cls, path: Path | str, **overrides) -> "ScreenerSettings":
        data: dict = {}
        path = Path(path)
        if path.exists():
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
            if "pillar_weights" in raw:
                data["pillar_weights"] = PillarWeights(**raw["pillar_weights"])
            for key in (
                "winsorize_lower_pct",
                "winsorize_upper_pct",
                "min_data_coverage_pct",
                "prerevenue_floor_usd",
                "cache_ttl_hours",
                "max_concurrent_requests",
            ):
                if key in raw:
                    data[key] = raw[key]
        data.update(overrides)
        settings = cls(**data)
        settings.pillar_weights.validate_sum()
        return settings
