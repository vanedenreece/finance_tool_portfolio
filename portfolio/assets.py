"""Asset universe and bucket constraints.

Single source of truth for tickers, bucket assignments, and policy weights.
Edit ASSETS to change the universe. Edit BUCKETS to change the policy split.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Asset:
    ticker: str
    name: str
    bucket: str


# Real ETFs covering the three buckets.
# Replace with your preferred tickers (UCITS equivalents for non-US investors, etc.).
ASSETS: list[Asset] = [
    # Fixed Income (50%)
    Asset("IEF",  "iShares 7-10Y Treasury",        "Fixed Income"),
    Asset("LQD",  "iShares IG Corporate",          "Fixed Income"),
    Asset("HYG",  "iShares High Yield Corporate",  "Fixed Income"),
    Asset("EMB",  "iShares EM Sovereign USD",      "Fixed Income"),
    # Equities (40%)
    Asset("SPY",  "SPDR S&P 500",                  "Equities"),
    Asset("IWM",  "iShares Russell 2000",          "Equities"),
    Asset("EFA",  "iShares MSCI EAFE",             "Equities"),
    Asset("EEM",  "iShares MSCI Emerging Markets", "Equities"),
    # Alternatives (10%)
    Asset("GLD",  "SPDR Gold Trust",               "Alternatives"),
    Asset("VNQ",  "Vanguard Real Estate",          "Alternatives"),
]

# Bucket policy: ticker bucket -> target weight. Must sum to 1.0.
BUCKETS: dict[str, float] = {
    "Fixed Income": 0.50,
    "Equities":     0.40,
    "Alternatives": 0.10,
}


def tickers() -> list[str]:
    return [a.ticker for a in ASSETS]


def bucket_of(ticker: str) -> str:
    for a in ASSETS:
        if a.ticker == ticker:
            return a.bucket
    raise KeyError(f"Unknown ticker: {ticker}")


def assets_in_bucket(bucket: str) -> list[Asset]:
    return [a for a in ASSETS if a.bucket == bucket]


def validate() -> None:
    """Sanity-check the configuration. Raises ValueError on problems."""
    if abs(sum(BUCKETS.values()) - 1.0) > 1e-9:
        raise ValueError(f"BUCKETS must sum to 1.0, got {sum(BUCKETS.values())}")
    for a in ASSETS:
        if a.bucket not in BUCKETS:
            raise ValueError(f"Asset {a.ticker} has unknown bucket {a.bucket!r}")
    for b in BUCKETS:
        if not assets_in_bucket(b):
            raise ValueError(f"Bucket {b!r} has no assets assigned")


validate()
