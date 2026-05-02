"""Historical price data via yfinance with on-disk caching.

Cache lives in ./cache/ as parquet files. Delete the folder to force refresh.
"""

from __future__ import annotations

import hashlib
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def _cache_key(tickers: list[str], start: str, end: str) -> Path:
    payload = "|".join(sorted(tickers)) + f"@{start}>{end}"
    h = hashlib.md5(payload.encode()).hexdigest()[:12]
    return CACHE_DIR / f"prices_{h}.parquet"


def fetch_prices(
    tickers: list[str],
    start: str = "2015-01-01",
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch adjusted close prices.

    Returns a DataFrame indexed by date with one column per ticker.
    Drops any ticker that has no data; logs which were dropped.
    """
    end = end or date.today().isoformat()
    cache_path = _cache_key(tickers, start, end)

    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        # If the cache is < 7 days stale, accept it
        cache_age = date.today() - date.fromtimestamp(cache_path.stat().st_mtime)
        if cache_age < timedelta(days=7):
            return df

    # Lazy import so the module loads even if yfinance isn't installed yet
    import yfinance as yf

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,    # adjusts for splits/dividends
        progress=False,
        group_by="ticker",
    )

    # yfinance returns a multi-index frame for >1 ticker, single-level for 1 ticker
    if len(tickers) == 1:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        prices = pd.DataFrame({tk: raw[tk]["Close"] for tk in tickers if tk in raw.columns.get_level_values(0)})

    prices = prices.dropna(how="all")
    missing = [tk for tk in tickers if tk not in prices.columns or prices[tk].dropna().empty]
    if missing:
        print(f"[data] Warning: no data for {missing} — dropped from universe.")
        prices = prices.drop(columns=[c for c in missing if c in prices.columns], errors="ignore")

    prices.to_parquet(cache_path)
    return prices


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns. NaN rows are dropped."""
    import numpy as np
    return np.log(prices / prices.shift(1)).dropna(how="all")


def annualize_returns(daily_returns: pd.DataFrame, periods_per_year: int = 252) -> pd.Series:
    """Annualized mean return per asset (arithmetic mean * periods).

    For log returns, multiplying by 252 gives annualized log return; we exponentiate
    to convert to simple-return space which is what the optimizer expects.
    """
    import numpy as np
    mu_log = daily_returns.mean() * periods_per_year
    return np.exp(mu_log) - 1.0


def annualize_cov(daily_returns: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    """Annualized sample covariance. See covariance.py for shrinkage estimators."""
    return daily_returns.cov() * periods_per_year


def load_prices_csv(path: str) -> pd.DataFrame:
    """Fallback: load prices from a CSV with a 'Date' column and one column per ticker.

    Useful if yfinance is unavailable or you want to use your own data source
    (Bloomberg export, FRED, etc.).
    """
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    return df
