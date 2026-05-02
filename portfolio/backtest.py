"""Walk-forward backtester.

Methodology:
- Look-back window of N months to estimate covariance and (optionally) returns
- Rebalance at fixed intervals (default monthly)
- Apply transaction costs as a fraction of turnover
- Compare against a benchmark portfolio (default: 60/40 SPY/IEF)

This is a simple research backtester, not a production risk system. It does NOT model:
- Slippage beyond a flat cost
- Bid-ask spreads
- Borrowing costs (we're long-only anyway)
- Tax drag
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .covariance import ledoit_wolf_cov
from .data import log_returns
from .optimize import min_variance


@dataclass
class BacktestResult:
    equity: pd.Series              # cumulative return index, starting at 1.0
    weights: pd.DataFrame          # weights at each rebalance date
    returns: pd.Series             # per-period (daily) returns net of costs
    turnover: pd.Series            # turnover at each rebalance
    summary: dict                  # CAGR, vol, Sharpe, max drawdown


def _summary_stats(returns: pd.Series, periods_per_year: int = 252) -> dict:
    if len(returns) == 0:
        return {"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "MaxDD": np.nan}
    growth = (1 + returns).prod()
    years = len(returns) / periods_per_year
    cagr = growth ** (1 / years) - 1 if years > 0 else np.nan
    vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = (returns.mean() * periods_per_year) / vol if vol > 0 else np.nan
    equity = (1 + returns).cumprod()
    drawdown = (equity / equity.cummax() - 1).min()
    return {"CAGR": float(cagr), "Vol": float(vol), "Sharpe": float(sharpe), "MaxDD": float(drawdown)}


def backtest(
    prices: pd.DataFrame,
    weight_fn: Callable[[pd.DataFrame], pd.Series],
    rebalance_freq: str = "ME",     # month-end
    lookback_days: int = 504,       # ~2 years
    transaction_cost_bps: float = 10.0,  # 10bps per dollar traded
    start: Optional[str] = None,
) -> BacktestResult:
    """Run a walk-forward backtest.

    Parameters
    ----------
    prices : DataFrame of adjusted prices, daily
    weight_fn : callable taking a price-history DataFrame and returning a weight Series.
                Called fresh at each rebalance with the trailing lookback window.
    rebalance_freq : pandas offset alias ('ME' = month end, 'QE' = quarter end, 'YE' = year end)
    lookback_days : trading days of history used to estimate weights
    transaction_cost_bps : one-way cost in basis points (10 = 0.10%)
    start : optional ISO date; backtest begins on or after this date and after enough lookback
    """
    daily_ret = prices.pct_change().dropna(how="all")
    if start:
        daily_ret = daily_ret.loc[start:]

    # Find rebalance dates: month-ends in the price index, after lookback warmup
    if lookback_days >= len(prices):
        raise ValueError(f"lookback_days ({lookback_days}) >= price history ({len(prices)})")
    warmup_date = prices.index[lookback_days]
    candidate_dates = pd.date_range(prices.index[0], prices.index[-1], freq=rebalance_freq)
    rebal_dates = [d for d in candidate_dates if d >= warmup_date and d <= prices.index[-1]]
    # Snap each rebalance date to the next available trading day
    rebal_dates = [prices.index[prices.index.get_indexer([d], method="bfill")[0]]
                   for d in rebal_dates if prices.index.get_indexer([d], method="bfill")[0] != -1]
    rebal_dates = sorted(set(rebal_dates))

    if not rebal_dates:
        raise ValueError("No rebalance dates available; lookback may be too long.")

    weights_history: dict[pd.Timestamp, pd.Series] = {}
    current_weights = pd.Series(0.0, index=prices.columns)
    portfolio_returns = []
    turnover_log: dict[pd.Timestamp, float] = {}
    cost_rate = transaction_cost_bps / 1e4

    # Iterate day by day; rebalance when we hit a rebalance date
    rebal_set = set(rebal_dates)
    for dt in daily_ret.index:
        if dt in rebal_set:
            window_end_idx = prices.index.get_loc(dt)
            window_start_idx = max(0, window_end_idx - lookback_days)
            window = prices.iloc[window_start_idx:window_end_idx]
            try:
                new_weights = weight_fn(window).reindex(prices.columns).fillna(0.0)
            except Exception as e:
                # If optimization fails, hold previous weights
                print(f"[backtest] {dt.date()}: weight_fn failed ({e}); holding.")
                new_weights = current_weights.copy()
            turnover = float((new_weights - current_weights).abs().sum())
            cost = turnover * cost_rate
            turnover_log[dt] = turnover
            weights_history[dt] = new_weights.copy()
            # Charge cost on rebalance day
            day_ret = float(current_weights @ daily_ret.loc[dt].fillna(0.0)) - cost
            current_weights = new_weights
        else:
            day_ret = float(current_weights @ daily_ret.loc[dt].fillna(0.0))
        portfolio_returns.append((dt, day_ret))

    ret_series = pd.Series(dict(portfolio_returns)).sort_index()
    equity = (1 + ret_series).cumprod()
    weights_df = pd.DataFrame(weights_history).T
    turnover_series = pd.Series(turnover_log).sort_index()
    summary = _summary_stats(ret_series)

    return BacktestResult(
        equity=equity,
        weights=weights_df,
        returns=ret_series,
        turnover=turnover_series,
        summary=summary,
    )


# ---------- Convenience: a default min-variance weight function ----------

def make_min_variance_weight_fn(buckets=None, use_shrinkage: bool = True):
    """Return a weight_fn suitable for backtest()."""
    def fn(price_window: pd.DataFrame) -> pd.Series:
        rets = log_returns(price_window).dropna()
        if use_shrinkage:
            cov = ledoit_wolf_cov(rets)
        else:
            cov = rets.cov() * 252
        result = min_variance(cov, buckets=buckets)
        return result.weights
    return fn


def benchmark_60_40_weights(prices: pd.DataFrame) -> pd.Series:
    """A static 60% SPY / 40% IEF benchmark (or whatever stock/bond proxies are present)."""
    w = pd.Series(0.0, index=prices.columns)
    if "SPY" in prices.columns:
        w["SPY"] = 0.60
    if "IEF" in prices.columns:
        w["IEF"] = 0.40
    if w.sum() == 0:
        raise ValueError("Neither SPY nor IEF in prices — can't build 60/40 benchmark.")
    return w / w.sum()
