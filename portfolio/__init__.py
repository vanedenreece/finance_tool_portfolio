"""Portfolio management toolkit.

Quick start:

    from portfolio import (
        fetch_prices, log_returns, ledoit_wolf_cov,
        min_variance, max_sharpe, efficient_frontier,
        View, black_litterman,
        backtest, make_min_variance_weight_fn,
        plot_frontier, plot_equity_curves,
    )
    from portfolio.assets import tickers, BUCKETS

    prices = fetch_prices(tickers(), start='2018-01-01')
    rets = log_returns(prices)
    cov = ledoit_wolf_cov(rets)
    mu = (rets.mean() * 252)
    res = min_variance(cov, mu)
    print(res.weights)
"""

from .assets import ASSETS, BUCKETS, tickers, bucket_of, assets_in_bucket
from .data import fetch_prices, log_returns, annualize_returns, annualize_cov, load_prices_csv
from .covariance import sample_cov, ledoit_wolf_cov, to_correlation
from .optimize import min_variance, max_sharpe, target_return, OptResult
from .frontier import efficient_frontier, frontier_summary
from .black_litterman import View, black_litterman, implied_equilibrium_returns
from .backtest import backtest, BacktestResult, make_min_variance_weight_fn, benchmark_60_40_weights
from .plotting import plot_frontier, plot_equity_curves, plot_weights_over_time

__all__ = [
    "ASSETS", "BUCKETS", "tickers", "bucket_of", "assets_in_bucket",
    "fetch_prices", "log_returns", "annualize_returns", "annualize_cov", "load_prices_csv",
    "sample_cov", "ledoit_wolf_cov", "to_correlation",
    "min_variance", "max_sharpe", "target_return", "OptResult",
    "efficient_frontier", "frontier_summary",
    "View", "black_litterman", "implied_equilibrium_returns",
    "backtest", "BacktestResult", "make_min_variance_weight_fn", "benchmark_60_40_weights",
    "plot_frontier", "plot_equity_curves", "plot_weights_over_time",
]
