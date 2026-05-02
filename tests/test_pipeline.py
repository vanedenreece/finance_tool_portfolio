"""End-to-end smoke test using synthetic price data.

If this passes, the package is internally consistent. yfinance is tested separately.
"""

import sys
from pathlib import Path


# adding the Path to the root directory. Allows for the tests to be run whatever the project root folder is names
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#Initialize test counter
TEST_COUNT = 0

import numpy as np
import pandas as pd
from portfolio import (
    tickers, BUCKETS,
    log_returns, ledoit_wolf_cov, sample_cov,
    min_variance, max_sharpe, efficient_frontier, frontier_summary,
    View, black_litterman, implied_equilibrium_returns,
    backtest, make_min_variance_weight_fn, benchmark_60_40_weights,
)

print("=" * 60)
print("Synthetic data pipeline test")
print("=" * 60)

# Synthesize 5 years of daily prices with realistic vol/correlation
np.random.seed(42)
tk_list = tickers()
n_assets = len(tk_list)
n_days = 252 * 5


# Annual vols and a plausible correlation pattern
true_vols = np.array([0.06, 0.075, 0.11, 0.13, 0.16, 0.22, 0.18, 0.24, 0.18, 0.20])
true_corr = np.eye(n_assets) * 0.3 + 0.15  # baseline weak positive correlation
# Boost correlation within Equities and within FI
fi_idx = [0, 1, 2, 3]
eq_idx = [4, 5, 6, 7]
for i in fi_idx:
    for j in fi_idx:
        if i != j: true_corr[i, j] = 0.55
for i in eq_idx:
    for j in eq_idx:
        if i != j: true_corr[i, j] = 0.75
np.fill_diagonal(true_corr, 1.0)

true_cov_daily = (np.outer(true_vols, true_vols) * true_corr) / 252
true_mu_daily = np.array([0.04, 0.055, 0.075, 0.07, 0.09, 0.105, 0.08, 0.11, 0.05, 0.085]) / 252

daily_rets = np.random.multivariate_normal(true_mu_daily, true_cov_daily, size=n_days)
dates = pd.bdate_range(end="2024-12-31", periods=n_days)
prices = pd.DataFrame(100 * np.exp(np.cumsum(daily_rets, axis=0)),
                      index=dates, columns=tk_list)

print(f"\nSynthetic prices: {prices.shape}, range {prices.index[0].date()} to {prices.index[-1].date()}")

# 1) Returns and covariance
rets = log_returns(prices)
cov = ledoit_wolf_cov(rets)
mu = (rets.mean() * 252)
mu = np.exp(mu) - 1.0  # convert to simple-return space
print(f"\nEstimated annual mu range: {mu.min():.2%} to {mu.max():.2%}")
print(f"Diag of cov (vols): {np.sqrt(np.diag(cov.values)).round(3)}")
TEST_COUNT+=1

# 2) Min-variance with bucket constraints
mvp = min_variance(cov, mu)
print(f"\nMin-Variance: {mvp}")
print("  weights:")
for tk, w in mvp.weights.items():
    print(f"    {tk}: {w:.2%}")
# Validate buckets
from portfolio.assets import bucket_of
bucket_sums = {b: 0.0 for b in BUCKETS}
for tk, w in mvp.weights.items():
    bucket_sums[bucket_of(tk)] += w
print(f"  bucket sums: {bucket_sums}")
for b, target in BUCKETS.items():
    assert abs(bucket_sums[b] - target) < 1e-4, f"Bucket {b}: {bucket_sums[b]} != {target}"
print("  ✓ Bucket constraints satisfied")
TEST_COUNT+=1


# 3) Max-Sharpe
msr = max_sharpe(cov, mu)
print(f"\nMax-Sharpe: {msr}")
bucket_sums = {b: 0.0 for b in BUCKETS}
for tk, w in msr.weights.items():
    bucket_sums[bucket_of(tk)] += w
for b, target in BUCKETS.items():
    assert abs(bucket_sums[b] - target) < 1e-4
print("  ✓ Bucket constraints satisfied")
TEST_COUNT+=1

# 4) Efficient frontier
ef = efficient_frontier(cov, mu, n_points=15)
print(f"\nEfficient frontier: {len(ef)} points")
print(f"  vol range: {ef['vol'].min():.2%} to {ef['vol'].max():.2%}")
print(f"  return range: {ef['achieved_return'].min():.2%} to {ef['achieved_return'].max():.2%}")
# Frontier should be monotonically increasing in vol as return increases
assert ef['vol'].iloc[0] <= ef['vol'].iloc[-1], "Frontier not monotonic"
print("  ✓ Frontier is well-formed")
TEST_COUNT+=1

# 5) Frontier summary
summary = frontier_summary(cov, mu)
print(f"\nFrontier summary:")
print(summary.round(4).to_string())
TEST_COUNT+=1

# 6) Black-Litterman
# Pretend market weights are equal-weight, define a couple of views
mkt_w = pd.Series(1.0 / n_assets, index=tk_list)
views = [
    View(assets={"SPY": 1.0}, return_=0.10, confidence=0.6),
    View(assets={"EEM": 1.0, "SPY": -1.0}, return_=0.02, confidence=0.4),
]
mu_bl, cov_bl = black_litterman(cov, mkt_w, views)
print(f"\nBlack-Litterman posterior mu range: {mu_bl.min():.2%} to {mu_bl.max():.2%}")
res_bl = min_variance(cov_bl, mu_bl)
print(f"BL Min-Variance: {res_bl}")
bucket_sums = {b: 0.0 for b in BUCKETS}
for tk, w in res_bl.weights.items():
    bucket_sums[bucket_of(tk)] += w
for b, target in BUCKETS.items():
    assert abs(bucket_sums[b] - target) < 1e-4
print("  ✓ BL portfolio respects bucket constraints")
TEST_COUNT+=1

# 7) Backtest — short window for speed
print("\nRunning backtest (this takes a moment)...")
weight_fn = make_min_variance_weight_fn()
bt = backtest(prices, weight_fn, lookback_days=252, rebalance_freq="QE")
print(f"  Equity curve: {len(bt.equity)} days")
print(f"  Final equity: {bt.equity.iloc[-1]:.4f}")
print(f"  Rebalances:   {len(bt.weights)}")
print(f"  Summary: {bt.summary}")
TEST_COUNT+=1

# Benchmark backtest
def bench_fn(window):
    return benchmark_60_40_weights(window)
bt_bench = backtest(prices, bench_fn, lookback_days=252, rebalance_freq="QE")
print(f"\n  60/40 benchmark final: {bt_bench.equity.iloc[-1]:.4f}")
print(f"  60/40 summary: {bt_bench.summary}")
TEST_COUNT+=1

print("\n" + "=" * 60)
print(f"ALL {TEST_COUNT} TESTS PASSED")
print("=" * 60)
