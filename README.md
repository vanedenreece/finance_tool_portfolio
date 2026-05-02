# Portfolio Management Tool

Bucket-constrained Markowitz portfolio optimization in Python. A working tool, structured for extension.

## What it does

Given a policy split — by default 50% Fixed Income, 40% Equities, 10% Alternatives — this finds the optimal mix **inside** each bucket. The bucket weights themselves are fixed; that's strategic asset allocation. The optimization happens on the satellite holdings.

Five capabilities:

1. **Real historical data** via yfinance, with on-disk caching
2. **Robust covariance estimation** with Ledoit-Wolf shrinkage
3. **Convex optimization** for min-variance, max-Sharpe, and target-return portfolios under bucket constraints
4. **Black-Litterman** to blend market-implied returns with your subjective views
5. **Walk-forward backtesting** with periodic rebalancing and transaction costs

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/01_walkthrough.ipynb
```

If you've never set up Python before, read `SETUP.md` first.

## Project layout

```
portfolio/                 # the package — import this
├── assets.py             # universe and bucket policy
├── data.py               # yfinance fetching with caching
├── covariance.py         # sample + Ledoit-Wolf
├── optimize.py           # min-variance, max-Sharpe, target-return (cvxpy)
├── frontier.py           # efficient frontier
├── black_litterman.py    # views overlay
├── backtest.py           # walk-forward backtester
└── plotting.py           # frontier and equity-curve charts

notebooks/
└── 01_walkthrough.ipynb  # the driving notebook — start here

tests/
└── test_pipeline.py      # synthetic-data smoke test (offline)
```

## Customization

- **Change the asset universe**: edit `ASSETS` in `portfolio/assets.py`. UCITS ETFs, individual stocks, factor ETFs — any tickers yfinance can resolve.
- **Change the policy split**: edit `BUCKETS` in the same file. Must sum to 1.0; every bucket must contain at least one asset.
- **Use your own data**: `load_prices_csv("path.csv")` accepts a CSV with a `Date` column and one column per ticker.
- **Add a new optimizer objective**: drop a function into `optimize.py` mirroring `min_variance` — same constraint pattern, different `cp.Minimize(...)` expression.

## Caveats

- Mean-variance optimization is sensitive to expected-return inputs. The bucket constraints in this tool act as heavy regularization, which is why the results are reasonably stable; without them, point estimates of historical means produce wildly concentrated portfolios.
- The backtester models transaction costs as a flat bps charge on turnover. Real execution involves spreads, market impact, and (depending on jurisdiction) taxes — not modeled here.
- Yahoo Finance data is free but imperfect. For a production tool, swap in a paid feed via `data.py`.

