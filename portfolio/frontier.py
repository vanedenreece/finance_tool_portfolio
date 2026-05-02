"""Efficient frontier construction.

Two flavors:
- bucket-constrained frontier: traces the frontier subject to your policy buckets
- unconstrained frontier: classic Markowitz frontier for comparison
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .optimize import min_variance, max_sharpe, target_return


def _max_feasible_return(
    cov: pd.DataFrame,
    expected_returns: pd.Series,
    buckets: Optional[dict[str, float]] = None,
) -> float:
    """Solve for the highest expected return achievable under the constraints.

    This is a linear program: max mu' w  s.t. sum(w)=1, w>=0, bucket constraints.
    When bucket constraints bind, this is < mu.max().
    """
    import cvxpy as cp
    from . import assets as A

    if buckets is None:
        buckets = A.BUCKETS

    tickers = list(cov.columns)
    n = len(tickers)
    mu = expected_returns.reindex(tickers).values

    w = cp.Variable(n, nonneg=True)
    constraints = [cp.sum(w) == 1.0]
    for bucket_name, target in buckets.items():
        idx = [i for i, tk in enumerate(tickers) if A.bucket_of(tk) == bucket_name]
        if idx:
            constraints.append(cp.sum(w[idx]) == target)

    prob = cp.Problem(cp.Maximize(mu @ w), constraints)
    prob.solve(solver=cp.CLARABEL)
    if w.value is None:
        return float(expected_returns.max())  # fallback
    return float(mu @ w.value)


def efficient_frontier(
    cov: pd.DataFrame,
    expected_returns: pd.Series,
    buckets: Optional[dict[str, float]] = None,
    n_points: int = 30,
    risk_free: float = 0.0,
) -> pd.DataFrame:
    """Trace the efficient frontier between the min-variance and max-return portfolios.

    Returns a DataFrame with columns: target_return, achieved_return, vol, sharpe,
    plus one column per ticker holding the optimal weight at that point.

    The upper end is the maximum return achievable under the bucket constraints,
    which can be substantially below mu.max() when the bucket policy binds.
    """
    mvp = min_variance(cov, expected_returns, buckets=buckets, risk_free=risk_free)
    if not np.isfinite(mvp.expected_return):
        raise RuntimeError(f"Min-variance failed: status={mvp.status}")

    r_min = mvp.expected_return
    r_max = _max_feasible_return(cov, expected_returns, buckets=buckets)
    if r_max <= r_min:
        # Degenerate case — only one feasible portfolio
        return pd.DataFrame([{
            "target_return": r_min, "achieved_return": r_min,
            "vol": mvp.volatility, "sharpe": mvp.sharpe,
            **mvp.weights.to_dict(),
        }])
    # Tiny shave off the top to avoid numerical infeasibility right at the boundary
    targets = np.linspace(r_min, r_max - 1e-6, n_points)

    rows = []
    for t in targets:
        res = target_return(cov, expected_returns, t, buckets=buckets, risk_free=risk_free)
        if not np.isfinite(res.expected_return):
            continue
        row = {
            "target_return": t,
            "achieved_return": res.expected_return,
            "vol": res.volatility,
            "sharpe": res.sharpe,
            **res.weights.to_dict(),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def frontier_summary(
    cov: pd.DataFrame,
    expected_returns: pd.Series,
    buckets: Optional[dict[str, float]] = None,
    risk_free: float = 0.0,
) -> pd.DataFrame:
    """Return a one-row-per-portfolio table for the key portfolios:
    min-variance, max-Sharpe (tangency), and equal-weight as a baseline.
    """
    mvp = min_variance(cov, expected_returns, buckets=buckets, risk_free=risk_free)
    msr = max_sharpe(cov, expected_returns, buckets=buckets, risk_free=risk_free)

    # Equal weight: 1/N, ignoring bucket constraints (this is the naive baseline)
    n = len(cov.columns)
    eq = pd.Series(1.0 / n, index=cov.columns)
    eq_er = float(eq @ expected_returns)
    eq_vol = float(np.sqrt(eq.values @ cov.values @ eq.values))
    eq_sh = (eq_er - risk_free) / eq_vol if eq_vol > 0 else np.nan

    rows = [
        {"portfolio": "Min Variance", "E[R]": mvp.expected_return, "Vol": mvp.volatility, "Sharpe": mvp.sharpe},
        {"portfolio": "Max Sharpe",   "E[R]": msr.expected_return, "Vol": msr.volatility, "Sharpe": msr.sharpe},
        {"portfolio": "Equal Weight (1/N)", "E[R]": eq_er, "Vol": eq_vol, "Sharpe": eq_sh},
    ]
    return pd.DataFrame(rows).set_index("portfolio")
