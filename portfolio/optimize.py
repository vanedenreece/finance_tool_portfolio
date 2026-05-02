"""Portfolio optimization with bucket constraints.

Uses cvxpy for proper convex optimization. Min-variance is a quadratic program.
Max-Sharpe is non-convex in general but becomes a convex QP after a standard
transformation (Cornuejols & Tutuncu 2007), which is what we use here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd

from . import assets as A


@dataclass
class OptResult:
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float
    status: str
    objective: str = ""
    diagnostics: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"OptResult({self.objective}, status={self.status}, "
            f"E[R]={self.expected_return:.2%}, vol={self.volatility:.2%}, "
            f"Sharpe={self.sharpe:.3f})"
        )


def _bucket_constraints(
    w: cp.Variable,
    tickers: list[str],
    buckets: dict[str, float] | None,
) -> list[cp.Constraint]:
    """Build cvxpy equality constraints enforcing bucket weight targets.

    If buckets is None, no bucket constraints are added (only sum-to-1 + non-negativity
    are enforced upstream).
    """
    if buckets is None:
        return []
    cons = []
    for bucket_name, target in buckets.items():
        idx = [i for i, tk in enumerate(tickers) if A.bucket_of(tk) == bucket_name]
        if idx:
            cons.append(cp.sum(w[idx]) == target)
    return cons


def min_variance(
    cov: pd.DataFrame,
    expected_returns: Optional[pd.Series] = None,
    buckets: Optional[dict[str, float]] = None,
    risk_free: float = 0.0,
    max_weight: float = 1.0,
) -> OptResult:
    """Minimum-variance portfolio.

    Parameters
    ----------
    cov : annualized covariance matrix (NxN)
    expected_returns : optional, only used to report E[R] and Sharpe of the result
    buckets : optional dict mapping bucket name -> target weight. Defaults to A.BUCKETS.
              Pass {} or None to disable bucket constraints.
    risk_free : annualized risk-free rate, used only for the reported Sharpe ratio
    max_weight : per-asset cap (1.0 = no cap)

    The objective is min w' Sigma w  subject to:
      sum(w) = 1
      w >= 0  (long-only)
      bucket sums = targets (if buckets specified)
      w_i <= max_weight
    """
    if buckets is None:
        buckets = A.BUCKETS

    tickers = list(cov.columns)
    n = len(tickers)
    Sigma = cov.values

    w = cp.Variable(n, nonneg=True)
    constraints = [cp.sum(w) == 1.0, w <= max_weight]
    constraints += _bucket_constraints(w, tickers, buckets)

    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma)))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, canon_backend=cp.SCIPY_CANON_BACKEND)
    # prob.solve(solver=cp.CLARABEL)

    if w.value is None:
        return OptResult(
            weights=pd.Series(np.nan, index=tickers),
            expected_return=np.nan, volatility=np.nan, sharpe=np.nan,
            status=prob.status, objective="min_variance",
        )

    weights = pd.Series(w.value, index=tickers).clip(lower=0)
    weights = weights / weights.sum()  # numerical clean-up

    vol = float(np.sqrt(weights.values @ Sigma @ weights.values))
    er = float(weights @ expected_returns) if expected_returns is not None else np.nan
    sh = (er - risk_free) / vol if expected_returns is not None and vol > 0 else np.nan

    return OptResult(
        weights=weights, expected_return=er, volatility=vol, sharpe=sh,
        status=prob.status, objective="min_variance",
    )


def max_sharpe(
    cov: pd.DataFrame,
    expected_returns: pd.Series,
    buckets: Optional[dict[str, float]] = None,
    risk_free: float = 0.0,
    max_weight: float = 1.0,
) -> OptResult:
    """Maximum Sharpe-ratio portfolio (tangency portfolio).

    Uses the standard QP transformation: introduce auxiliary y >= 0, kappa > 0 with
        w = y / kappa
    Maximize (mu - rf)' w / sqrt(w' Sigma w)  becomes
        minimize y' Sigma y  subject to (mu - rf)' y = 1, y >= 0, kappa > 0, ...
    The bucket constraints  sum(w_bucket) = target  become  sum(y_bucket) = target * kappa
    which is linear in (y, kappa).
    """
    if buckets is None:
        buckets = A.BUCKETS

    tickers = list(cov.columns)
    n = len(tickers)
    Sigma = cov.values
    mu_excess = (expected_returns.reindex(tickers).values - risk_free)

    # If max excess return is non-positive, max-Sharpe is undefined / unbounded
    if np.max(mu_excess) <= 0:
        return OptResult(
            weights=pd.Series(np.nan, index=tickers),
            expected_return=np.nan, volatility=np.nan, sharpe=np.nan,
            status="infeasible_no_positive_excess_return", objective="max_sharpe",
        )

    y = cp.Variable(n, nonneg=True)
    kappa = cp.Variable(nonneg=True)

    constraints = [
        mu_excess @ y == 1.0,            # normalization
        cp.sum(y) == kappa,              # sum(w) = 1
        y <= max_weight * kappa,         # per-asset cap in w-space
    ]
    # Bucket constraints in y-space:  sum(y_bucket) == target * kappa
    for bucket_name, target in buckets.items():
        idx = [i for i, tk in enumerate(tickers) if A.bucket_of(tk) == bucket_name]
        if idx:
            constraints.append(cp.sum(y[idx]) == target * kappa)

    objective = cp.Minimize(cp.quad_form(y, cp.psd_wrap(Sigma)))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)

    if y.value is None or kappa.value is None or kappa.value < 1e-12:
        return OptResult(
            weights=pd.Series(np.nan, index=tickers),
            expected_return=np.nan, volatility=np.nan, sharpe=np.nan,
            status=prob.status, objective="max_sharpe",
        )

    w_val = y.value / kappa.value
    weights = pd.Series(w_val, index=tickers).clip(lower=0)
    weights = weights / weights.sum()

    er = float(weights @ expected_returns)
    vol = float(np.sqrt(weights.values @ Sigma @ weights.values))
    sh = (er - risk_free) / vol if vol > 0 else np.nan

    return OptResult(
        weights=weights, expected_return=er, volatility=vol, sharpe=sh,
        status=prob.status, objective="max_sharpe",
    )


def target_return(
    cov: pd.DataFrame,
    expected_returns: pd.Series,
    target: float,
    buckets: Optional[dict[str, float]] = None,
    risk_free: float = 0.0,
    max_weight: float = 1.0,
) -> OptResult:
    """Minimum-variance portfolio achieving an expected return >= target.

    Used to trace the efficient frontier point-by-point.
    """
    if buckets is None:
        buckets = A.BUCKETS

    tickers = list(cov.columns)
    n = len(tickers)
    Sigma = cov.values
    mu = expected_returns.reindex(tickers).values

    w = cp.Variable(n, nonneg=True)
    constraints = [cp.sum(w) == 1.0, w <= max_weight, mu @ w >= target]
    constraints += _bucket_constraints(w, tickers, buckets)

    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma))), constraints)
    prob.solve(solver=cp.CLARABEL)

    if w.value is None:
        return OptResult(
            weights=pd.Series(np.nan, index=tickers),
            expected_return=np.nan, volatility=np.nan, sharpe=np.nan,
            status=prob.status, objective=f"target_return({target:.2%})",
        )

    weights = pd.Series(w.value, index=tickers).clip(lower=0)
    weights = weights / weights.sum()
    er = float(weights @ expected_returns)
    vol = float(np.sqrt(weights.values @ Sigma @ weights.values))
    sh = (er - risk_free) / vol if vol > 0 else np.nan

    return OptResult(
        weights=weights, expected_return=er, volatility=vol, sharpe=sh,
        status=prob.status, objective=f"target_return({target:.2%})",
    )
