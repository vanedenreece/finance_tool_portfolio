"""Black-Litterman model.

The classic formulation blends:
  - an equilibrium prior (returns implied by market-cap weights via reverse optimization)
  - your subjective views (P, Q with uncertainty Omega)
producing a posterior mean that's typically less extreme than either input alone.

Why use it: pure historical means are noisy, and unconstrained Markowitz on those
means produces wildly concentrated portfolios. BL gives you a principled way to
express "I'm bearish on EM equity" or "I think IG will beat HY by 1.5%" without
hand-tuning expected returns.

References:
- Black & Litterman (1992)
- Idzorek (2005), "A Step-By-Step Guide to the Black-Litterman Model"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class View:
    """A single view on portfolio returns.

    For an absolute view: assets={"SPY": 1.0}, return_=0.10  ("SPY will return 10%")
    For a relative view: assets={"SPY": 1.0, "EFA": -1.0}, return_=0.02
                         ("SPY will outperform EFA by 2%")
    confidence: in [0, 1]. 1.0 = certain, 0.0 = useless. Translated to Omega via Idzorek.
    """
    assets: dict[str, float]
    return_: float
    confidence: float = 0.5

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")


def implied_equilibrium_returns(
    cov: pd.DataFrame,
    market_weights: pd.Series,
    risk_aversion: float = 2.5,
) -> pd.Series:
    """Reverse optimization: pi = lambda * Sigma * w_mkt.

    risk_aversion (lambda) of 2.5 is a common default that ties to roughly a
    6-7% global equity risk premium given typical covariance.
    """
    pi = risk_aversion * cov.values @ market_weights.reindex(cov.index).values
    return pd.Series(pi, index=cov.index)


def _build_PQ(
    views: list[View],
    tickers: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    K = len(views)
    N = len(tickers)
    P = np.zeros((K, N))
    Q = np.zeros(K)
    for k, v in enumerate(views):
        for tk, weight in v.assets.items():
            if tk not in tickers:
                raise KeyError(f"View references unknown ticker {tk!r}")
            P[k, tickers.index(tk)] = weight
        Q[k] = v.return_
    return P, Q


def _omega_from_confidence(
    P: np.ndarray,
    cov: np.ndarray,
    confidences: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Idzorek-style Omega from confidence levels.

    The view variance is scaled inverse to confidence. We use the simpler
    He-Litterman convention: Omega_kk = tau * (P_k Sigma P_k') / confidence
    so that confidence -> 1 means very tight view, confidence -> 0 means very loose.
    Returns a diagonal matrix.
    """
    K = P.shape[0]
    omega_diag = np.zeros(K)
    for k in range(K):
        base_var = float(P[k] @ cov @ P[k]) * tau
        # Avoid div-by-zero; floor confidence at a tiny epsilon
        c = max(confidences[k], 1e-3)
        omega_diag[k] = base_var / c
    return np.diag(omega_diag)


def black_litterman(
    cov: pd.DataFrame,
    market_weights: pd.Series,
    views: list[View],
    risk_aversion: float = 2.5,
    tau: float = 0.05,
) -> tuple[pd.Series, pd.DataFrame]:
    """Compute BL posterior expected returns and posterior covariance.

    Returns
    -------
    posterior_mu : pd.Series of expected returns (annualized)
    posterior_cov : pd.DataFrame of the posterior covariance, which is what you
                    should plug into the optimizer (it accounts for view uncertainty).

    Math
    ----
    Prior:    mu ~ N(pi, tau * Sigma)
    Views:    P @ mu = Q + epsilon,  epsilon ~ N(0, Omega)
    Posterior mean:
        mu_bl = [(tau Sigma)^-1 + P' Omega^-1 P]^-1 [(tau Sigma)^-1 pi + P' Omega^-1 Q]
    Posterior covariance of mu (uncertainty in the mean):
        M = [(tau Sigma)^-1 + P' Omega^-1 P]^-1
    Posterior covariance of returns:
        Sigma_bl = Sigma + M
    """
    tickers = list(cov.columns)
    Sigma = cov.values
    pi = implied_equilibrium_returns(cov, market_weights, risk_aversion).values

    P, Q = _build_PQ(views, tickers)
    confidences = np.array([v.confidence for v in views])
    Omega = _omega_from_confidence(P, Sigma, confidences, tau)

    tau_sigma_inv = np.linalg.inv(tau * Sigma)
    omega_inv = np.linalg.inv(Omega)

    M_inv = tau_sigma_inv + P.T @ omega_inv @ P
    M = np.linalg.inv(M_inv)

    mu_bl = M @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)
    sigma_bl = Sigma + M

    return (
        pd.Series(mu_bl, index=tickers),
        pd.DataFrame(sigma_bl, index=tickers, columns=tickers),
    )
