"""Covariance matrix estimators.

Sample covariance is unbiased but noisy, especially when N (assets) is close to T (observations).
Shrinkage estimators trade bias for variance, generally producing better out-of-sample portfolios.

References:
- Ledoit & Wolf (2004), "Honey, I shrunk the sample covariance matrix"
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sample_cov(returns: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    """Plain annualized sample covariance."""
    return returns.cov() * periods_per_year


def ledoit_wolf_cov(returns: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage toward a constant-correlation target.

    Returns the annualized shrunk covariance matrix.
    """
    from sklearn.covariance import LedoitWolf

    daily_lw = LedoitWolf().fit(returns.values).covariance_
    annual = daily_lw * periods_per_year
    return pd.DataFrame(annual, index=returns.columns, columns=returns.columns)


def to_correlation(cov: pd.DataFrame) -> pd.DataFrame:
    """Convert a covariance matrix to a correlation matrix."""
    vols = np.sqrt(np.diag(cov.values))
    outer = np.outer(vols, vols)
    corr = cov.values / outer
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)
