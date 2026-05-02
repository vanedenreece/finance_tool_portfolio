"""Plotting helpers. matplotlib-only to keep dependencies light."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_frontier(
    frontier: pd.DataFrame,
    cov: pd.DataFrame,
    expected_returns: pd.Series,
    extra_points: Optional[dict[str, tuple[float, float]]] = None,
    title: str = "Efficient Frontier",
    ax=None,
):
    """Plot the efficient frontier (vol on x, return on y).

    Parameters
    ----------
    frontier : output of efficient_frontier()
    cov, expected_returns : used to plot individual assets as reference dots
    extra_points : optional dict mapping label -> (vol, return) for highlighted portfolios
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(frontier["vol"], frontier["achieved_return"], "-", linewidth=2, label="Frontier", color="#1f77b4")

    # Individual asset dots
    asset_vols = np.sqrt(np.diag(cov.values))
    asset_rets = expected_returns.reindex(cov.columns).values
    ax.scatter(asset_vols, asset_rets, s=40, alpha=0.6, color="gray", label="Individual assets")
    for tk, v, r in zip(cov.columns, asset_vols, asset_rets):
        ax.annotate(tk, (v, r), fontsize=8, alpha=0.7,
                    xytext=(5, 5), textcoords="offset points")

    # Highlighted portfolios
    if extra_points:
        colors = ["#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
        for i, (label, (v, r)) in enumerate(extra_points.items()):
            ax.scatter([v], [r], s=120, marker="*", color=colors[i % len(colors)],
                       label=label, zorder=5, edgecolors="black", linewidths=0.5)

    ax.set_xlabel("Volatility (annualized)")
    ax.set_ylabel("Expected Return (annualized)")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    return ax


def plot_equity_curves(
    curves: dict[str, pd.Series],
    title: str = "Backtest: Cumulative Returns",
    ax=None,
):
    """Plot one or more equity curves on the same axes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    for label, series in curves.items():
        ax.plot(series.index, series.values, label=label, linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return (start = 1.0)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    return ax


def plot_weights_over_time(weights: pd.DataFrame, title: str = "Weights Over Time", ax=None):
    """Stacked area chart of portfolio weights at each rebalance date."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    weights.plot.area(ax=ax, stacked=True, alpha=0.85, linewidth=0)
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    return ax
