"""Portfolio optimization utilities based on Modern Portfolio Theory."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data import compute_portfolio_metrics


def _generate_random_weights(num_assets: int) -> np.ndarray:
    weights = np.random.random(num_assets)
    return weights / np.sum(weights)


def simulate_random_portfolios(
    returns: pd.DataFrame,
    num_portfolios: int = 5000,
    risk_free_rate: float = 0.02,
) -> pd.DataFrame:
    """Generate random portfolios and calculate their metrics."""
    tickers = list(returns.columns)
    portfolios = []

    for _ in range(num_portfolios):
        weights = _generate_random_weights(len(tickers))
        metrics = compute_portfolio_metrics(weights, returns, risk_free_rate)
        row = {
            "Return": metrics.expected_return,
            "Volatility": metrics.volatility,
            "Sharpe Ratio": metrics.sharpe_ratio,
        }
        row.update({ticker: weight for ticker, weight in zip(tickers, weights)})
        portfolios.append(row)

    portfolio_df = pd.DataFrame(portfolios)
    portfolio_df["Portfolio Label"] = "Random Portfolio"
    return portfolio_df


def identify_key_portfolios(portfolios: pd.DataFrame) -> dict[str, pd.Series]:
    """Return max-Sharpe, min-volatility, best-return, and worst-return portfolios."""
    return {
        "max_sharpe": portfolios.loc[portfolios["Sharpe Ratio"].idxmax()],
        "min_volatility": portfolios.loc[portfolios["Volatility"].idxmin()],
        "best_return": portfolios.loc[portfolios["Return"].idxmax()],
        "worst_return": portfolios.loc[portfolios["Return"].idxmin()],
    }


def build_allocation_table(
    portfolio_row: pd.Series,
    tickers: list[str],
    label: str,
) -> pd.DataFrame:
    """Convert a portfolio row into a readable allocation table."""
    weights = portfolio_row[tickers].astype(float)
    table = pd.DataFrame(
        {
            "Ticker": tickers,
            "Weight": weights.values,
            "Allocation %": (weights.values * 100).round(2),
            "Portfolio": label,
        }
    )
    return table.sort_values("Allocation %", ascending=False).reset_index(drop=True)


def compare_portfolios(
    optimized_portfolio: pd.Series,
    equal_weight_metrics: dict[str, float],
) -> pd.DataFrame:
    """Build a side-by-side comparison between optimized and equal-weight portfolios."""
    optimized_summary = {
        "Portfolio": "Optimized (Max Sharpe)",
        "Expected Return": optimized_portfolio["Return"],
        "Volatility": optimized_portfolio["Volatility"],
        "Sharpe Ratio": optimized_portfolio["Sharpe Ratio"],
    }
    equal_weight_summary = {
        "Portfolio": "Equal Weight",
        "Expected Return": equal_weight_metrics["return"],
        "Volatility": equal_weight_metrics["volatility"],
        "Sharpe Ratio": equal_weight_metrics["sharpe_ratio"],
    }
    return pd.DataFrame([optimized_summary, equal_weight_summary])


def prepare_efficient_frontier_points(
    portfolios: pd.DataFrame,
    bins: int = 60,
) -> pd.DataFrame:
    """Approximate the efficient frontier from simulated portfolios."""
    work = portfolios[["Return", "Volatility"]].copy().sort_values("Volatility")
    work["Return Bucket"] = pd.cut(work["Return"], bins=bins, duplicates="drop")
    frontier = (
        work.groupby("Return Bucket", observed=False)["Volatility"]
        .min()
        .reset_index()
        .dropna()
    )
    frontier["Return"] = frontier["Return Bucket"].apply(lambda bucket: bucket.mid)
    return frontier[["Volatility", "Return"]].sort_values("Volatility")
