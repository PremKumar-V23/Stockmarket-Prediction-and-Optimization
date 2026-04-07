"""Data access and risk analytics helpers for the portfolio dashboard."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import io
from typing import Iterable

import numpy as np
import pandas as pd


TRADING_DAYS = 252


@dataclass
class PortfolioMetrics:
    expected_return: float
    volatility: float
    sharpe_ratio: float


def normalize_tickers(tickers: Iterable[str]) -> list[str]:
    """Return cleaned ticker symbols without duplicates."""
    cleaned = []
    for ticker in tickers:
        value = str(ticker).strip().upper()
        if value and value not in cleaned:
            cleaned.append(value)
    return cleaned


def fetch_stock_data(
    tickers: Iterable[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch adjusted close prices from Yahoo Finance."""
    try:
        import yfinance as yf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The 'yfinance' package is not installed. Install project dependencies with "
            "'pip install -r requirements.txt' and rerun the app."
        ) from exc

    symbols = normalize_tickers(tickers)
    if not symbols:
        raise ValueError("Please provide at least one valid stock ticker.")

    price_frames: list[pd.Series] = []
    failed_tickers: list[str] = []

    for symbol in symbols:
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
                stderr_buffer
            ):
                data = yf.download(
                    symbol,
                    start=start_date,
                    end=end_date,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
        except Exception:
            failed_tickers.append(symbol)
            continue

        if data.empty:
            failed_tickers.append(symbol)
            continue

        if "Close" in data.columns:
            series = data["Close"].copy()
        else:
            series = data.squeeze().copy()

        if isinstance(series, pd.DataFrame):
            series = series.squeeze(axis=1)

        if not isinstance(series, pd.Series):
            failed_tickers.append(symbol)
            continue

        series = series.dropna()
        if series.empty:
            failed_tickers.append(symbol)
            continue

        price_frames.append(series.rename(symbol))

    if not price_frames:
        raise ValueError(
            "No market data was returned. Yahoo Finance may be temporarily unreachable, "
            "or the selected tickers may be unavailable."
        )

    prices = pd.concat(price_frames, axis=1)
    prices = prices.dropna(how="all").ffill().dropna()
    if prices.empty:
        raise ValueError("Price data became empty after cleaning missing values.")

    prices.attrs["failed_tickers"] = failed_tickers
    return prices


def calculate_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily percentage returns."""
    returns = prices.pct_change().dropna(how="all")
    if returns.empty:
        raise ValueError("Unable to compute daily returns from the selected data.")
    return returns


def calculate_summary_metrics(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.02,
) -> pd.DataFrame:
    """Calculate expected return, volatility, and Sharpe ratio for each stock."""
    annual_returns = returns.mean() * TRADING_DAYS
    annual_volatility = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe_ratio = np.where(
        annual_volatility != 0,
        (annual_returns - risk_free_rate) / annual_volatility,
        np.nan,
    )

    return pd.DataFrame(
        {
            "Expected Annual Return": annual_returns,
            "Annual Volatility": annual_volatility,
            "Sharpe Ratio": sharpe_ratio,
        }
    ).sort_index()


def calculate_covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Annualized covariance matrix."""
    return returns.cov() * TRADING_DAYS


def calculate_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Correlation matrix."""
    return returns.corr()


def compute_portfolio_metrics(
    weights: np.ndarray,
    returns: pd.DataFrame,
    risk_free_rate: float = 0.02,
) -> PortfolioMetrics:
    """Compute portfolio expected return, volatility, and Sharpe ratio."""
    annual_returns = returns.mean() * TRADING_DAYS
    annual_covariance = returns.cov() * TRADING_DAYS

    portfolio_return = float(np.dot(weights, annual_returns))
    portfolio_volatility = float(
        np.sqrt(np.dot(weights.T, np.dot(annual_covariance, weights)))
    )
    if portfolio_volatility == 0:
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    return PortfolioMetrics(
        expected_return=portfolio_return,
        volatility=portfolio_volatility,
        sharpe_ratio=float(sharpe_ratio),
    )


def equal_weight_allocation(columns: Iterable[str]) -> pd.Series:
    """Create an equal-weight allocation vector."""
    names = list(columns)
    if not names:
        raise ValueError("No assets available for equal-weight allocation.")
    weights = np.repeat(1 / len(names), len(names))
    return pd.Series(weights, index=names, name="Weight")
