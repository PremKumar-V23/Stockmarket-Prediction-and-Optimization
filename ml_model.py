"""Machine learning helpers for stock price prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def create_features(price_series: pd.Series, lags: int = 5) -> pd.DataFrame:
    """Create lag-based features from a single price series."""
    df = pd.DataFrame({"Close": price_series.copy()})
    for lag in range(1, lags + 1):
        df[f"Lag_{lag}"] = df["Close"].shift(lag)
    df["Rolling_Mean_5"] = df["Close"].rolling(window=5).mean().shift(1)
    df["Rolling_Std_5"] = df["Close"].rolling(window=5).std().shift(1)
    return df.dropna()


def train_prediction_model(
    price_series: pd.Series,
    model_name: str = "Random Forest",
) -> dict[str, object]:
    """Train a regression model and prepare predictions."""
    feature_df = create_features(price_series)
    if len(feature_df) < 20:
        raise ValueError("Not enough data points available to train the prediction model.")

    X = feature_df.drop(columns=["Close"])
    y = feature_df["Close"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    if model_name == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_split=4,
            random_state=42,
        )

    model.fit(X_train, y_train)
    predictions = pd.Series(model.predict(X_test), index=y_test.index, name="Predicted")

    metrics = {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "R2 Score": float(r2_score(y_test, predictions)),
    }

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test.rename("Actual"),
        "predictions": predictions,
        "metrics": metrics,
        "feature_data": feature_df,
    }


def forecast_next_price(
    price_series: pd.Series,
    model_name: str = "Random Forest",
) -> float:
    """Forecast the next closing price using the latest lagged values."""
    trained = train_prediction_model(price_series, model_name=model_name)
    model = trained["model"]
    latest_features = trained["feature_data"].drop(columns=["Close"]).iloc[[-1]]
    return float(model.predict(latest_features)[0])
