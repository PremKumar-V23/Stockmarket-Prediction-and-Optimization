from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data import (
    calculate_correlation_matrix,
    calculate_covariance_matrix,
    calculate_daily_returns,
    calculate_summary_metrics,
    compute_portfolio_metrics,
    equal_weight_allocation,
    fetch_stock_data,
)
from ml_model import forecast_next_price, train_prediction_model
from optimization import (
    build_allocation_table,
    compare_portfolios,
    identify_key_portfolios,
    prepare_efficient_frontier_points,
    simulate_random_portfolios,
)
from utils import (
    dataframe_to_csv_bytes,
    dataframe_to_csv_bytes_no_index,
    get_default_dates,
    inject_custom_css,
    metric_card,
    render_footer,
    render_hero,
    set_page_config,
)


DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "TSLA",
    "NVDA",
    "META",
    "JPM",
    "NFLX",
    "IBM",
    "AMD",
    "INTC",
    "ORCL",
    "ADBE",
    "CRM",
    "QCOM",
    "AVGO",
    "CSCO",
    "WMT",
    "COST",
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "LT.NS",
    "HINDUNILVR.NS",
    "ITC.NS",
    "BHARTIARTL.NS",
]


@st.cache_data(show_spinner=False)
def load_market_data(tickers: tuple[str, ...], start_date, end_date) -> pd.DataFrame:
    return fetch_stock_data(list(tickers), str(start_date), str(end_date))


def parse_custom_tickers(raw_text: str) -> list[str]:
    return [ticker.strip().upper() for ticker in raw_text.split(",") if ticker.strip()]


def render_sidebar() -> tuple[list[str], object, object, float, int]:
    st.sidebar.header("Input Parameters")
    start_default, end_default = get_default_dates()

    selected_tickers = st.sidebar.multiselect(
        "Select Stocks",
        options=DEFAULT_TICKERS,
        default=["AAPL", "MSFT", "GOOGL", "AMZN"],
    )
    custom_tickers = st.sidebar.text_input(
        "Add Custom Tickers",
        value="",
        help="Enter extra ticker symbols separated by commas, for example: TCS.NS, INFY.NS",
    )
    start_date = st.sidebar.date_input("Start Date", value=start_default)
    end_date = st.sidebar.date_input("End Date", value=end_default)
    risk_free_rate = st.sidebar.slider(
        "Risk-Free Rate",
        min_value=0.0,
        max_value=0.15,
        value=0.02,
        step=0.005,
        format="%.3f",
        help="Risk-free rate is the theoretical annual return from a nearly riskless investment, used in Sharpe ratio calculations.",
    )
    num_portfolios = st.sidebar.slider(
        "Random Portfolios",
        min_value=1000,
        max_value=20000,
        value=5000,
        step=1000,
        help="Number of simulated portfolios used to approximate the efficient frontier. Higher values improve coverage but take longer to compute.",
    )
    tickers = list(dict.fromkeys(selected_tickers + parse_custom_tickers(custom_tickers)))
    return tickers, start_date, end_date, risk_free_rate, num_portfolios


def plot_price_chart(prices: pd.DataFrame) -> go.Figure:
    fig = px.line(
        prices,
        x=prices.index,
        y=prices.columns,
        labels={"x": "Date", "value": "Adjusted Close Price", "variable": "Ticker"},
        title="Historical Stock Prices",
    )
    fig.update_layout(legend_title_text="Ticker", template="plotly_white")
    return fig


def plot_heatmap(matrix: pd.DataFrame, title: str) -> go.Figure:
    fig = px.imshow(
        matrix,
        text_auto=".2f",
        color_continuous_scale="Blues",
        aspect="auto",
        title=title,
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_efficient_frontier(
    portfolios: pd.DataFrame,
    frontier: pd.DataFrame,
    key_portfolios: dict[str, pd.Series],
) -> go.Figure:
    fig = px.scatter(
        portfolios,
        x="Volatility",
        y="Return",
        color="Sharpe Ratio",
        color_continuous_scale="Viridis",
        title="Efficient Frontier and Random Portfolios",
        hover_data={"Sharpe Ratio": ":.3f"},
    )
    fig.update_traces(
        selector=dict(mode="markers"),
        marker=dict(size=8, opacity=0.55),
        showlegend=False,
    )
    fig.add_trace(
        go.Scatter(
            x=frontier["Volatility"],
            y=frontier["Return"],
            mode="lines",
            name="Efficient Frontier",
            line=dict(color="#dc2626", width=3),
        )
    )

    markers = {
        "max_sharpe": ("Max Sharpe", "#16a34a", "star"),
        "min_volatility": ("Minimum Risk", "#2563eb", "diamond"),
        "best_return": ("Best Return", "#f59e0b", "triangle-up"),
        "worst_return": ("Worst Return", "#7c3aed", "x"),
    }
    for key, (label, color, symbol) in markers.items():
        row = key_portfolios[key]
        fig.add_trace(
            go.Scatter(
                x=[row["Volatility"]],
                y=[row["Return"]],
                mode="markers",
                marker=dict(size=15, color=color, symbol=symbol, line=dict(width=1)),
                name=label,
            )
        )

    fig.update_layout(
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(15, 23, 42, 0.82)",
            font=dict(color="#e2e8f0", size=14),
            bordercolor="rgba(148, 163, 184, 0.25)",
            borderwidth=1,
        ),
        coloraxis_colorbar=dict(
            title=dict(text="Sharpe Ratio", side="top", font=dict(color="#e2e8f0")),
            tickfont=dict(color="#e2e8f0"),
            x=1.08,
        ),
        margin=dict(t=90, r=90, b=40, l=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.22)",
        font=dict(color="#e2e8f0"),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.22)", zerolinecolor="rgba(148,163,184,0.22)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.22)", zerolinecolor="rgba(148,163,184,0.22)")
    return fig


def plot_allocation_pie(weights_table: pd.DataFrame, title: str) -> go.Figure:
    fig = px.pie(
        weights_table,
        names="Ticker",
        values="Allocation %",
        hole=0.35,
        title=title,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def plot_prediction_results(actual: pd.Series, predicted: pd.Series) -> go.Figure:
    chart_df = pd.concat([actual, predicted], axis=1).reset_index()
    chart_df.columns = ["Date", "Actual", "Predicted"]
    fig = px.line(
        chart_df,
        x="Date",
        y=["Actual", "Predicted"],
        labels={"value": "Price", "variable": "Series"},
        title="Actual vs Predicted Stock Price",
    )
    fig.update_layout(template="plotly_white")
    return fig


def render_stock_analysis(prices: pd.DataFrame) -> None:
    st.subheader("Stock Analysis")
    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Stocks Selected", str(prices.shape[1]))
    with col2:
        metric_card("Observations", str(prices.shape[0]))
    with col3:
        metric_card("Latest Trading Date", str(prices.index.max().date()))

    st.plotly_chart(plot_price_chart(prices), width="stretch")
    st.dataframe(prices.tail(10), width="stretch")
    st.download_button(
        "Export Price Data to CSV",
        data=dataframe_to_csv_bytes(prices),
        file_name="historical_prices.csv",
        mime="text/csv",
    )


def render_risk_analysis(returns: pd.DataFrame, risk_free_rate: float) -> None:
    st.subheader("Risk Analysis")
    summary = calculate_summary_metrics(returns, risk_free_rate)
    covariance_matrix = calculate_covariance_matrix(returns)
    correlation_matrix = calculate_correlation_matrix(returns)

    avg_return = summary["Expected Annual Return"].mean()
    avg_volatility = summary["Annual Volatility"].mean()
    best_sharpe = summary["Sharpe Ratio"].max()

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Average Expected Return", f"{avg_return:.2%}")
    with col2:
        metric_card("Average Volatility", f"{avg_volatility:.2%}")
    with col3:
        metric_card("Best Sharpe Ratio", f"{best_sharpe:.3f}")

    st.markdown("#### Daily Returns")
    st.dataframe(returns.tail(15), width="stretch")

    st.markdown("#### Summary Metrics")
    st.dataframe(summary.style.format("{:.4f}"), width="stretch")

    heatmap_col1, heatmap_col2 = st.columns(2)
    with heatmap_col1:
        st.plotly_chart(
            plot_heatmap(correlation_matrix, "Correlation Heatmap"),
            width="stretch",
        )
    with heatmap_col2:
        st.plotly_chart(
            plot_heatmap(covariance_matrix, "Covariance Matrix"),
            width="stretch",
        )

    st.download_button(
        "Export Risk Metrics to CSV",
        data=dataframe_to_csv_bytes(summary),
        file_name="risk_metrics.csv",
        mime="text/csv",
    )


def render_portfolio_optimization(
    returns: pd.DataFrame,
    risk_free_rate: float,
    num_portfolios: int,
) -> dict[str, object]:
    st.subheader("Portfolio Optimization")
    portfolios = simulate_random_portfolios(returns, num_portfolios, risk_free_rate)
    key_portfolios = identify_key_portfolios(portfolios)
    frontier = prepare_efficient_frontier_points(portfolios)
    tickers = list(returns.columns)

    optimized = key_portfolios["max_sharpe"]
    minimum_risk = key_portfolios["min_volatility"]
    best_return = key_portfolios["best_return"]
    worst_return = key_portfolios["worst_return"]

    equal_weights = equal_weight_allocation(tickers)
    equal_metrics = compute_portfolio_metrics(
        equal_weights.values, returns, risk_free_rate=risk_free_rate
    )
    equal_metrics_dict = {
        "return": equal_metrics.expected_return,
        "volatility": equal_metrics.volatility,
        "sharpe_ratio": equal_metrics.sharpe_ratio,
    }

    top_col1, top_col2, top_col3, top_col4 = st.columns(4)
    with top_col1:
        metric_card("Optimal Return", f"{optimized['Return']:.2%}")
    with top_col2:
        metric_card("Optimal Volatility", f"{optimized['Volatility']:.2%}")
    with top_col3:
        metric_card("Optimal Sharpe", f"{optimized['Sharpe Ratio']:.3f}")
    with top_col4:
        metric_card("Minimum Risk", f"{minimum_risk['Volatility']:.2%}")

    st.plotly_chart(
        plot_efficient_frontier(portfolios, frontier, key_portfolios),
        width="stretch",
    )

    legend_help = pd.DataFrame(
        [
            {"Legend": "Efficient Frontier", "Meaning": "Best achievable return levels for different risk ranges based on the simulated portfolios."},
            {"Legend": "Max Sharpe", "Meaning": "Portfolio with the highest Sharpe ratio, giving the best risk-adjusted return."},
            {"Legend": "Minimum Risk", "Meaning": "Portfolio with the lowest volatility among all simulated portfolios."},
            {"Legend": "Best Return", "Meaning": "Portfolio with the highest expected annual return, regardless of risk."},
            {"Legend": "Worst Return", "Meaning": "Portfolio with the lowest expected annual return in the simulation."},
        ]
    )
    with st.expander("Legend Help for Efficient Frontier"):
        st.dataframe(legend_help, width="stretch", hide_index=True)

    optimized_table = build_allocation_table(optimized, tickers, "Optimized")
    min_risk_table = build_allocation_table(minimum_risk, tickers, "Minimum Risk")
    compare_df = compare_portfolios(optimized, equal_metrics_dict)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Optimal Portfolio Weights")
        st.dataframe(
            optimized_table.style.format({"Weight": "{:.4f}", "Allocation %": "{:.2f}"}),
            width="stretch",
        )
        st.plotly_chart(
            plot_allocation_pie(optimized_table, "Optimal Allocation"),
            width="stretch",
        )
    with col2:
        st.markdown("#### Minimum Risk Portfolio Weights")
        st.dataframe(
            min_risk_table.style.format({"Weight": "{:.4f}", "Allocation %": "{:.2f}"}),
            width="stretch",
        )
        st.plotly_chart(
            plot_allocation_pie(min_risk_table, "Minimum Risk Allocation"),
            width="stretch",
        )

    st.markdown("#### Optimized vs Equal Weight")
    st.dataframe(
        compare_df.style.format(
            {
                "Expected Return": "{:.2%}",
                "Volatility": "{:.2%}",
                "Sharpe Ratio": "{:.3f}",
            }
        ),
        width="stretch",
    )

    st.markdown("#### Investment Return Calculator (INR)")
    investment_amount = st.number_input(
        "Enter investment amount in Rupees",
        min_value=1000.0,
        value=100000.0,
        step=5000.0,
    )
    optimized_future_value = investment_amount * (1 + optimized["Return"])
    optimized_profit = optimized_future_value - investment_amount
    equal_future_value = investment_amount * (1 + equal_metrics_dict["return"])
    equal_profit = equal_future_value - investment_amount

    calc_col1, calc_col2 = st.columns(2)
    with calc_col1:
        metric_card("Optimized Portfolio Value", f"Rs. {optimized_future_value:,.2f}")
        metric_card("Optimized Estimated Profit", f"Rs. {optimized_profit:,.2f}")
    with calc_col2:
        metric_card("Equal Weight Value", f"Rs. {equal_future_value:,.2f}")
        metric_card("Equal Weight Estimated Profit", f"Rs. {equal_profit:,.2f}")

    extra_summary = pd.DataFrame(
        [
            {
                "Portfolio": "Best Return",
                "Expected Return": best_return["Return"],
                "Volatility": best_return["Volatility"],
                "Sharpe Ratio": best_return["Sharpe Ratio"],
            },
            {
                "Portfolio": "Worst Return",
                "Expected Return": worst_return["Return"],
                "Volatility": worst_return["Volatility"],
                "Sharpe Ratio": worst_return["Sharpe Ratio"],
            },
        ]
    )
    st.markdown("#### Best and Worst Portfolios")
    st.dataframe(
        extra_summary.style.format(
            {
                "Expected Return": "{:.2%}",
                "Volatility": "{:.2%}",
                "Sharpe Ratio": "{:.3f}",
            }
        ),
        width="stretch",
    )

    st.download_button(
        "Export Portfolio Simulation to CSV",
        data=dataframe_to_csv_bytes_no_index(portfolios),
        file_name="portfolio_simulation.csv",
        mime="text/csv",
    )

    return {
        "optimized": optimized,
        "comparison": compare_df,
    }


def render_prediction(prices: pd.DataFrame) -> dict[str, object]:
    st.subheader("Machine Learning Prediction")
    ticker = st.selectbox("Select stock for prediction", list(prices.columns))
    model_name = st.radio(
        "Choose ML model",
        options=["Random Forest", "Linear Regression"],
        horizontal=True,
    )

    try:
        result = train_prediction_model(prices[ticker], model_name=model_name)
        next_price = forecast_next_price(prices[ticker], model_name=model_name)
    except ValueError as exc:
        st.warning(str(exc))
        return {
            "ticker": ticker,
            "model_name": model_name,
            "next_price": float("nan"),
        }
    metrics = result["metrics"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("MAE", f"{metrics['MAE']:.2f}")
    with col2:
        metric_card("RMSE", f"{metrics['RMSE']:.2f}")
    with col3:
        metric_card("R2 Score", f"{metrics['R2 Score']:.3f}")
    with col4:
        metric_card("Next Predicted Price", f"{next_price:.2f}")

    st.plotly_chart(
        plot_prediction_results(result["y_test"], result["predictions"]),
        width="stretch",
    )

    prediction_table = pd.concat([result["y_test"], result["predictions"]], axis=1)
    st.dataframe(prediction_table.tail(20), width="stretch")
    st.download_button(
        "Export Prediction Results to CSV",
        data=dataframe_to_csv_bytes(prediction_table),
        file_name=f"{ticker.lower()}_prediction_results.csv",
        mime="text/csv",
    )

    return {
        "ticker": ticker,
        "model_name": model_name,
        "next_price": next_price,
    }


def render_final_summary(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    optimization_result: dict[str, object],
    prediction_result: dict[str, object],
) -> None:
    st.subheader("Final Result Summary")
    optimized = optimization_result["optimized"]
    comparison = optimization_result["comparison"]

    summary_df = pd.DataFrame(
        [
            {
                "Section": "Dataset",
                "Observation": f"{prices.shape[1]} stocks analyzed over {prices.shape[0]} trading days",
            },
            {
                "Section": "Risk",
                "Observation": f"Average daily return across assets: {returns.mean().mean():.4%}",
            },
            {
                "Section": "Optimization",
                "Observation": (
                    f"Best Sharpe portfolio return {optimized['Return']:.2%}, "
                    f"volatility {optimized['Volatility']:.2%}"
                ),
            },
            {
                "Section": "Comparison",
                "Observation": (
                    f"Optimized Sharpe {comparison.iloc[0]['Sharpe Ratio']:.3f} vs "
                    f"equal-weight Sharpe {comparison.iloc[1]['Sharpe Ratio']:.3f}"
                ),
            },
            {
                "Section": "Prediction",
                "Observation": (
                    f"{prediction_result['model_name']} predicts next {prediction_result['ticker']} "
                    f"price near {prediction_result['next_price']:.2f}"
                ),
            },
        ]
    )

    st.dataframe(summary_df, width="stretch")
    st.markdown(
        """
        **Project Conclusion:** The dashboard combines descriptive analytics, risk evaluation,
        Modern Portfolio Theory, and machine learning prediction into a unified decision-support tool.
        It helps investors compare assets, quantify risk, optimize weights, and study short-term price behavior.
        """
    )

    st.download_button(
        "Export Final Summary to CSV",
        data=dataframe_to_csv_bytes_no_index(summary_df),
        file_name="final_project_summary.csv",
        mime="text/csv",
    )


def main() -> None:
    set_page_config()
    inject_custom_css()
    render_hero()

    tickers, start_date, end_date, risk_free_rate, num_portfolios = render_sidebar()

    if not tickers:
        st.info("Select at least one stock ticker from the sidebar to begin the analysis.")
        st.stop()

    if start_date >= end_date:
        st.error("Start date must be earlier than end date.")
        st.stop()

    try:
        with st.spinner("Fetching market data and preparing analytics..."):
            prices = load_market_data(tuple(tickers), start_date, end_date)
            returns = calculate_daily_returns(prices)
    except Exception as exc:
        st.error(f"Unable to load stock data: {exc}")
        st.stop()

    failed_tickers = prices.attrs.get("failed_tickers", [])
    if failed_tickers:
        st.warning(
            "Some tickers could not be loaded and were skipped: "
            + ", ".join(failed_tickers)
        )

    pages = [
        "Stock Analysis",
        "Risk Analysis",
        "Portfolio Optimization",
        "Prediction (Machine Learning)",
        "Final Result Summary",
    ]
    selected_page = st.radio("Navigate Dashboard", options=pages, horizontal=True)

    if selected_page == "Stock Analysis":
        render_stock_analysis(prices)
    elif selected_page == "Risk Analysis":
        render_risk_analysis(returns, risk_free_rate)
    elif selected_page == "Portfolio Optimization":
        render_portfolio_optimization(returns, risk_free_rate, num_portfolios)
    elif selected_page == "Prediction (Machine Learning)":
        render_prediction(prices)
    else:
        optimization_result = render_portfolio_optimization(
            returns, risk_free_rate, num_portfolios
        )
        prediction_result = render_prediction(prices)
        render_final_summary(prices, returns, optimization_result, prediction_result)

    render_footer()


if __name__ == "__main__":
    main()
