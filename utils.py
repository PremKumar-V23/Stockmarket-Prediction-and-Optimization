"""Shared UI and export helpers for the Streamlit application."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st


def set_page_config() -> None:
    st.set_page_config(
        page_title="Portfolio Optimization and Risk Analysis",
        layout="wide",
    )


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
            .main {
                background: linear-gradient(180deg, #f7f9fc 0%, #eef4ff 100%);
            }
            .block-container {
                padding-top: 1.2rem;
                padding-bottom: 2rem;
            }
            .metric-card {
                background: white;
                border-radius: 18px;
                padding: 1rem 1.2rem;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
                border: 1px solid rgba(148, 163, 184, 0.15);
            }
            .hero {
                background: linear-gradient(135deg, #0f172a, #1d4ed8);
                padding: 1.4rem 1.6rem;
                border-radius: 22px;
                color: white;
                margin-bottom: 1rem;
                box-shadow: 0 18px 45px rgba(29, 78, 216, 0.28);
            }
            .small-note {
                font-size: 0.9rem;
                color: #475569;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h2 style="margin: 0 0 0.35rem 0;">Portfolio Optimization and Risk Analysis using Machine Learning</h2>
            <p style="margin: 0; opacity: 0.92;">
                Interactive postgraduate-level dashboard for stock analysis, risk evaluation, optimization,
                and machine learning-based prediction.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        """
        <div style="
            margin-top: 2rem;
            padding: 1rem 0 0.5rem 0;
            text-align: center;
            color: #94a3b8;
            font-size: 0.95rem;
            border-top: 1px solid rgba(148, 163, 184, 0.18);
        ">
            Made By V.PremKumar
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(title: str, value: str, help_text: str | None = None) -> None:
    note = f"<div class='small-note'>{help_text}</div>" if help_text else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="small-note">{title}</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #0f172a;">{value}</div>
            {note}
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_default_dates() -> tuple[date, date]:
    end = date.today()
    start = end - timedelta(days=365 * 3)
    return start, end


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8")


def dataframe_to_csv_bytes_no_index(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
