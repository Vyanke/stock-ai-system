"""
dashboard/app.py
Northgate AI — Interactive 7-Panel Quantitative Forecasting & Portfolio Dashboard.
Run with: streamlit run dashboard/app.py
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Northgate AI | Stock Prediction & Portfolio System",
    layout="wide",
)

st.title("Northgate AI — Market Prediction & Portfolio Optimization System")
st.caption(
    "Educational Research Platform | Strictly Leak-Free Chronological Pipeline | "
    "Not Personalized Financial Advice"
)

panel = pd.read_parquet("data/processed/features_with_sentiment.parquet")
panel["Date"] = pd.to_datetime(panel["Date"])
bench_df = pd.read_csv("reports/all_model_benchmarks.csv")
preds_df = pd.read_parquet("data/processed/test_predictions_all.parquet")
preds_df["Date"] = pd.to_datetime(preds_df["Date"])
fi_df = pd.read_csv("reports/feature_importance.csv")
weights_df = pd.read_csv("reports/portfolio_weights.csv")
frontier_df = pd.read_csv("reports/efficient_frontier.csv")
mc_df = pd.read_parquet("data/processed/monte_carlo_frontier.parquet")
bt_df = pd.read_parquet("data/processed/portfolio_backtest.parquet")
bt_df["Date"] = pd.to_datetime(bt_df["Date"])
bt_summary = pd.read_csv("reports/portfolio_backtest_summary.csv")
rec_df = pd.read_csv("reports/recommendations.csv")

# Sidebar controls
st.sidebar.header("System Controls")
tickers = sorted(panel["Ticker"].unique())
selected_ticker = st.sidebar.selectbox("Select Equity Ticker", tickers, index=0)
min_d, max_d = panel["Date"].min().date(), panel["Date"].max().date()
date_range = st.sidebar.date_input("Date Window", [min_d, max_d], min_value=min_d, max_value=max_d)

t_df = panel[panel["Ticker"] == selected_ticker].copy()
if len(date_range) == 2:
    t_df = t_df[
        (t_df["Date"].dt.date >= date_range[0]) & (t_df["Date"].dt.date <= date_range[1])
    ]

tabs = st.tabs(
    [
        "1. Market Overview",
        "2. Exploratory Analysis",
        "3. ML & DL Benchmarks",
        "4. Forecast Visualizer",
        "5. News Sentiment",
        "6. Efficient Frontier",
        "7. Buy/Hold/Sell Signals",
    ]
)

# Panel 1: Market Overview
with tabs[0]:
    st.subheader(f"{selected_ticker} — Price Action, Moving Averages & Volatility")
    latest = t_df.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest Close", f"${latest['Adj Close']:.2f}", f"{latest['ret1']*100:+.2f}%")
    c2.metric("RSI (14)", f"{latest['rsi14']:.1f}")
    c3.metric("Annualized Vol (21d)", f"{latest['vol21']*np.sqrt(252)*100:.1f}%")
    c4.metric("10Y-3M Yield Spread", f"{latest['yield_spread']:.2f}%")

    fig_px = go.Figure()
    fig_px.add_trace(
        go.Candlestick(
            x=t_df["Date"],
            open=t_df["Open"],
            high=t_df["High"],
            low=t_df["Low"],
            close=t_df["Close"],
            name="OHLC",
        )
    )
    fig_px.add_trace(go.Scatter(x=t_df["Date"], y=t_df["sma20"], name="SMA 20", line=dict(width=1.5)))
    fig_px.add_trace(go.Scatter(x=t_df["Date"], y=t_df["sma50"], name="SMA 50", line=dict(width=1.5)))
    fig_px.add_trace(go.Scatter(x=t_df["Date"], y=t_df["sma200"], name="SMA 200", line=dict(width=1.5)))
    fig_px.update_layout(height=480, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_px, use_container_width=True)

# Panel 2: EDA
with tabs[1]:
    st.subheader("Return Distribution & Cross-Asset Correlation")
    col_a, col_b = st.columns(2)
    with col_a:
        fig_hist = px.histogram(
            t_df, x="ret1", nbins=60, marginal="box", title=f"{selected_ticker} Daily Log Returns"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    with col_b:
        piv_ret = panel.pivot(index="Date", columns="Ticker", values="ret1").dropna()
        corr = piv_ret.corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="10-Stock Return Correlation Matrix"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

# Panel 3: Model Comparison
with tabs[2]:
    st.subheader("Unified Out-of-Sample Model Benchmark (AAPL Target)")
    st.dataframe(bench_df, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        fig_rmse = px.bar(
            bench_df, x="Model", y="RMSE ($)", color="Family", title="Out-of-Sample Price RMSE ($)"
        )
        st.plotly_chart(fig_rmse, use_container_width=True)
    with col2:
        fig_fi = px.bar(
            fi_df.head(15).iloc[::-1],
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 15 XGBoost Feature Importances",
        )
        st.plotly_chart(fig_fi, use_container_width=True)

# Panel 4: Forecast Visualizer
with tabs[3]:
    st.subheader("Out-of-Sample Actual vs. Predicted Close (Test Window)")
    avail_models = [c for c in preds_df.columns if c not in ("Date", "Actual_Close", "Prev_Close", "Actual_Ret")]
    chosen = st.multiselect(
        "Select Models to Overlay", avail_models, default=["Random Forest", "XGBoost", "LSTM"]
    )
    fig_fc = go.Figure()
    fig_fc.add_trace(
        go.Scatter(x=preds_df["Date"], y=preds_df["Actual_Close"], name="Actual Close", line=dict(color="black", width=2))
    )
    for m in chosen:
        fig_fc.add_trace(go.Scatter(x=preds_df["Date"], y=preds_df[m], name=m, line=dict(width=1.5)))
    fig_fc.update_layout(height=450)
    st.plotly_chart(fig_fc, use_container_width=True)

# Panel 5: Sentiment Analytics
with tabs[4]:
    st.subheader(f"{selected_ticker} — FinBERT News Sentiment vs. Price")
    fig_sent = go.Figure()
    fig_sent.add_trace(
        go.Bar(x=t_df["Date"], y=t_df["sent_mean"], name="Daily Sentiment", opacity=0.45)
    )
    fig_sent.add_trace(
        go.Scatter(x=t_df["Date"], y=t_df["sent_ewm5"], name="5-Day EWM Sentiment", line=dict(width=2))
    )
    fig_sent.update_layout(height=400)
    st.plotly_chart(fig_sent, use_container_width=True)

# Panel 6: Portfolio Optimization & Backtest
with tabs[5]:
    st.subheader("Markowitz Efficient Frontier & Out-of-Sample Backtest")
    c_left, c_right = st.columns([3, 2])
    with c_left:
        sample_mc = mc_df.sample(min(4000, len(mc_df)), random_state=42)
        fig_ef = px.scatter(
            sample_mc,
            x="Volatility",
            y="Return",
            color="Sharpe",
            color_continuous_scale="Viridis",
            opacity=0.45,
            title="Monte Carlo Cloud & Analytical Efficient Frontier",
        )
        fig_ef.add_trace(
            go.Scatter(
                x=frontier_df["Volatility"],
                y=frontier_df["Return"],
                mode="lines",
                name="Efficient Frontier",
                line=dict(color="red", width=3),
            )
        )
        st.plotly_chart(fig_ef, use_container_width=True)
    with c_right:
        st.markdown("**Optimal Asset Allocation Weights**")
        st.dataframe(weights_df, use_container_width=True)

    st.markdown("**Out-of-Sample Cumulative Wealth Growth ($1 Initial)**")
    fig_bt = px.line(bt_df, x="Date", y=["Max Sharpe (MPT)", "Min Volatility (MPT)", "Equal Weight (1/N)"])
    st.plotly_chart(fig_bt, use_container_width=True)
    st.dataframe(bt_summary, use_container_width=True)

# Panel 7: Decision Support Recommendations
with tabs[6]:
    st.subheader("Rule-Based BUY / HOLD / SELL Decision Support Table")
    st.warning(
        "EDUCATIONAL DISCLAIMER: Signals are produced by a deterministic academic model "
        "combining forecast return, FinBERT sentiment, and volatility risk. Not investment advice."
    )
    st.dataframe(rec_df, use_container_width=True)
