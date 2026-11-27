# Northgate AI — Stock Market Prediction & Portfolio Optimization System
**Capstone Technical Report**

## 1. Executive Summary
This project implements an end-to-end, leak-free quantitative financial forecasting and portfolio allocation system across a 10-stock diversified US equity universe (`AAPL`, `MSFT`, `JPM`, `XOM`, `JNJ`, `PG`, `NVDA`, `KO`, `CAT`, `HD`) over 2015–2026. The system combines stationary technical and macroeconomic feature engineering, ProsusAI/FinBERT financial news sentiment analysis, classical Machine Learning (Ridge, Random Forest, XGBoost, SVR), PyTorch Deep Learning sequence models (LSTM, GRU, BiLSTM, Transformer), and Markowitz Mean-Variance portfolio optimization.

## 2. Data Pipeline & Leakage Prevention
- **Sources:** Adjusted OHLCV equity prices, S&P 500 (`^GSPC`), CBOE Volatility Index (`^VIX`), FRED macroeconomic series (`DGS10`, `DGS3MO`, `CPIAUCSL`, `UNRATE`), and timestamped financial headlines.
- **Anti-Leakage Controls:**
  1. Chronological `70% / 15% / 15%` train/validation/test partitioning with zero random shuffling.
  2. All feature scalers (`StandardScaler`) and covariance estimators (`LedoitWolf`) are fit strictly on the training split.
  3. News headlines published at or after 16:00 ET are shifted forward to the next trading session before aggregating daily sentiment.

## 3. Exploratory Data Analysis & Stylized Facts
- **Non-Stationarity:** Raw adjusted close prices fail the Augmented Dickey-Fuller (ADF) unit-root test ($p > 0.05$), whereas daily log returns $r_t = \ln(P_t / P_{t-1})$ are stationary ($p < 0.001$).
- **Fat Tails & Volatility Clustering:** Daily equity returns exhibit excess kurtosis $> 3$ (leptokurtic tails) and strong autocorrelation in squared returns $r_t^2$.

## 4. Feature Engineering & Mathematical Foundations
Engineered 30+ stationary indicators (multi-horizon returns, price-to-SMA ratios, MACD, RSI-14, ATR-14, Bollinger Band width, volume z-scores, yield curve spread, and 5/10-day EWM sentiment). Core quantitative metrics (sample covariance, CAPM Beta, Sharpe, Sortino, Maximum Drawdown, 95% Historical VaR) and gradient-descent linear regression were implemented from scratch in `src/math_utils.py` and verified against NumPy/OLS to machine precision.

## 5. Model Benchmarking & Critical Interpretation
Evaluated on out-of-sample test data (`AAPL` target) against a Naive Random Walk ($r_{t+1} = 0$):

| Model | Family | RMSE ($) | MAE ($) | MAPE (%) | Return R² | Directional Acc (%) | Strategy Sharpe |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Naive Random Walk | Baseline | 4.5654 | 3.1350 | 1.2285 | 0.0025 | 0.00 | 0.000 |
| Ridge Regression | Classical ML | 4.7395 | 3.3656 | 1.3073 | -0.0566 | 47.96 | -0.241 |
| Random Forest | Classical ML | **4.5216** | **3.1194** | **1.2236** | **0.0188** | 54.44 | 1.398 |
| XGBoost | Classical ML | 4.6237 | 3.1776 | 1.2458 | -0.0341 | **56.83** | **1.533** |
| SVR (RBF) | Classical ML | 6.0387 | 4.6018 | 1.8063 | -0.7055 | 50.36 | 0.108 |
| LSTM | Deep Learning | 4.5592 | 3.1393 | 1.2303 | -0.0016 | 49.88 | 0.346 |
| GRU | Deep Learning | 4.6101 | 3.1849 | 1.2453 | -0.0198 | 48.20 | -0.622 |
| BiLSTM | Deep Learning | 4.6021 | 3.1760 | 1.2419 | -0.0163 | 46.28 | -0.758 |
| Transformer | Deep Learning | 10.2036 | 8.8853 | 3.3082 | -3.7186 | 46.52 | -0.788 |

**Key Takeaways:** Consistent with the Efficient Market Hypothesis, next-day return $R^2$ values remain close to zero. Tree ensembles (**Random Forest** and **XGBoost**) outperform high-capacity Transformers on tabular daily financial sequences because self-attention overfits noise on ~2,000 daily observations, whereas regularized decision trees capture non-linear regime interactions (`56.83%` directional accuracy).

## 6. Portfolio Optimization & Risk-Adjusted Performance
Using Ledoit-Wolf shrinkage covariance on in-sample returns with a `30%` maximum single-asset cap, `src/portfolio.py` constructs the Markowitz Efficient Frontier, simulates 20,000 Monte Carlo portfolios, and evaluates out-of-sample cumulative performance net of `10 bps` quarterly rebalancing transaction costs (full tables exported in `reports/portfolio_weights.csv` and `reports/portfolio_backtest_summary.csv`).

## 7. Limitations & Disclaimer
- Historical backtests do not account for market impact, slippage, or regime shifts outside the 2015–2026 sample.
- **Disclaimer:** This system is developed strictly for educational and research purposes and does not constitute personalized investment advice.
