"""
src/math_utils.py
From-scratch implementations of core quantitative finance mathematics
and manual gradient-descent linear regression, verified against NumPy/SciPy.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def simple_returns(prices: np.ndarray) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)
    return (prices[1:] - prices[:-1]) / prices[:-1]


def log_returns(prices: np.ndarray) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)
    return np.log(prices[1:] / prices[:-1])


def annualised_return(daily_returns: np.ndarray) -> float:
    return float(np.mean(daily_returns) * TRADING_DAYS)


def annualised_volatility(daily_returns: np.ndarray) -> float:
    variance = np.mean((daily_returns - np.mean(daily_returns)) ** 2)
    return float(np.sqrt(variance) * np.sqrt(TRADING_DAYS))


def manual_covariance_matrix(returns_matrix: np.ndarray) -> np.ndarray:
    """Computes sample covariance matrix from scratch: E[(r_a - mu_a)(r_b - mu_b)]."""
    X = np.asarray(returns_matrix, dtype=float)
    n = X.shape[0]
    centered = X - np.mean(X, axis=0, keepdims=True)
    return (centered.T @ centered) / (n - 1)


def compute_beta(asset_returns: np.ndarray, market_returns: np.ndarray) -> float:
    """Beta_i = Cov(r_i, r_m) / Var(r_m)."""
    cov = np.mean((asset_returns - np.mean(asset_returns)) * (market_returns - np.mean(market_returns)))
    var_m = np.mean((market_returns - np.mean(market_returns)) ** 2)
    return float(cov / (var_m + 1e-12))


def sharpe_ratio(daily_returns: np.ndarray, rf_annual: float = 0.04) -> float:
    mu_ann = annualised_return(daily_returns)
    vol_ann = annualised_volatility(daily_returns)
    return float((mu_ann - rf_annual) / (vol_ann + 1e-12))


def sortino_ratio(daily_returns: np.ndarray, rf_annual: float = 0.04) -> float:
    mu_ann = annualised_return(daily_returns)
    rf_daily = rf_annual / TRADING_DAYS
    downside = np.minimum(0.0, daily_returns - rf_daily)
    downside_vol = np.sqrt(np.mean(downside ** 2)) * np.sqrt(TRADING_DAYS)
    return float((mu_ann - rf_annual) / (downside_vol + 1e-12))


def max_drawdown(daily_returns: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + daily_returns)
    peaks = np.maximum.accumulate(wealth)
    drawdowns = (wealth / peaks) - 1.0
    return float(np.min(drawdowns))


def calmar_ratio(daily_returns: np.ndarray) -> float:
    mdd = abs(max_drawdown(daily_returns))
    return float(annualised_return(daily_returns) / (mdd + 1e-12))


def historical_var_95(daily_returns: np.ndarray) -> float:
    return float(np.percentile(daily_returns, 5))


def gd_linreg(X: np.ndarray, y: np.ndarray, lr: float = 0.05, epochs: int = 3000):
    """From-scratch gradient descent for linear regression."""
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    b = 0.0
    for _ in range(epochs):
        yhat = X @ w + b
        err = yhat - y
        w -= lr * (2.0 / n) * (X.T @ err)
        b -= lr * (2.0 / n) * float(np.sum(err))
    return w, b


def verify_implementations():
    rng = np.random.default_rng(42)
    R = rng.normal(0.0005, 0.015, size=(500, 3))

    # 1. Verify covariance matrix against numpy.cov
    cov_manual = manual_covariance_matrix(R)
    cov_np = np.cov(R, rowvar=False)
    assert np.allclose(cov_manual, cov_np, atol=1e-10), "Covariance mismatch!"

    # 2. Verify gradient descent against closed-form OLS
    X = rng.normal(size=(300, 4))
    true_w = np.array([1.5, -2.0, 0.7, 3.1])
    y = X @ true_w + 0.5
    w_gd, b_gd = gd_linreg(X, y, lr=0.05, epochs=3000)
    X_aug = np.column_stack([X, np.ones(len(X))])
    ols_params = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    assert np.allclose(w_gd, ols_params[:-1], atol=1e-5) and np.isclose(b_gd, ols_params[-1], atol=1e-5)
    print("Verification passed: Manual covariance & gradient descent match analytical solutions.")


if __name__ == "__main__":
    verify_implementations()
