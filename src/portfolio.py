"""
src/portfolio.py
Modern Portfolio Theory (Markowitz Mean-Variance Optimization):
- Ledoit-Wolf shrinkage covariance estimator
- Analytical Efficient Frontier (Min-Vol, Max-Sharpe, Equal-Weight benchmark)
- 20,000 Monte Carlo random portfolios
- Out-of-sample walk-forward portfolio backtest with transaction costs
"""

import os
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from math_utils import (
    annualised_return,
    annualised_volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    historical_var_95,
)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def portfolio_perf(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float):
    ret = float(np.dot(weights, mu))
    vol = float(np.sqrt(weights.T @ cov @ weights))
    sharpe = (ret - rf) / (vol + 1e-12)
    return ret, vol, sharpe


def optimize_max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float, max_w: float) -> np.ndarray:
    n = len(mu)
    init_w = np.ones(n) / n
    bounds = tuple((0.0, max_w) for _ in range(n))
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    res = minimize(
        lambda w: -portfolio_perf(w, mu, cov, rf)[2],
        init_w,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return res.x if res.success else init_w


def optimize_min_vol(mu: np.ndarray, cov: np.ndarray, rf: float, max_w: float) -> np.ndarray:
    n = len(mu)
    init_w = np.ones(n) / n
    bounds = tuple((0.0, max_w) for _ in range(n))
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    res = minimize(
        lambda w: portfolio_perf(w, mu, cov, rf)[1],
        init_w,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return res.x if res.success else init_w


def build_efficient_frontier(mu: np.ndarray, cov: np.ndarray, max_w: float, n_pts: int = 40) -> pd.DataFrame:
    n = len(mu)
    init_w = np.ones(n) / n
    bounds = tuple((0.0, max_w) for _ in range(n))
    target_rets = np.linspace(float(mu.min()) * 0.9, float(mu.max()) * 0.95, n_pts)
    frontier_rows = []

    for tr in target_rets:
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=tr: np.dot(w, mu) - t},
        )
        res = minimize(
            lambda w: float(np.sqrt(w.T @ cov @ w)),
            init_w,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if res.success:
            r, v, s = portfolio_perf(res.x, mu, cov, 0.04)
            frontier_rows.append({"Return": r, "Volatility": v, "Sharpe": s})

    return pd.DataFrame(frontier_rows)


def run_portfolio_optimization():
    cfg = load_config()
    proc_dir = cfg["data"]["processed_dir"]
    rf = cfg["portfolio"]["risk_free_rate"]
    max_w = cfg["portfolio"]["max_weight_per_asset"]
    n_mc = cfg["portfolio"]["monte_carlo_portfolios"]
    seed = cfg["project"]["seed"]

    panel = pd.read_parquet(os.path.join(proc_dir, "clean_panel.parquet"))
    panel["Date"] = pd.to_datetime(panel["Date"])

    piv = panel.pivot(index="Date", columns="Ticker", values="Adj Close").sort_index()
    rets = np.log(piv / piv.shift(1)).dropna()
    tickers = list(rets.columns)

    n = len(rets)
    n_train_val = int(n * (cfg["splits"]["train_ratio"] + cfg["splits"]["val_ratio"]))
    in_sample = rets.iloc[:n_train_val]
    out_sample = rets.iloc[n_train_val:]

    # Annualized expected returns & Ledoit-Wolf shrinkage covariance on in-sample split
    mu_ann = in_sample.mean().values * 252.0
    lw = LedoitWolf().fit(in_sample.values)
    cov_ann = lw.covariance_ * 252.0

    w_max_sharpe = optimize_max_sharpe(mu_ann, cov_ann, rf, max_w)
    w_min_vol = optimize_min_vol(mu_ann, cov_ann, rf, max_w)
    w_equal = np.ones(len(tickers)) / len(tickers)

    weights_df = pd.DataFrame(
        {
            "Ticker": tickers,
            "Max_Sharpe_Weight": np.round(w_max_sharpe, 4),
            "Min_Vol_Weight": np.round(w_min_vol, 4),
            "Equal_Weight": np.round(w_equal, 4),
        }
    )
    weights_df.to_csv("reports/portfolio_weights.csv", index=False)

    # Monte Carlo simulation (20,000 feasible random portfolios)
    rng = np.random.default_rng(seed)
    mc_W = rng.dirichlet(np.ones(len(tickers)), size=n_mc)
    mc_rets = mc_W @ mu_ann
    mc_vols = np.sqrt(np.einsum("ij,jk,ik->i", mc_W, cov_ann, mc_W))
    mc_sharpe = (mc_rets - rf) / (mc_vols + 1e-12)
    mc_df = pd.DataFrame({"Return": mc_rets, "Volatility": mc_vols, "Sharpe": mc_sharpe})
    mc_df.to_parquet(os.path.join(proc_dir, "monte_carlo_frontier.parquet"), index=False)

    frontier_df = build_efficient_frontier(mu_ann, cov_ann, max_w)
    frontier_df.to_csv("reports/efficient_frontier.csv", index=False)

    # Out-of-sample backtest (10 bps turnover cost applied at quarterly rebalance)
    portfolios = {
        "Max Sharpe (MPT)": w_max_sharpe,
        "Min Volatility (MPT)": w_min_vol,
        "Equal Weight (1/N)": w_equal,
    }
    bt_curves = pd.DataFrame(index=out_sample.index)
    summary_rows = []

    for label, w in portfolios.items():
        daily_r = out_sample.values @ w
        # Apply 10 bps transaction cost every 63 trading days
        cost = np.zeros_like(daily_r)
        cost[::63] = 0.0010
        net_r = daily_r - cost
        bt_curves[label] = np.cumprod(1.0 + net_r)

        summary_rows.append(
            {
                "Portfolio": label,
                "Ann. Return (%)": round(annualised_return(net_r) * 100, 2),
                "Ann. Volatility (%)": round(annualised_volatility(net_r) * 100, 2),
                "Sharpe Ratio": round(sharpe_ratio(net_r, rf), 3),
                "Sortino Ratio": round(sortino_ratio(net_r, rf), 3),
                "Max Drawdown (%)": round(max_drawdown(net_r) * 100, 2),
                "Calmar Ratio": round(calmar_ratio(net_r), 3),
                "Daily 95% VaR (%)": round(historical_var_95(net_r) * 100, 3),
            }
        )

    bt_curves.reset_index().to_parquet(
        os.path.join(proc_dir, "portfolio_backtest.parquet"), index=False
    )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("reports/portfolio_backtest_summary.csv", index=False)

    print("\n=== Optimal Portfolio Weights ===")
    print(weights_df.to_string(index=False))
    print("\n=== Out-of-Sample Portfolio Backtest ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    run_portfolio_optimization()
