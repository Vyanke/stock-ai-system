"""
src/models_ml.py
Classical ML & Naive Random-Walk Baseline benchmarking pipeline.
Strictly fits scalers on training split only to prevent data leakage.
"""

import os
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


EXCLUDE_COLS = {
    "Date",
    "Ticker",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "is_outlier",
    "target_ret1",
    "target_next_close",
    "bench_adj_close",
    "sma10",
    "sma20",
    "sma50",
    "sma200",
}


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_metrics(
    y_true_ret: np.ndarray,
    y_pred_ret: np.ndarray,
    prev_close: np.ndarray,
    next_close_true: np.ndarray,
) -> dict:
    # Price-level reconstruction: P_hat_{t+1} = P_t * exp(r_hat_{t+1})
    next_close_pred = prev_close * np.exp(y_pred_ret)

    rmse_price = float(np.sqrt(mean_squared_error(next_close_true, next_close_pred)))
    mae_price = float(mean_absolute_error(next_close_true, next_close_pred))
    mape_price = float(
        np.mean(np.abs((next_close_true - next_close_pred) / (next_close_true + 1e-9))) * 100.0
    )
    r2_ret = float(r2_score(y_true_ret, y_pred_ret))

    # Directional Accuracy: sign(predicted return) == sign(actual return)
    dir_acc = float(np.mean(np.sign(y_pred_ret) == np.sign(y_true_ret)) * 100.0)

    # Simulated directional long/short strategy Sharpe ratio (annualized)
    strat_ret = np.sign(y_pred_ret) * y_true_ret
    sharpe = float(
        (np.mean(strat_ret) * 252) / (np.std(strat_ret) * np.sqrt(252) + 1e-9)
    )

    return {
        "RMSE ($)": round(rmse_price, 4),
        "MAE ($)": round(mae_price, 4),
        "MAPE (%)": round(mape_price, 4),
        "Return R2": round(r2_ret, 4),
        "Directional Acc (%)": round(dir_acc, 2),
        "Strategy Sharpe": round(sharpe, 3),
    }


def run_ml_models():
    cfg = load_config()
    proc_dir = cfg["data"]["processed_dir"]
    target_ticker = cfg["features"]["target_ticker"]
    seed = cfg["project"]["seed"]

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    df = pd.read_parquet(os.path.join(proc_dir, "features_with_sentiment.parquet"))
    df = df[df["Ticker"] == target_ticker].sort_values("Date").reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].values
    y = df["target_ret1"].values
    prev_close = df["Adj Close"].values
    next_close = df["target_next_close"].values
    dates = df["Date"].values

    n = len(df)
    n_train = int(n * cfg["splits"]["train_ratio"])
    n_val = int(n * cfg["splits"]["val_ratio"])

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]

    prev_test = prev_close[n_train + n_val :]
    next_test = next_close[n_train + n_val :]
    dates_test = dates[n_train + n_val :]

    # Fit scaler strictly on training data only
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    joblib.dump({"scaler": scaler, "feature_cols": feature_cols}, "models/ml_scaler.joblib")

    models = {
        "Ridge Regression": Ridge(alpha=10.0, random_state=seed),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=10, random_state=seed, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            verbosity=0,
        ),
        "SVR (RBF)": SVR(C=1.0, epsilon=0.005, kernel="rbf"),
    }

    results = []
    pred_df = pd.DataFrame(
        {
            "Date": dates_test,
            "Actual_Close": next_test,
            "Prev_Close": prev_test,
            "Actual_Ret": y_test,
        }
    )

    # 1. Naive Random Walk Baseline: predicts 0 log return (P_hat_{t+1} = P_t)
    rw_pred_ret = np.zeros_like(y_test)
    rw_metrics = compute_metrics(y_test, rw_pred_ret, prev_test, next_test)
    rw_metrics["Model"] = "Naive Random Walk"
    rw_metrics["Family"] = "Baseline"
    results.append(rw_metrics)
    pred_df["Naive Random Walk"] = prev_test

    # 2. Train & evaluate each classical ML model
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred_test = model.predict(X_test_s)

        metrics = compute_metrics(y_test, y_pred_test, prev_test, next_test)
        metrics["Model"] = name
        metrics["Family"] = "Classical ML"
        results.append(metrics)

        pred_df[name] = prev_test * np.exp(y_pred_test)
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(model, f"models/{safe_name}.joblib")

        # Save XGBoost feature importances for dashboard & report
        if name == "XGBoost":
            fi = pd.DataFrame(
                {"Feature": feature_cols, "Importance": model.feature_importances_}
            ).sort_values("Importance", ascending=False)
            fi.to_csv("reports/feature_importance.csv", index=False)

    res_df = pd.DataFrame(results)[
        [
            "Model",
            "Family",
            "RMSE ($)",
            "MAE ($)",
            "MAPE (%)",
            "Return R2",
            "Directional Acc (%)",
            "Strategy Sharpe",
        ]
    ]
    res_df.to_csv("reports/ml_benchmarks.csv", index=False)
    pred_df.to_parquet(os.path.join(proc_dir, "test_predictions_ml.parquet"), index=False)

    print(f"\n=== Classical ML Test Benchmarks ({target_ticker}) ===")
    print(res_df.to_string(index=False))


if __name__ == "__main__":
    run_ml_models()
