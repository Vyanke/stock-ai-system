"""
src/features.py
Leak-free feature engineering and sliding-window sequence builder.
Every feature at row t uses strictly information available at or before t.
"""

import os
import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_ticker_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").copy()
    p = df["Adj Close"]
    h = df["High"]
    l = df["Low"]
    v = df["Volume"]

    # 1. Stationary returns
    df["ret1"] = np.log(p).diff(1)
    df["ret5"] = np.log(p).diff(5)
    df["ret10"] = np.log(p).diff(10)
    df["ret21"] = np.log(p).diff(21)

    # 2. Trend indicators (SMA, EMA, price-to-MA ratios, MACD)
    for w in (10, 20, 50, 200):
        df[f"sma{w}"] = p.rolling(w).mean()
        df[f"p_sma{w}"] = p / df[f"sma{w}"]

    ema12 = p.ewm(span=12, adjust=False).mean()
    ema26 = p.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # 3. Momentum: RSI(14) & Rate of Change
    delta = p.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-9)
    df["rsi14"] = 100.0 - (100.0 / (1.0 + rs))
    df["roc10"] = p.pct_change(10)

    # 4. Volatility: Rolling std, ATR(14), Bollinger Band Width
    df["vol21"] = df["ret1"].rolling(21).std()
    tr = pd.concat(
        [(h - l), (h - p.shift(1)).abs(), (l - p.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    df["atr14"] = tr.rolling(14).mean() / p
    mid = p.rolling(20).mean()
    sd = p.rolling(20).std()
    df["bb_width"] = (4.0 * sd) / (mid + 1e-9)

    # 5. Volume z-score
    vol_mean = v.rolling(20).mean()
    vol_std = v.rolling(20).std() + 1e-9
    df["vol_zscore"] = (v - vol_mean) / vol_std

    # 6. Calendar features
    dt_idx = pd.to_datetime(df["Date"])
    df["dow"] = dt_idx.dt.dayofweek
    df["month"] = dt_idx.dt.month

    # 7. Lagged log returns (autoregressive context)
    for lag in (1, 2, 3, 5):
        df[f"ret1_lag{lag}"] = df["ret1"].shift(lag)

    # Target: next-day log return (strictly shifted forward by 1 for supervised target)
    df["target_ret1"] = df["ret1"].shift(-1)
    df["target_next_close"] = p.shift(-1)

    return df.dropna().reset_index(drop=True)


def make_sequences(X: np.ndarray, y: np.ndarray, L: int = 60, horizon: int = 1):
    """Builds non-overlapping-boundary sliding windows of shape (samples, L, n_features)."""
    Xs, ys = [], []
    for i in range(len(X) - L - horizon + 1):
        Xs.append(X[i : i + L])
        ys.append(y[i + L + horizon - 1])
    return np.array(Xs), np.array(ys)


def build_feature_panel():
    cfg = load_config()
    proc_dir = cfg["data"]["processed_dir"]
    panel = pd.read_parquet(os.path.join(proc_dir, "clean_panel.parquet"))

    feat_frames = []
    for ticker, grp in panel.groupby("Ticker"):
        feat_frames.append(compute_ticker_features(grp))

    full_features = pd.concat(feat_frames, ignore_index=True)
    out_path = os.path.join(proc_dir, "features_panel.parquet")
    full_features.to_parquet(out_path, index=False)
    print(f"Saved engineered features to {out_path} ({full_features.shape[0]} rows, {full_features.shape[1]} cols)")


if __name__ == "__main__":
    build_feature_panel()
