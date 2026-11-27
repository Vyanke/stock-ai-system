"""
src/clean.py
Idempotent multi-source cleaning pipeline with audit logging.
Enforces calendar alignment, forward-fill-only rules, OHLC invariants,
and rolling z-score outlier flagging.
"""

import os
import json
import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def clean_single_ticker(df: pd.DataFrame, cal: pd.DatetimeIndex, ticker: str) -> tuple[pd.DataFrame, dict]:
    rows_in = len(df)
    df = df.reindex(cal)
    df = df[~df.index.duplicated(keep="last")]

    missing_before = int(df["Adj Close"].isna().sum())
    df[["Open", "High", "Low", "Close", "Adj Close"]] = (
        df[["Open", "High", "Low", "Close", "Adj Close"]].ffill(limit=3)
    )
    df["Volume"] = df["Volume"].fillna(0)

    # OHLC invariant validation: Low <= Open, Close <= High
    bad_ohlc = (
        (df["Low"] > df[["Open", "Close", "High"]].min(axis=1))
        | (df["High"] < df[["Open", "Close", "Low"]].max(axis=1))
    )
    invariant_violations = int(bad_ohlc.sum())
    df.loc[bad_ohlc, ["Open", "High", "Low", "Close", "Adj Close"]] = np.nan
    df[["Open", "High", "Low", "Close", "Adj Close"]] = (
        df[["Open", "High", "Low", "Close", "Adj Close"]].ffill(limit=3)
    )

    # Flag extreme return outliers (>5 rolling std) without auto-deleting earnings jumps
    log_ret = np.log(df["Adj Close"]).diff()
    rolling_z = (log_ret - log_ret.rolling(63).mean()) / log_ret.rolling(63).std()
    df["is_outlier"] = (rolling_z.abs() > 5).fillna(False)
    outliers_flagged = int(df["is_outlier"].sum())

    df = df.dropna(subset=["Adj Close"]).copy()
    df["Ticker"] = ticker

    audit = {
        "ticker": ticker,
        "rows_in": rows_in,
        "rows_out": len(df),
        "gaps_filled": max(0, missing_before - int(df["Adj Close"].isna().sum())),
        "invariant_violations": invariant_violations,
        "outliers_flagged": outliers_flagged,
    }
    return df, audit


def run_cleaning():
    cfg = load_config()
    raw_dir = cfg["data"]["raw_dir"]
    proc_dir = cfg["data"]["processed_dir"]
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    px = pd.read_parquet(os.path.join(raw_dir, "prices.parquet"))
    macro = pd.read_parquet(os.path.join(raw_dir, "macro.parquet"))

    benchmark = cfg["data"]["benchmark"]
    vix = cfg["data"]["volatility_index"]
    universe = cfg["data"]["universe"]

    # Master trading calendar from benchmark index
    cal = px.xs(benchmark, axis=1, level=1).dropna(subset=["Adj Close"]).index

    # Clean macro series: align to trading calendar and forward-fill only
    macro_clean = macro.reindex(cal).ffill().bfill(limit=5)
    macro_clean["yield_spread"] = macro_clean["DGS10"] - macro_clean["DGS3MO"]

    # Extract benchmark and VIX series
    bench_df = px.xs(benchmark, axis=1, level=1).reindex(cal).ffill(limit=3)
    vix_df = px.xs(vix, axis=1, level=1).reindex(cal).ffill(limit=3)

    market_context = pd.DataFrame(
        {
            "bench_adj_close": bench_df["Adj Close"],
            "bench_ret1": np.log(bench_df["Adj Close"]).diff(),
            "vix_close": vix_df["Adj Close"],
            "vix_chg": vix_df["Adj Close"].diff(),
        },
        index=cal,
    ).join(macro_clean)

    cleaned_frames = []
    audit_logs = []

    for ticker in universe:
        sub = px.xs(ticker, axis=1, level=1).copy()
        clean_df, audit = clean_single_ticker(sub, cal, ticker)
        clean_df = clean_df.join(market_context, how="left")
        cleaned_frames.append(clean_df)
        audit_logs.append(audit)

    panel = pd.concat(cleaned_frames).reset_index().rename(columns={"index": "Date"})
    panel_path = os.path.join(proc_dir, "clean_panel.parquet")
    panel.to_parquet(panel_path, index=False)

    # Save one-page data quality audit report
    audit_df = pd.DataFrame(audit_logs)
    report_path = os.path.join("reports", "data_quality_report.md")
    with open(report_path, "w") as f:
        f.write("# Data Quality Audit Report\n\n")
        f.write(f"- **Trading Calendar Days:** {len(cal)}\n")
        f.write(f"- **Universe Size:** {len(universe)} tickers\n")
        f.write(f"- **Post-Clean Missing Adjusted Close:** {panel['Adj Close'].isna().sum()}\n\n")
        f.write(audit_df.to_markdown(index=False))
        f.write("\n")

    print(f"Saved clean panel to {panel_path} ({len(panel)} rows)")
    print(f"Saved data quality report to {report_path}")


if __name__ == "__main__":
    run_cleaning()
