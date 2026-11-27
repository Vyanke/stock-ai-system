"""
src/recommend.py
Transparent Rule-Based Decision Support Engine (BUY / HOLD / SELL).
Combines normalized forecast return, FinBERT sentiment, and volatility risk
with explicit human-readable rationales and mandatory financial disclaimer.
"""

import os
import joblib
import numpy as np
import pandas as pd
import yaml

from models_ml import EXCLUDE_COLS

DISCLAIMER = (
    "DISCLAIMER: Generated for educational and research purposes only. "
    "Not personalized financial or investment advice."
)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_recommendations():
    cfg = load_config()
    proc_dir = cfg["data"]["processed_dir"]
    w_cfg = cfg["recommendations"]["weights"]
    buy_th = cfg["recommendations"]["buy_threshold"]
    sell_th = cfg["recommendations"]["sell_threshold"]

    df = pd.read_parquet(os.path.join(proc_dir, "features_with_sentiment.parquet"))
    df["Date"] = pd.to_datetime(df["Date"])

    bundle = joblib.load("models/ml_scaler.joblib")
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_cols"]
    xgb_model = joblib.load("models/xgboost.joblib")

    rows = []
    for ticker, grp in df.groupby("Ticker"):
        grp = grp.sort_values("Date").reset_index(drop=True)
        latest = grp.iloc[-1]

        X_lat = scaler.transform(grp[feature_cols].iloc[[-1]].values)
        pred_ret = float(xgb_model.predict(X_lat)[0])
        pred_pct = pred_ret * 100.0

        # Normalize components to [-1, +1]
        f_score = float(np.clip(pred_ret / 0.015, -1.0, 1.0))
        s_score = float(np.clip(latest["sent_ewm5"], -1.0, 1.0))

        # Low volatility + neutral RSI boosts risk score; high vol / overbought penalizes
        vol_pct = float(latest["vol21"]) * np.sqrt(252) * 100.0
        rsi_val = float(latest["rsi14"])
        r_score = float(np.clip(1.0 - (vol_pct / 45.0), -1.0, 1.0))
        if rsi_val > 72:
            r_score -= 0.25
        elif rsi_val < 30:
            r_score += 0.15
        r_score = float(np.clip(r_score, -1.0, 1.0))

        composite = (
            w_cfg["forecast"] * f_score
            + w_cfg["sentiment"] * s_score
            + w_cfg["risk"] * r_score
        )
        confidence = round(min(abs(composite) * 100.0 + 35.0, 95.0), 1)

        if composite > buy_th:
            action = "BUY"
        elif composite < sell_th:
            action = "SELL"
        else:
            action = "HOLD"

        rationale = (
            f"1-day model forecast {pred_pct:+.2f}%, 5-day EWM FinBERT sentiment "
            f"{latest['sent_ewm5']:+.2f}, annualized volatility {vol_pct:.1f}%, RSI(14) {rsi_val:.1f}."
        )

        rows.append(
            {
                "Ticker": ticker,
                "As_Of_Date": str(latest["Date"].date()),
                "Last_Close ($)": round(float(latest["Adj Close"]), 2),
                "Forecast_Ret (%)": round(pred_pct, 3),
                "Sentiment_EWM5": round(float(latest["sent_ewm5"]), 3),
                "Ann_Vol (%)": round(vol_pct, 1),
                "Composite_Score": round(composite, 3),
                "Action": action,
                "Confidence (%)": confidence,
                "Rationale": rationale,
            }
        )

    rec_df = pd.DataFrame(rows).sort_values("Composite_Score", ascending=False)
    rec_df.to_csv("reports/recommendations.csv", index=False)

    print("\n=== Northgate AI Decision Support Signals ===")
    print(rec_df[["Ticker", "Last_Close ($)", "Forecast_Ret (%)", "Sentiment_EWM5", "Composite_Score", "Action", "Confidence (%)"]].to_string(index=False))
    print(f"\n{DISCLAIMER}")


if __name__ == "__main__":
    generate_recommendations()
