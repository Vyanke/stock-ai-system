"""
src/sentiment.py
Financial news sentiment pipeline using ProsusAI/finbert with NLTK VADER fallback.
Enforces strict market-close look-ahead prevention: headlines published after
16:00 ET are aligned to the next trading session before feature aggregation.
"""

import os
import warnings
import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def align_to_trading_session(
    news_df: pd.DataFrame, trading_dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Maps each headline timestamp to the earliest trading session whose close
    (16:00) occurs AFTER the headline's publication timestamp.
    """
    cal = np.sort(pd.to_datetime(trading_dates).normalize().unique())
    aligned_dates = []

    for ts in pd.to_datetime(news_df["timestamp"]):
        pub_day = ts.normalize()
        # If published at or after 16:00 (market close), shift to next calendar day
        effective_day = pub_day + pd.Timedelta(days=1) if ts.hour >= 16 else pub_day
        idx = np.searchsorted(cal, effective_day, side="left")
        if idx < len(cal):
            aligned_dates.append(cal[idx])
        else:
            aligned_dates.append(pd.NaT)

    out = news_df.copy()
    out["Date"] = aligned_dates
    return out.dropna(subset=["Date"])


def score_with_finbert(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Scores texts using ProsusAI/finbert: P(positive) - P(negative) in [-1, 1]."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_name = "ProsusAI/finbert"
    print(f"Loading {model_name} for financial sentiment scoring...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    scores = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
            ).to(device)
            probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
            # FinBERT label order: 0=positive, 1=negative, 2=neutral
            batch_scores = probs[:, 0] - probs[:, 1]
            scores.extend(batch_scores.tolist())
    return np.array(scores, dtype=float)


def score_with_vader(texts: list[str]) -> np.ndarray:
    """Fast lexical baseline using NLTK VADER compound sentiment score."""
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)

    sia = SentimentIntensityAnalyzer()
    return np.array([sia.polarity_scores(t)["compound"] for t in texts], dtype=float)


def run_sentiment():
    cfg = load_config()
    raw_dir = cfg["data"]["raw_dir"]
    proc_dir = cfg["data"]["processed_dir"]

    panel = pd.read_parquet(os.path.join(proc_dir, "features_panel.parquet"))
    panel["Date"] = pd.to_datetime(panel["Date"])
    news = pd.read_parquet(os.path.join(raw_dir, "news.parquet"))

    trading_dates = pd.DatetimeIndex(panel["Date"].unique())
    aligned_news = align_to_trading_session(news, trading_dates)

    # Deduplicate texts before scoring for speed
    unique_texts = aligned_news["text"].unique().tolist()
    print(f"Scoring {len(unique_texts)} unique headlines across {len(aligned_news)} records...")

    vader_scores = score_with_vader(unique_texts)
    vader_map = dict(zip(unique_texts, vader_scores))
    aligned_news["vader_score"] = aligned_news["text"].map(vader_map)

    try:
        finbert_scores = score_with_finbert(unique_texts)
        finbert_map = dict(zip(unique_texts, finbert_scores))
        aligned_news["finbert_score"] = aligned_news["text"].map(finbert_map)
        scorer_used = "FinBERT + VADER"
    except Exception as exc:
        print(f"FinBERT offline/download fallback triggered ({exc}); using VADER scores.")
        aligned_news["finbert_score"] = aligned_news["vader_score"]
        scorer_used = "VADER Fallback"

    # Aggregate daily per ticker: mean score, positive ratio, article volume
    daily_sent = (
        aligned_news.groupby(["Date", "ticker"])
        .agg(
            sent_mean=("finbert_score", "mean"),
            vader_mean=("vader_score", "mean"),
            sent_pos_ratio=("finbert_score", lambda s: float((s > 0.15).mean())),
            news_volume=("finbert_score", "count"),
        )
        .reset_index()
        .rename(columns={"ticker": "Ticker"})
    )

    # Merge onto full feature panel and compute exponentially weighted rolling sentiment
    merged = panel.merge(daily_sent, on=["Date", "Ticker"], how="left")
    merged["sent_mean"] = merged["sent_mean"].fillna(0.0)
    merged["vader_mean"] = merged["vader_mean"].fillna(0.0)
    merged["sent_pos_ratio"] = merged["sent_pos_ratio"].fillna(0.0)
    merged["news_volume"] = merged["news_volume"].fillna(0)

    merged = merged.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    merged["sent_ewm5"] = merged.groupby("Ticker")["sent_mean"].transform(
        lambda s: s.ewm(span=5, adjust=False).mean()
    )
    merged["sent_ewm10"] = merged.groupby("Ticker")["sent_mean"].transform(
        lambda s: s.ewm(span=10, adjust=False).mean()
    )

    out_path = os.path.join(proc_dir, "features_with_sentiment.parquet")
    merged.to_parquet(out_path, index=False)
    print(f"[{scorer_used}] Saved sentiment-enriched dataset to {out_path}")


if __name__ == "__main__":
    run_sentiment()
