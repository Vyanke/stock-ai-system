"""
src/models_dl.py
PyTorch Deep Learning sequence models:
1. Vanilla LSTM
2. GRU
3. Bidirectional LSTM
4. Transformer Encoder (Multi-Head Self-Attention)
Trained with HuberLoss, Adam, ReduceLROnPlateau, and EarlyStopping.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import yaml

from features import make_sequences
from models_ml import EXCLUDE_COLS, compute_metrics


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_dim, 32, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out[:, -1, :])
        return self.fc(out).squeeze(-1)


class GRUModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.gru1 = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.gru2 = nn.GRU(hidden_dim, 32, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.gru1(x)
        out = self.drop1(out)
        out, _ = self.gru2(out)
        out = self.drop2(out[:, -1, :])
        return self.fc(out).squeeze(-1)


class BiLSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 48, dropout: float = 0.2):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.bilstm(x)
        out = self.drop(out[:, -1, :])
        out = self.relu(self.fc1(out))
        return self.fc2(out).squeeze(-1)


class TransformerForecaster(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, 120, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        seq_len = x.size(1)
        h = self.input_proj(x) + self.pos_emb[:, :seq_len, :]
        h = self.encoder(h)
        return self.fc(h[:, -1, :]).squeeze(-1)


def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    X_test_t: torch.Tensor,
    device: torch.device,
    epochs: int = 25,
    lr: float = 1e-3,
    patience: int = 5,
) -> tuple[np.ndarray, list[float], list[float]]:
    model.to(device)
    criterion = nn.HuberLoss(delta=0.02)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    best_state = None
    wait = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        t_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item() * len(bx)
        t_loss /= len(train_loader.dataset)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                v_loss += criterion(model(bx), by).item() * len(bx)
        v_loss /= len(val_loader.dataset)

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        scheduler.step(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        preds = model(X_test_t.to(device)).cpu().numpy()

    return preds, train_losses, val_losses


def run_dl_models():
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    proc_dir = cfg["data"]["processed_dir"]
    target_ticker = cfg["features"]["target_ticker"]
    L = cfg["features"]["sequence_length"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_parquet(os.path.join(proc_dir, "features_with_sentiment.parquet"))
    df = df[df["Ticker"] == target_ticker].sort_values("Date").reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X_raw = df[feature_cols].values
    y_raw = df["target_ret1"].values
    prev_close = df["Adj Close"].values
    next_close = df["target_next_close"].values
    dates = df["Date"].values

    n = len(df)
    n_train = int(n * cfg["splits"]["train_ratio"])
    n_val = int(n * cfg["splits"]["val_ratio"])

    # Fit scaler strictly on training split before windowing
    mean = X_raw[:n_train].mean(axis=0, keepdims=True)
    std = X_raw[:n_train].std(axis=0, keepdims=True) + 1e-8
    X_scaled = (X_raw - mean) / std

    X_seq, y_seq = make_sequences(X_scaled, y_raw, L=L, horizon=1)
    prev_seq = prev_close[L:]
    next_seq = next_close[L:]
    dates_seq = dates[L:]

    # Align sequence boundaries to chronological splits
    tr_end = n_train - L
    va_end = n_train + n_val - L

    X_tr, y_tr = X_seq[:tr_end], y_seq[:tr_end]
    X_va, y_va = X_seq[tr_end:va_end], y_seq[tr_end:va_end]
    X_te, y_te = X_seq[va_end:], y_seq[va_end:]
    prev_te = prev_seq[va_end:]
    next_te = next_seq[va_end:]
    dates_te = dates_seq[va_end:]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)),
        batch_size=32,
        shuffle=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_va, dtype=torch.float32), torch.tensor(y_va, dtype=torch.float32)),
        batch_size=32,
        shuffle=False,
    )
    X_test_t = torch.tensor(X_te, dtype=torch.float32)

    input_dim = X_tr.shape[-1]
    architectures = {
        "LSTM": LSTMModel(input_dim),
        "GRU": GRUModel(input_dim),
        "BiLSTM": BiLSTMModel(input_dim),
        "Transformer": TransformerForecaster(input_dim),
    }

    dl_results = []
    pred_df = pd.read_parquet(os.path.join(proc_dir, "test_predictions_ml.parquet"))
    # Align ML predictions to sequence test dates
    pred_df = pred_df[pred_df["Date"].isin(dates_te)].reset_index(drop=True)

    history_rows = []
    for name, net in architectures.items():
        print(f"Training PyTorch {name} on {device}...")
        preds_ret, tr_hist, va_hist = train_and_evaluate(
            net, train_loader, val_loader, X_test_t, device
        )
        metrics = compute_metrics(y_te, preds_ret, prev_te, next_te)
        metrics["Model"] = name
        metrics["Family"] = "Deep Learning"
        dl_results.append(metrics)

        if len(preds_ret) == len(pred_df):
            pred_df[name] = prev_te * np.exp(preds_ret)

        torch.save(net.state_dict(), f"models/{name.lower()}.pt")
        for ep, (tl, vl) in enumerate(zip(tr_hist, va_hist), start=1):
            history_rows.append({"Model": name, "Epoch": ep, "Train_Loss": tl, "Val_Loss": vl})

    dl_df = pd.DataFrame(dl_results)
    ml_df = pd.read_csv("reports/ml_benchmarks.csv")
    full_bench = pd.concat([ml_df, dl_df], ignore_index=True)[
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
    full_bench.to_csv("reports/all_model_benchmarks.csv", index=False)
    pd.DataFrame(history_rows).to_csv("reports/dl_training_history.csv", index=False)
    pred_df.to_parquet(os.path.join(proc_dir, "test_predictions_all.parquet"), index=False)

    print(f"\n=== Unified ML + DL Benchmark Table ({target_ticker}) ===")
    print(full_bench.to_string(index=False))


if __name__ == "__main__":
    run_dl_models()
