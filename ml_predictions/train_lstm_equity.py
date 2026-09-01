"""LSTM for next-day directional prediction of Chilean equity returns.

Reads the chile_equity_features view already materialized in DuckDB by
etl/build_duckdb.py (log_return, sma_20, realized_vol_20d over ECH, the
documented proxy for ^IPSA — see etl/fetch_chile_equity.py).
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "chile_fintech.duckdb"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
SEQ_LEN = 20
EPOCHS = 30


class DirectionalLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def build_sequences(arr: np.ndarray, targets: np.ndarray, seq_len: int):
    xs, ys = [], []
    for i in range(len(arr) - seq_len):
        xs.append(arr[i : i + seq_len])
        ys.append(targets[i + seq_len])
    return np.stack(xs), np.array(ys)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(
        "SELECT fecha, log_return, sma_20, realized_vol_20d FROM chile_equity_features "
        "WHERE log_return IS NOT NULL AND realized_vol_20d IS NOT NULL ORDER BY fecha"
    ).fetchdf()
    con.close()

    features = df[["log_return", "sma_20", "realized_vol_20d"]].to_numpy(dtype=np.float32)
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
    direction = (df["log_return"].shift(-1) > 0).astype(np.float32).to_numpy()

    X, y = build_sequences(features, direction, SEQ_LEN)
    X, y = X[:-1], y[:-1]  # drop last row, target for it depends on a shifted NaN

    split = int(len(X) * 0.8)
    X_train, X_test = torch.tensor(X[:split]), torch.tensor(X[split:])
    y_train, y_test = torch.tensor(y[:split]), torch.tensor(y[split:])

    model = DirectionalLSTM(n_features=X.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad()
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        test_logits = model(X_test)
        test_pred = (torch.sigmoid(test_logits) > 0.5).float()
        accuracy = (test_pred == y_test).float().mean().item()
        baseline_majority = max(y_test.mean().item(), 1 - y_test.mean().item())

    print(f"train loss (final epoch): {loss.item():.4f}")
    print(f"test accuracy: {accuracy:.4f}")
    print(f"majority-class baseline: {baseline_majority:.4f}")

    torch.save(model.state_dict(), REPORTS_DIR / "lstm_directional.pt")
    with open(REPORTS_DIR / "lstm_metrics.txt", "w") as f:
        f.write(f"test_accuracy={accuracy:.4f}\n")
        f.write(f"majority_class_baseline={baseline_majority:.4f}\n")
        f.write(f"n_train={len(X_train)}\nn_test={len(X_test)}\n")

    print(f"-> {REPORTS_DIR.relative_to(REPORTS_DIR.parents[1])}")


if __name__ == "__main__":
    main()
