"""Score the held-out sets with the already-trained models and export predictions
as JSON for the Go service to serve.

Architecture note: the Go service does NOT re-implement XGBoost/LSTM inference.
It serves predictions produced by an offline batch-scoring job (this script) —
the standard train-offline/serve-online split, where Go's job is request
concurrency, not numerical computation.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import polars as pl
import torch
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
ML_DIR = ROOT / "ml_predictions"
OUT_DIR = Path(__file__).resolve().parent / "data"
FEATURES = [
    "income_clp", "age", "tenure_months", "loan_amount_clp", "dti",
    "tpm_at_origination", "num_prior_delinquencies", "has_formal_employment",
]


class DirectionalLSTM(torch.nn.Module):
    def __init__(self, n_features: int, hidden: int = 32):
        super().__init__()
        self.lstm = torch.nn.LSTM(n_features, hidden, batch_first=True)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def export_credit_predictions() -> list[dict]:
    model = xgb.XGBClassifier()
    model.load_model(ML_DIR / "reports" / "pd_model.json")

    df = pl.read_parquet(ML_DIR / "data" / "credit_portfolio_synthetic.parquet").to_pandas()
    sample = df.sample(n=200, random_state=7)
    proba = model.predict_proba(sample[FEATURES])[:, 1]

    return [
        {
            "applicant_id": int(idx),
            "pd_score": round(float(p), 4),
            "dti": round(float(row.dti), 3),
            "num_prior_delinquencies": int(row.num_prior_delinquencies),
            "actual_default": int(row.default),
        }
        for idx, (p, row) in enumerate(zip(proba, sample.itertuples()))
    ]


def export_equity_snapshot() -> dict:
    con = duckdb.connect(str(ROOT / "data" / "chile_fintech.duckdb"), read_only=True)
    row = con.execute(
        "SELECT fecha, close, log_return, sma_20, realized_vol_20d FROM chile_equity_features "
        "WHERE realized_vol_20d IS NOT NULL ORDER BY fecha DESC LIMIT 1"
    ).fetchone()
    con.close()
    fecha, close, log_return, sma_20, realized_vol_20d = row
    return {
        "as_of": str(fecha),
        "close": round(float(close), 4),
        "log_return": round(float(log_return), 6),
        "sma_20": round(float(sma_20), 4),
        "realized_vol_20d": round(float(realized_vol_20d), 6),
        "lstm_test_accuracy": 0.512,
        "majority_class_baseline": 0.536,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "credit_predictions": export_credit_predictions(),
        "equity_snapshot": export_equity_snapshot(),
    }
    out_path = OUT_DIR / "predictions.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"{len(payload['credit_predictions'])} credit predictions, 1 equity snapshot -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
