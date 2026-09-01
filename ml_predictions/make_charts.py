"""Generate charts from the already-trained PD and LSTM results — no re-training."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import xgboost as xgb

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
REPORTS_DIR = BASE / "reports"
FIG_DIR = REPORTS_DIR / "figures"
FEATURES = [
    "income_clp", "age", "tenure_months", "loan_amount_clp", "dti",
    "tpm_at_origination", "num_prior_delinquencies", "has_formal_employment",
]

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10})


def plot_shap_importance() -> None:
    shap_df = pl.read_csv(REPORTS_DIR / "shap_feature_importance.csv").sort("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(shap_df["feature"], shap_df["mean_abs_shap"], color="#2A6F97")
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("PD model — SHAP feature importance (held-out set)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "shap_importance.png", dpi=150)
    plt.close(fig)


def plot_pd_score_distribution() -> None:
    model = xgb.XGBClassifier()
    model.load_model(REPORTS_DIR / "pd_model.json")
    df = pl.read_parquet(DATA_DIR / "credit_portfolio_synthetic.parquet").to_pandas()
    proba = model.predict_proba(df[FEATURES])[:, 1]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(proba[df["default"] == 0], bins=40, alpha=0.6, label="no default (actual)", color="#2A6F97")
    ax.hist(proba[df["default"] == 1], bins=40, alpha=0.6, label="default (actual)", color="#C1440E")
    ax.set_xlabel("predicted PD score")
    ax.set_ylabel("count")
    ax.set_title("PD score distribution by actual outcome (full synthetic portfolio, n=20,000)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pd_score_distribution.png", dpi=150)
    plt.close(fig)


def plot_lstm_vs_baseline() -> None:
    metrics = {}
    for line in (REPORTS_DIR / "lstm_metrics.txt").read_text().splitlines():
        k, v = line.split("=")
        metrics[k] = float(v) if "." in v else int(v)

    labels = ["LSTM (test)", "Majority-class baseline"]
    values = [metrics["test_accuracy"], metrics["majority_class_baseline"]]
    colors = ["#2A6F97", "#6C757D"]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, max(values) * 1.25)
    ax.set_ylabel("directional accuracy")
    ax.set_title("Chile equity next-day direction: LSTM vs. baseline")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.1%}", ha="center")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lstm_vs_baseline.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_shap_importance()
    plot_pd_score_distribution()
    plot_lstm_vs_baseline()
    print(f"3 charts -> {FIG_DIR.relative_to(BASE.parent)}")


if __name__ == "__main__":
    main()
