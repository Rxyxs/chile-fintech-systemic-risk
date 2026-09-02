"""Tests for ml_predictions/train_pd_model.py — the XGBoost PD model + SHAP pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml_predictions"))
import simulate_credit_portfolio as sim  # noqa: E402
import train_pd_model as train  # noqa: E402


def _write_synthetic_portfolio(tmp_path, monkeypatch):
    monkeypatch.setattr(sim, "OUT_DIR", tmp_path)
    sim.main()


def test_model_beats_majority_class_baseline(tmp_path, monkeypatch):
    _write_synthetic_portfolio(tmp_path, monkeypatch)
    monkeypatch.setattr(train, "DATA_DIR", tmp_path)
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(train, "REPORTS_DIR", reports_dir)

    train.main()

    metrics_text = (reports_dir / "metrics.txt").read_text()
    auc = float([line for line in metrics_text.splitlines() if line.startswith("held_out_auc=")][0].split("=")[1])
    assert auc > 0.55, f"held-out AUC {auc:.4f} is barely above random (0.5) — model isn't learning signal"


def test_shap_importance_covers_all_features_and_is_sorted(tmp_path, monkeypatch):
    _write_synthetic_portfolio(tmp_path, monkeypatch)
    monkeypatch.setattr(train, "DATA_DIR", tmp_path)
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(train, "REPORTS_DIR", reports_dir)

    train.main()

    shap_df = pl.read_csv(reports_dir / "shap_feature_importance.csv")
    assert set(shap_df["feature"].to_list()) == set(train.FEATURES)
    values = shap_df["mean_abs_shap"].to_list()
    assert values == sorted(values, reverse=True), "SHAP importances should be written sorted descending"


def test_dti_is_a_top_predictor(tmp_path, monkeypatch):
    """dti has the largest coefficient in the synthetic label's logit, so it should show up as important."""
    _write_synthetic_portfolio(tmp_path, monkeypatch)
    monkeypatch.setattr(train, "DATA_DIR", tmp_path)
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(train, "REPORTS_DIR", reports_dir)

    train.main()

    shap_df = pl.read_csv(reports_dir / "shap_feature_importance.csv")
    top_3 = shap_df.head(3)["feature"].to_list()
    assert "dti" in top_3
