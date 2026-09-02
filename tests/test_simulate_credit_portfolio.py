"""Tests for ml_predictions/simulate_credit_portfolio.py — the synthetic PD dataset generator."""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml_predictions"))
import simulate_credit_portfolio as sim  # noqa: E402


def test_generates_expected_row_count_and_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(sim, "OUT_DIR", tmp_path)
    sim.main()

    out_path = tmp_path / "credit_portfolio_synthetic.parquet"
    assert out_path.exists()

    df = pl.read_parquet(out_path)
    assert df.height == sim.N
    assert set(df.columns) == {
        "income_clp", "age", "tenure_months", "loan_amount_clp", "dti",
        "tpm_at_origination", "num_prior_delinquencies", "has_formal_employment", "default",
    }


def test_default_label_is_binary_and_not_degenerate(tmp_path, monkeypatch):
    monkeypatch.setattr(sim, "OUT_DIR", tmp_path)
    sim.main()

    df = pl.read_parquet(tmp_path / "credit_portfolio_synthetic.parquet")
    assert set(df["default"].unique().to_list()) <= {0, 1}
    default_rate = df["default"].mean()
    assert 0.02 < default_rate < 0.5, (
        f"default rate {default_rate:.3%} outside a plausible consumer-credit range — "
        "check the logit coefficients in simulate_credit_portfolio.py"
    )


def test_dti_is_clipped_to_documented_bounds(tmp_path, monkeypatch):
    monkeypatch.setattr(sim, "OUT_DIR", tmp_path)
    sim.main()

    df = pl.read_parquet(tmp_path / "credit_portfolio_synthetic.parquet")
    assert df["dti"].min() >= 0.01
    assert df["dti"].max() <= 1.5


def test_higher_delinquency_correlates_with_higher_default_rate(tmp_path, monkeypatch):
    """Sanity check that the synthetic label actually reflects the risk factors it's built from."""
    monkeypatch.setattr(sim, "OUT_DIR", tmp_path)
    sim.main()

    df = pl.read_parquet(tmp_path / "credit_portfolio_synthetic.parquet")
    clean = df.filter(pl.col("num_prior_delinquencies") == 0)["default"].mean()
    risky = df.filter(pl.col("num_prior_delinquencies") >= 2)["default"].mean()
    assert risky > clean, "borrowers with prior delinquencies should default more often than clean ones"


def test_is_deterministic_given_fixed_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(sim, "OUT_DIR", tmp_path)
    sim.main()
    first = pl.read_parquet(tmp_path / "credit_portfolio_synthetic.parquet")

    sim.main()  # overwrites the same file
    second = pl.read_parquet(tmp_path / "credit_portfolio_synthetic.parquet")

    assert first.equals(second)
