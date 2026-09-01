"""Train an XGBoost Probability-of-Default model with mandatory SHAP explainability."""
from __future__ import annotations

from pathlib import Path

import polars as pl
import shap
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent / "data"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
FEATURES = [
    "income_clp",
    "age",
    "tenure_months",
    "loan_amount_clp",
    "dti",
    "tpm_at_origination",
    "num_prior_delinquencies",
    "has_formal_employment",
]


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(DATA_DIR / "credit_portfolio_synthetic.parquet").to_pandas()

    X, y = df[FEATURES], df["default"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=42,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"held-out AUC: {auc:.4f}")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    mean_abs_shap = (
        pl.DataFrame({"feature": FEATURES, "mean_abs_shap": abs(shap_values.values).mean(axis=0)})
        .sort("mean_abs_shap", descending=True)
    )
    print(mean_abs_shap)

    model.save_model(REPORTS_DIR / "pd_model.json")
    mean_abs_shap.write_csv(REPORTS_DIR / "shap_feature_importance.csv")
    with open(REPORTS_DIR / "metrics.txt", "w") as f:
        f.write(f"held_out_auc={auc:.4f}\n")
        f.write(f"n_train={len(X_train)}\nn_test={len(X_test)}\n")
        f.write(f"default_rate={y.mean():.4f}\n")

    print(f"-> {REPORTS_DIR.relative_to(REPORTS_DIR.parents[1])}")


if __name__ == "__main__":
    main()
