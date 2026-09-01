"""Generate a synthetic Chilean consumer-credit portfolio for PD modeling.

No public dataset of real Chilean borrower-level credit data exists (for good
reason — it's personal financial data). This module generates a documented
synthetic portfolio instead, following the same honesty standard as
chile-credit-risk-scoring-engine and catching-credit-card-fraud: features are
economically motivated (income, DTI, macro exposure via TPM at origination),
default is a logistic function of those features plus noise, and this file
is the single source of truth for how "real" the labels are — synthetic,
not observed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

OUT_DIR = Path(__file__).resolve().parent / "data"
N = 20_000
SEED = 42


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    income_clp = rng.lognormal(mean=14.5, sigma=0.6, size=N)  # ~ CLP 500k-3M/month
    age = rng.integers(18, 75, size=N)
    tenure_months = rng.integers(0, 240, size=N)
    loan_amount = rng.lognormal(mean=14.0, sigma=0.8, size=N)
    dti = np.clip(rng.normal(0.35, 0.15, size=N), 0.01, 1.5)
    tpm_at_origination = rng.choice([1.0, 4.5, 6.5, 11.25, 4.5], size=N)  # historical TPM regimes
    num_prior_delinquencies = rng.poisson(0.4, size=N)
    has_formal_employment = rng.binomial(1, 0.72, size=N)

    logit = (
        -3.2
        + 2.4 * dti
        + 0.55 * num_prior_delinquencies
        - 0.35 * has_formal_employment
        + 0.08 * (tpm_at_origination - 4.5)
        - 0.15 * np.log1p(income_clp / 1e6)
        + 0.10 * (loan_amount / income_clp)
        - 0.01 * (tenure_months / 12)
        + rng.normal(0, 0.6, size=N)
    )
    p_default = 1 / (1 + np.exp(-logit))
    default = rng.binomial(1, p_default)

    df = pl.DataFrame(
        {
            "income_clp": income_clp,
            "age": age,
            "tenure_months": tenure_months,
            "loan_amount_clp": loan_amount,
            "dti": dti,
            "tpm_at_origination": tpm_at_origination,
            "num_prior_delinquencies": num_prior_delinquencies,
            "has_formal_employment": has_formal_employment,
            "default": default,
        }
    )

    out_path = OUT_DIR / "credit_portfolio_synthetic.parquet"
    df.write_parquet(out_path)
    print(f"synthetic portfolio: {df.height} rows, default rate = {df['default'].mean():.3%}")
    print(f"-> {out_path.relative_to(out_path.parents[2])}")


if __name__ == "__main__":
    main()
