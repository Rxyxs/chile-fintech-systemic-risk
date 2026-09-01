"""Export DuckDB tables + synthetic credit portfolio to CSV for Julia clustering.

Julia reads plain CSV here instead of talking to DuckDB directly — a
deliberate simplicity choice: this project already treats parquet/CSV as the
cross-language boundary (see /etl), and adding DuckDB.jl as a dependency
buys nothing for a script that only needs two flat tables.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "data"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(ROOT / "data" / "chile_fintech.duckdb"), read_only=True)
    equity = con.execute(
        "SELECT fecha, log_return, sma_20, realized_vol_20d FROM chile_equity_features "
        "WHERE log_return IS NOT NULL AND realized_vol_20d IS NOT NULL ORDER BY fecha"
    ).fetchdf()
    con.close()
    equity.to_csv(OUT_DIR / "chile_equity_features.csv", index=False)
    print(f"chile_equity_features: {len(equity)} rows")

    credit = pl.read_parquet(ROOT / "ml_predictions" / "data" / "credit_portfolio_synthetic.parquet")
    credit.write_csv(OUT_DIR / "credit_portfolio_synthetic.csv")
    print(f"credit_portfolio_synthetic: {credit.height} rows")


if __name__ == "__main__":
    main()
