"""Load raw parquet sources into a local DuckDB analytical store."""
from __future__ import annotations

from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "chile_fintech.duckdb"


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))

    con.execute(
        f"""
        CREATE OR REPLACE TABLE bcch_indicators AS
        SELECT fecha::DATE AS fecha, indicador, valor
        FROM read_parquet('{(RAW_DIR / 'bcch_indicators_combined.parquet').as_posix()}')
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE chile_equity_daily AS
        SELECT fecha::DATE AS fecha, open, high, low, close, volume
        FROM read_parquet('{(RAW_DIR / 'chile_equity.parquet').as_posix()}')
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW chile_equity_features AS
        WITH returns AS (
            SELECT
                fecha,
                close,
                volume,
                ln(close / lag(close) OVER (ORDER BY fecha)) AS log_return
            FROM chile_equity_daily
        )
        SELECT
            fecha,
            close,
            volume,
            log_return,
            avg(close) OVER (ORDER BY fecha ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS sma_20,
            stddev(log_return) OVER (ORDER BY fecha ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS realized_vol_20d
        FROM returns
        ORDER BY fecha
        """
    )

    n_ind = con.execute("SELECT count(*) FROM bcch_indicators").fetchone()[0]
    n_eq = con.execute("SELECT count(*) FROM chile_equity_daily").fetchone()[0]
    print(f"bcch_indicators: {n_ind} rows")
    print(f"chile_equity_daily: {n_eq} rows")
    print(f"chile_fintech.duckdb -> {DB_PATH.relative_to(ROOT)}")

    con.close()


if __name__ == "__main__":
    main()
