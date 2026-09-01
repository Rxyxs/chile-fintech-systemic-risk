"""Export the latest real spot price and realized volatility for the Monte Carlo engine."""
from __future__ import annotations

from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "data"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(ROOT / "data" / "chile_fintech.duckdb"), read_only=True)
    row = con.execute(
        "SELECT close, realized_vol_20d FROM chile_equity_features "
        "WHERE realized_vol_20d IS NOT NULL ORDER BY fecha DESC LIMIT 1"
    ).fetchone()
    con.close()

    spot, daily_vol = row
    annualized_vol = daily_vol * (252 ** 0.5)

    out_path = OUT_DIR / "market_params.csv"
    out_path.write_text(f"spot,annualized_vol\n{spot},{annualized_vol}\n")
    print(f"spot={spot:.4f} annualized_vol={annualized_vol:.4f} -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
