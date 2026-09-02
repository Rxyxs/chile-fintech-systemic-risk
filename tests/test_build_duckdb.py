"""Tests for etl/build_duckdb.py — the raw-parquet-to-DuckDB load and the chile_equity_features view."""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "etl"))
import build_duckdb as etl  # noqa: E402


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    root = tmp_path
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)

    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(25)]
    equity = pl.DataFrame(
        {
            "fecha": dates,
            "open": [100.0 + i for i in range(25)],
            "high": [101.0 + i for i in range(25)],
            "low": [99.0 + i for i in range(25)],
            "close": [100.0 + i for i in range(25)],
            "volume": [1000] * 25,
        }
    )
    equity.write_parquet(raw / "chile_equity.parquet")

    indicators = pl.DataFrame({"fecha": dates[:5], "indicador": ["tpm"] * 5, "valor": [4.5] * 5})
    indicators.write_parquet(raw / "bcch_indicators_combined.parquet")

    monkeypatch.setattr(etl, "ROOT", root)
    monkeypatch.setattr(etl, "RAW_DIR", raw)
    monkeypatch.setattr(etl, "DB_PATH", root / "data" / "chile_fintech.duckdb")
    return root


def test_build_duckdb_creates_expected_row_counts(raw_dir):
    etl.main()

    import duckdb

    con = duckdb.connect(str(raw_dir / "data" / "chile_fintech.duckdb"), read_only=True)
    n_eq = con.execute("SELECT count(*) FROM chile_equity_daily").fetchone()[0]
    n_ind = con.execute("SELECT count(*) FROM bcch_indicators").fetchone()[0]
    con.close()

    assert n_eq == 25
    assert n_ind == 5


def test_chile_equity_features_log_return_matches_manual_calc(raw_dir):
    etl.main()

    import duckdb

    con = duckdb.connect(str(raw_dir / "data" / "chile_fintech.duckdb"), read_only=True)
    df = con.execute(
        "SELECT fecha, close, log_return FROM chile_equity_features ORDER BY fecha"
    ).fetchdf()
    con.close()

    assert df["log_return"].iloc[0] != df["log_return"].iloc[0]  # first row: NaN, no prior close
    # close goes 100 -> 101 on day 2: log_return should be ln(101/100)
    import math

    assert df["log_return"].iloc[1] == pytest.approx(math.log(101.0 / 100.0), abs=1e-9)


def test_chile_equity_features_sma20_window_is_capped_at_20_rows(raw_dir):
    etl.main()

    import duckdb

    con = duckdb.connect(str(raw_dir / "data" / "chile_fintech.duckdb"), read_only=True)
    df = con.execute(
        "SELECT fecha, close, sma_20 FROM chile_equity_features ORDER BY fecha"
    ).fetchdf()
    con.close()

    # last row (25th): mean of closes from day 6 to day 25 (20 rows), not all 25
    expected = sum(100.0 + i for i in range(5, 25)) / 20
    assert df["sma_20"].iloc[-1] == pytest.approx(expected)
