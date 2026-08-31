"""Ingest historical Chile equity market OHLCV data via Yahoo Finance into raw parquet.

NOTE on ticker choice: Yahoo Finance's own site shows ^IPSA (the native index)
trading through 2026, but yfinance's chart API for ^IPSA has a real data gap and
stops returning rows after 2019-06-14 regardless of the requested date range
(verified 2026-08-31 — not a bug in this script). We use ECH (iShares MSCI Chile
ETF, NYSE-listed) as a documented real-data proxy for Chilean equity exposure
instead, since it has an unbroken daily series through the present.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import yfinance as yf

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
TICKER = "ECH"  # iShares MSCI Chile ETF — proxy for ^IPSA (see module docstring)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    hist = yf.download(TICKER, period="10y", interval="1d", auto_adjust=True, progress=False)
    if hist.empty:
        raise SystemExit(f"No data returned for {TICKER}")

    hist = hist.reset_index()
    hist.columns = [c[0] if isinstance(c, tuple) else c for c in hist.columns]
    df = pl.from_pandas(hist).rename(
        {"Date": "fecha", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    )

    out_path = RAW_DIR / "chile_equity.parquet"
    df.write_parquet(out_path)
    print(f"{TICKER}: {df.height} rows, {df['fecha'].min()} -> {df['fecha'].max()} -> {out_path.relative_to(RAW_DIR.parents[1])}")


if __name__ == "__main__":
    main()
