"""Ingest Banco Central de Chile macro indicators from mindicador.cl into raw parquet."""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import requests

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
BASE_URL = "https://mindicador.cl/api"

INDICATORS = ["tpm", "uf", "dolar", "ipc", "imacec"]


def fetch_series(codigo: str) -> pl.DataFrame:
    resp = requests.get(f"{BASE_URL}/{codigo}", timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    serie = payload.get("serie", [])
    if not serie:
        raise ValueError(f"'{codigo}': empty series from API")
    df = pl.DataFrame(serie).with_columns(
        pl.col("fecha").str.to_datetime(time_zone="UTC").alias("fecha"),
        pl.lit(codigo).alias("indicador"),
    )
    return df.select(["fecha", "indicador", "valor"])


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for codigo in INDICATORS:
        try:
            df = fetch_series(codigo)
        except Exception as exc:  # network/API is external and can fail; report and continue
            print(f"[warn] {codigo}: {exc}", file=sys.stderr)
            continue
        out_path = RAW_DIR / f"{codigo}.parquet"
        df.write_parquet(out_path)
        print(f"{codigo}: {df.height} rows -> {out_path.relative_to(RAW_DIR.parents[1])}")
        frames.append(df)

    if not frames:
        raise SystemExit("No indicator series were fetched successfully.")

    combined = pl.concat(frames)
    combined.write_parquet(RAW_DIR / "bcch_indicators_combined.parquet")
    print(f"combined: {combined.height} rows across {len(frames)} indicators")


if __name__ == "__main__":
    main()
