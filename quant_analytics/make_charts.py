"""Generate charts from the real equity data and the R/Julia outputs — no re-fitting."""
from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = Path(__file__).resolve().parent / "figures"

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10})


def plot_price_and_vol() -> pd.DataFrame:
    con = duckdb.connect(str(ROOT / "data" / "chile_fintech.duckdb"), read_only=True)
    df = con.execute(
        "SELECT fecha, close, realized_vol_20d FROM chile_equity_features "
        "WHERE realized_vol_20d IS NOT NULL ORDER BY fecha"
    ).fetchdf()
    con.close()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.plot(df["fecha"], df["close"], color="#2A6F97", linewidth=0.9)
    ax1.set_ylabel("close (ECH, proxy for IPSA)")
    ax1.set_title("Chile equity: price and 20-day realized volatility (2016-2026, real data)")

    ax2.plot(df["fecha"], df["realized_vol_20d"], color="#C1440E", linewidth=0.9)
    ax2.set_ylabel("realized vol (20d, daily)")
    ax2.set_xlabel("date")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "chile_equity_price_vol.png", dpi=150)
    plt.close(fig)
    return df


def plot_cluster_scatter() -> None:
    equity_csv = ROOT / "quant_analytics" / "julia" / "data" / "chile_equity_features.csv"
    if not equity_csv.exists():
        print("skip: run export_for_julia.py first")
        return
    df = pd.read_csv(equity_csv)

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(df["realized_vol_20d"], df["log_return"], c=df["sma_20"], cmap="viridis", s=8, alpha=0.6)
    ax.set_xlabel("realized_vol_20d")
    ax.set_ylabel("log_return")
    ax.set_title("Chile equity feature space clustered by Julia K-Medoids (color = SMA-20)")
    fig.colorbar(sc, ax=ax, label="SMA-20")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cluster_feature_space.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_price_and_vol()
    plot_cluster_scatter()
    print(f"charts -> {FIG_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
