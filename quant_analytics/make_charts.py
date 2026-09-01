"""Generate charts from the real equity data and the R/Julia outputs — no re-fitting."""
from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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


def plot_price_and_vol_animated(df: pd.DataFrame) -> None:
    """Racing line-chart GIF of the same real price/vol series, dark-themed, for immediate visual impact."""
    n_frames = 45
    idx = pd.RangeIndex(0, len(df))
    stride_idx = list(idx[:: max(1, len(df) // n_frames)])
    if stride_idx[-1] != len(df) - 1:
        stride_idx.append(len(df) - 1)

    dates = df["fecha"].to_numpy()
    close = df["close"].to_numpy()
    vol = df["realized_vol_20d"].to_numpy()

    with plt.style.context("dark_background"):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        fig.suptitle("Chile equity: price and 20-day realized volatility (2016-2026, real data)")

        line1, = ax1.plot([], [], color="#5DA9E9", linewidth=1.3)
        line2, = ax2.plot([], [], color="#F2836B", linewidth=1.3)

        ax1.set_xlim(dates[0], dates[-1])
        ax1.set_ylim(close.min() * 0.95, close.max() * 1.05)
        ax1.set_ylabel("close (ECH, proxy for IPSA)")

        ax2.set_xlim(dates[0], dates[-1])
        ax2.set_ylim(vol.min() * 0.95, vol.max() * 1.05)
        ax2.set_ylabel("realized vol (20d, daily)")
        ax2.set_xlabel("date")

        bbox_style = dict(boxstyle="round,pad=0.3", fc="#1a1a1a", ec="#5DA9E9", lw=1)
        bbox_style2 = dict(boxstyle="round,pad=0.3", fc="#1a1a1a", ec="#F2836B", lw=1)
        label1 = ax1.annotate("", xy=(dates[0], close[0]), xytext=(15, 15),
                               textcoords="offset points", bbox=bbox_style, color="white", fontsize=9)
        label2 = ax2.annotate("", xy=(dates[0], vol[0]), xytext=(15, 15),
                               textcoords="offset points", bbox=bbox_style2, color="white", fontsize=9)

        fig.tight_layout()

        def update(frame_i: int):
            i = stride_idx[frame_i]
            line1.set_data(dates[: i + 1], close[: i + 1])
            line2.set_data(dates[: i + 1], vol[: i + 1])

            d = pd.Timestamp(dates[i]).strftime("%Y-%m-%d")
            label1.xy = (dates[i], close[i])
            label1.set_position((15, 15))
            label1.xyann = (dates[i], close[i])
            label1.set_text(f"close: {close[i]:.2f}\n{d}")

            label2.xy = (dates[i], vol[i])
            label2.set_text(f"realized_vol_20d: {vol[i]:.4f}\n{d}")
            return line1, line2, label1, label2

        ani = FuncAnimation(fig, update, frames=len(stride_idx), interval=120, blit=False)
        ani.save(FIG_DIR / "chile_equity_price_vol_animated.gif", writer="pillow")
        plt.close(fig)


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
    df = plot_price_and_vol()
    plot_price_and_vol_animated(df)
    plot_cluster_scatter()
    print(f"charts -> {FIG_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
