"""Plot the Monte Carlo 1-day PnL distribution and VaR/ES lines from the C++ engine's output."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).resolve().parent
FIG_DIR = BASE / "figures"

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10})


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(BASE / "cpp" / "data" / "pnl_sample.csv")
    pnl = df["pnl_1d"]

    var_99 = -pnl.quantile(0.01)
    es_99 = -pnl[pnl <= -var_99].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pnl, bins=80, color="#2A6F97", alpha=0.85)
    ax.axvline(-var_99, color="#C1440E", linestyle="--", label=f"VaR 99% = {var_99:,.0f}")
    ax.axvline(-es_99, color="#7A1F1F", linestyle=":", label=f"ES 99% = {es_99:,.0f}")
    ax.set_xlabel("1-day PnL (notional-scaled)")
    ax.set_ylabel("count (20,000-path sample)")
    ax.set_title("Monte Carlo 1-day PnL distribution — VaR/ES on real market params")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "montecarlo_var_distribution.png", dpi=150)
    plt.close(fig)
    print(f"chart -> {FIG_DIR.relative_to(BASE.parent)}")


if __name__ == "__main__":
    main()
