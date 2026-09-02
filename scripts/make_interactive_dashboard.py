"""Builds a self-contained interactive Plotly dashboard (offline HTML) from
real pipeline outputs: the DuckDB equity/volatility feature view built by
etl/build_duckdb.py, and the 200 scored credit predictions exported by
api/go/export_predictions.py (real XGBoost PD scores on the synthetic
portfolio). No fabricated data -- both sources come from an actual run.

Output: outputs/interactive/systemic_risk_dashboard.html
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "chile_fintech.duckdb"
PRED_PATH = ROOT / "api" / "go" / "data" / "predictions.json"
OUT_DIR = ROOT / "outputs" / "interactive"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB_PATH), read_only=True)
    eq = con.execute(
        "SELECT fecha, close, sma_20, realized_vol_20d FROM chile_equity_features "
        "WHERE realized_vol_20d IS NOT NULL ORDER BY fecha"
    ).fetchdf()
    con.close()

    preds = json.loads(PRED_PATH.read_text(encoding="utf-8"))
    credit = preds["credit_predictions"]
    dti = [c["dti"] for c in credit]
    pd_score = [c["pd_score"] for c in credit]
    default = [c["actual_default"] for c in credit]
    applicant_id = [c["applicant_id"] for c in credit]

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.58, 0.42],
        vertical_spacing=0.12,
        specs=[[{"secondary_y": True}], [{}]],
        subplot_titles=(
            "ECH (proxy IPSA) — precio de cierre vs. volatilidad realizada 20d (DuckDB SQL view)",
            "PD scores (XGBoost) vs. DTI — 200 solicitudes servidas por el API Go, coloreadas por default real",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=eq["fecha"], y=eq["close"], name="Cierre ECH (CLP)",
            line=dict(color="#3b6fd6", width=1.6),
            hovertemplate="%{x|%Y-%m-%d}<br>Cierre: %{y:.2f}<extra></extra>",
        ),
        row=1, col=1, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=eq["fecha"], y=eq["sma_20"], name="SMA-20",
            line=dict(color="#9aa7bd", width=1, dash="dot"),
            hovertemplate="%{x|%Y-%m-%d}<br>SMA-20: %{y:.2f}<extra></extra>",
        ),
        row=1, col=1, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=eq["fecha"], y=eq["realized_vol_20d"], name="Vol. realizada 20d",
            line=dict(color="#e0703c", width=1.4),
            fill="tozeroy", fillcolor="rgba(224,112,60,0.12)",
            hovertemplate="%{x|%Y-%m-%d}<br>Vol 20d: %{y:.4f}<extra></extra>",
        ),
        row=1, col=1, secondary_y=True,
    )

    colors = ["#2e9e5b" if d == 0 else "#c0392b" for d in default]
    fig.add_trace(
        go.Scatter(
            x=dti, y=pd_score, mode="markers", name="Solicitudes de crédito",
            marker=dict(color=colors, size=7, opacity=0.75, line=dict(width=0.4, color="white")),
            customdata=list(zip(applicant_id, default)),
            hovertemplate=(
                "applicant_id: %{customdata[0]}<br>DTI: %{x:.3f}<br>PD score: %{y:.3f}"
                "<br>default real: %{customdata[1]}<extra></extra>"
            ),
        ),
        row=2, col=1,
    )

    fig.update_layout(
        title=dict(
            text="chile-fintech-systemic-risk — dashboard interactivo (datos reales de esta sesión)",
            x=0.02, xanchor="left",
        ),
        template="plotly_white",
        height=900,
        legend=dict(orientation="h", y=1.06, x=0),
        margin=dict(t=110, l=60, r=60, b=40),
        hovermode="closest",
    )
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06), row=1, col=1)
    fig.update_yaxes(title_text="CLP", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Volatilidad realizada (20d)", row=1, col=1, secondary_y=True)
    fig.update_xaxes(title_text="DTI (debt-to-income)", row=2, col=1)
    fig.update_yaxes(title_text="PD score (XGBoost)", row=2, col=1)

    out_path = OUT_DIR / "systemic_risk_dashboard.html"
    fig.write_html(out_path, include_plotlyjs="inline", full_html=True)
    print(f"-> {out_path.relative_to(ROOT)}  ({len(eq)} equity rows, {len(credit)} credit predictions)")


if __name__ == "__main__":
    main()
