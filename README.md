[ 🇨🇱 Versión en Español ](#-español) &nbsp;|&nbsp; [ 🇺🇸 English Version ](#-english)

---

<a name="-español"></a>
# chile-fintech-systemic-risk

Arquitectura políglota para análisis de riesgo sistémico del mercado financiero chileno (IPSA, tasas del Banco Central, riesgo crediticio). Cada lenguaje se eligió por la tarea que resuelve mejor, no por completitud del portafolio — ver justificación por módulo abajo.

**Estado del proyecto: en construcción activa, multi-sesión.** Este README documenta honestamente qué está implementado y corriendo con datos reales, y qué es diseño pendiente de construir.

## Estructura

| Módulo | Lenguaje(s) | Estado |
|---|---|---|
| [`/etl`](etl) | Python + DuckDB (SQL) | ✅ Funcional, datos reales |
| [`/ml_predictions`](ml_predictions) | Python (XGBoost, PyTorch, SHAP) | ✅ Funcional (PD: sintético documentado; equity: datos reales) |
| [`/quant_analytics`](quant_analytics) | R + Julia | ⏳ Diseñado, pendiente |
| [`/core_engine`](core_engine) | C++/Rust/C | ⏳ Diseñado, pendiente |
| [`/api`](api) | Go + Java/C# | ⏳ Diseñado, pendiente |

## `/etl` — lo que ya corre

Dos ingestores en Python (Polars) contra APIs públicas reales:

- **`fetch_bcch_indicators.py`**: TPM, UF, dólar, IPC, IMACEC desde [mindicador.cl](https://mindicador.cl) (fuente pública del Banco Central de Chile).
- **`fetch_chile_equity.py`**: serie diaria OHLCV de `ECH` (iShares MSCI Chile ETF) vía Yahoo Finance. **Nota honesta:** se usa `ECH` en vez del índice nativo `^IPSA` porque la API de `yfinance` para `^IPSA` tiene un gap real de datos y deja de devolver filas después de 2019-06-14 sin importar el rango solicitado (verificado 2026-08-31) — un fallo de la fuente, documentado en vez de escondido, siguiendo el estándar del resto de este portafolio.
- **`build_duckdb.py`**: consolida ambas fuentes en `data/chile_fintech.duckdb`, con una vista `chile_equity_features` (retorno log, SMA-20, volatilidad realizada 20d) calculada en SQL puro sobre DuckDB.

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Windows
python etl/fetch_bcch_indicators.py
python etl/fetch_chile_equity.py
python etl/build_duckdb.py
```

Resultado verificado en esta sesión: 155 filas de indicadores BCCh, 2,511 filas de `chile_equity_daily` (2016-09-01 → 2026-08-31).

## `/ml_predictions` — lo que ya corre

```bash
python ml_predictions/simulate_credit_portfolio.py   # cartera sintética documentada
python ml_predictions/train_pd_model.py               # XGBoost + SHAP
python ml_predictions/train_lstm_equity.py             # LSTM direccional sobre datos reales
```

- **PD (XGBoost + SHAP):** AUC held-out = 0.624 sobre cartera de crédito sintética documentada (no hay fuente pública chilena a nivel de deudor individual). SHAP confirma que `dti` y morosidad previa dominan, como está diseñado en la simulación.
- **LSTM direccional (datos reales):** accuracy de test = 51.2%, **por debajo** del baseline de clase mayoritaria (53.6%). Hallazgo honesto, no descartado — mismo patrón que en `reading-market-turbulence`.

## Por qué esta separación de lenguajes

- **Python + SQL (DuckDB)** para ETL: ecosistema de datos maduro, DuckDB da almacenamiento columnar local sin la operación de un warehouse real, adecuado para series de tiempo financieras de este tamaño.
- **XGBoost/LightGBM + PyTorch** para modelado: árboles de gradiente para PD tabular con SHAP obligatorio (explicabilidad regulatoria), deep learning para series de tiempo donde la dependencia temporal importa más que features tabulares.
- **R** para econometría clásica (cointegración, Granger, ARIMA/GARCH): sigue siendo el ecosistema con mejor soporte y validación académica para estos tests específicos.
- **Julia** para clustering de alta velocidad: cálculo matricial sin el overhead del GIL de Python, relevante para clustering sobre ventanas rolling grandes.
- **C++/Rust** para el motor Monte Carlo de pricing/VaR: el *hot-path* del sistema necesita control de memoria y paralelismo real (OpenMP/rayon), no algo que Python pueda dar sin FFI de todas formas.
- **Go** para el API de predicciones: throughput de muchas requests concurrentes de scoring, no cómputo pesado — el modelo ya corrió offline.

## Próximos pasos

Ver el README de cada módulo pendiente para el diseño detallado y la razón de por qué aún no está implementado.

---

<a name="-english"></a>
# chile-fintech-systemic-risk (English)

Polyglot architecture for systemic risk analysis of the Chilean financial market (IPSA, Central Bank rates, credit risk). Each language was chosen for the task it solves best, not for portfolio completeness — see per-module justification above (same content, bilingual document).

**Project status: actively under construction, multi-session.** This README honestly documents what's implemented and running on real data versus what's designed-but-pending.

## What's actually running

Two Python (Polars) ingestors against real public APIs — Chilean Central Bank indicators via mindicador.cl, and daily Chile-equity OHLCV via Yahoo Finance (using `ECH` as a documented proxy for `^IPSA`, since yfinance's `^IPSA` feed has a real 2019 data gap) — consolidated into a local DuckDB store with a SQL feature view (log returns, SMA-20, 20-day realized volatility). Verified this session: 155 indicator rows, 2,511 equity rows spanning 2016-09-01 through 2026-08-31.

Everything else (`ml_predictions`, `quant_analytics`, `core_engine`, `api`) is designed but not yet built — see each folder's README for the detailed design and why it isn't implemented yet.
