[ 🇨🇱 Versión en Español ](#-español) &nbsp;|&nbsp; [ 🇺🇸 English Version ](#-english)

---

<a name="-español"></a>
# chile-fintech-systemic-risk

Arquitectura políglota para análisis de riesgo sistémico del mercado financiero chileno (IPSA, tasas del Banco Central, riesgo crediticio). Cada lenguaje se eligió por la tarea que resuelve mejor, no por completitud del portafolio — ver justificación por módulo abajo.

**Estado del proyecto: los 5 módulos son funcionales y corren con datos reales.** Este README documenta resultados reales, incluyendo hallazgos honestos-negativos, en vez de maquillarlos.

## Estructura

| Módulo | Lenguaje(s) | Estado |
|---|---|---|
| [`/etl`](etl) | Python + DuckDB (SQL) | ✅ Funcional, datos reales |
| [`/ml_predictions`](ml_predictions) | Python (XGBoost, PyTorch, SHAP) | ✅ Funcional (PD: sintético documentado; equity: datos reales) |
| [`/quant_analytics`](quant_analytics) | R + Julia | ✅ Funcional, datos reales |
| [`/core_engine`](core_engine) | C++ (OpenMP) | ✅ Funcional, datos reales |
| [`/api`](api) | Go + C# | ✅ Funcional, datos reales |

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

![Precio y volatilidad animados](quant_analytics/figures/chile_equity_price_vol_animated.gif)
![Precio y volatilidad realizada del activo chileno](quant_analytics/figures/chile_equity_price_vol.png)

La versión animada traza ambas series al ritmo real de los datos (submuestreados a ~45 frames de los 2,511 días) con una etiqueta flotante que marca el valor vigente en cada punto — útil para ver de un vistazo dónde se dispara la volatilidad, aunque el gráfico estático de abajo sigue siendo la referencia para lectura detallada.

Serie completa de `ECH` 2016-2026: el panel superior es el precio de cierre, el inferior la volatilidad realizada de 20 días calculada en la vista SQL de DuckDB — se ve cómo los picos de volatilidad coinciden con los tramos de precio más agitados, la base para todo lo que viene después (LSTM, GARCH, clustering).

## `/ml_predictions` — lo que ya corre

```bash
python ml_predictions/simulate_credit_portfolio.py   # cartera sintética documentada
python ml_predictions/train_pd_model.py               # XGBoost + SHAP
python ml_predictions/train_lstm_equity.py             # LSTM direccional sobre datos reales
```

- **PD (XGBoost + SHAP):** AUC held-out = 0.624 sobre cartera de crédito sintética documentada (no hay fuente pública chilena a nivel de deudor individual). SHAP confirma que `dti` y morosidad previa dominan, como está diseñado en la simulación.

![Importancia de features por SHAP](ml_predictions/reports/figures/shap_importance.png)

`dti` y `num_prior_delinquencies` concentran la mayor parte del |SHAP value| promedio — exactamente las dos variables con más peso en la función logística que generó los defaults sintéticos, así que este gráfico funciona como verificación de que el pipeline de explicabilidad recupera la señal real, no ruido.

- **LSTM direccional (datos reales):** accuracy de test = 51.2%, **por debajo** del baseline de clase mayoritaria (53.6%). Hallazgo honesto, no descartado — mismo patrón que en `reading-market-turbulence`.

![LSTM vs. baseline de clase mayoritaria](ml_predictions/reports/figures/lstm_vs_baseline.png)

La barra del LSTM queda por debajo de la del baseline — no es un gráfico decorativo, es la evidencia visual del hallazgo honesto-negativo: predecir siempre la clase más frecuente le gana al modelo entrenado.

## `/quant_analytics` — lo que ya corre

```bash
Rscript quant_analytics/r/macro_econometrics.R
python quant_analytics/julia/export_for_julia.py && julia quant_analytics/julia/cluster_profiles.jl
```

- **R:** cointegración y Granger no concluyentes por ventana de datos corta de la API gratuita (~31 obs/indicador), documentado honestamente; GARCH(1,1) muestra persistencia de volatilidad = 0.987, coherente con el LSTM que no supera el baseline.
- **Julia:** K-Medoids separa regímenes de volatilidad reales del activo chileno (cluster de alta volatilidad = retorno promedio negativo) y perfiles de riesgo crediticio (menor DTI = menor tasa de default observada).

## `/core_engine` — lo que ya corre

```bash
python core_engine/cpp/export_params.py
cl /O2 /openmp /EHsc core_engine/cpp/montecarlo_var.cpp /Fe:core_engine/cpp/montecarlo_var.exe
core_engine/cpp/montecarlo_var.exe core_engine/cpp/data/market_params.csv
```

Motor Monte Carlo (GBM) en C++20 + OpenMP, alimentado con spot y volatilidad reales del activo chileno: pricing de opción call europea y VaR/ES 99% a 1 día de una posición larga. Resultado real: **1M trayectorias en 14.4ms** con 16 hilos.

![Distribución de PnL simulado con VaR/ES](core_engine/figures/montecarlo_var_distribution.png)

Histograma de 20,000 trayectorias de PnL a 1 día (submuestra de los datos que efectivamente se graficaron; la métrica de VaR/ES se calcula igual sobre el millón completo). Las líneas verticales marcan dónde caen el VaR 99% y el Expected Shortfall 99% sobre esa distribución — la cola izquierda es justamente lo que ambas métricas están midiendo.

## `/api` — lo que ya corre

```bash
python api/go/export_predictions.py && go run api/go/main.go
cd api/corporate_stubs/legacy_core_stub && dotnet run
```

Servicio Go (stdlib, sin dependencias) que sirve predicciones ya calculadas por los modelos Python/XGBoost — verificado: `/health`, `/v1/equity/snapshot`, `/v1/credit/predictions` responden con datos reales. Stub C#/.NET de integración con core bancario legado, con decisiones de crédito basadas en los PD scores reales del modelo.

## Por qué esta separación de lenguajes

- **Python + SQL (DuckDB)** para ETL: ecosistema de datos maduro, DuckDB da almacenamiento columnar local sin la operación de un warehouse real, adecuado para series de tiempo financieras de este tamaño.
- **XGBoost/LightGBM + PyTorch** para modelado: árboles de gradiente para PD tabular con SHAP obligatorio (explicabilidad regulatoria), deep learning para series de tiempo donde la dependencia temporal importa más que features tabulares.
- **R** para econometría clásica (cointegración, Granger, ARIMA/GARCH): sigue siendo el ecosistema con mejor soporte y validación académica para estos tests específicos.
- **Julia** para clustering de alta velocidad: cálculo matricial sin el overhead del GIL de Python, relevante para clustering sobre ventanas rolling grandes.
- **C++/Rust** para el motor Monte Carlo de pricing/VaR: el *hot-path* del sistema necesita control de memoria y paralelismo real (OpenMP/rayon), no algo que Python pueda dar sin FFI de todas formas.
- **Go** para el API de predicciones: throughput de muchas requests concurrentes de scoring, no cómputo pesado — el modelo ya corrió offline.

## Resultados clave (esta sesión)

| Módulo | Métrica | Resultado |
|---|---|---|
| PD (XGBoost+SHAP) | AUC held-out | 0.624 |
| Dirección equity (LSTM) | Accuracy vs. baseline | 51.2% vs. **53.6%** (no supera baseline) |
| Econometría (R, GARCH) | Persistencia de volatilidad | 0.987 |
| Clustering (Julia) | Default rate, cluster de menor DTI | 6.1% (vs. 13.6% del más riesgoso) |
| Motor Monte Carlo (C++) | 1M trayectorias | 14.4 ms, 16 hilos |

## Próximos pasos

Los 5 módulos ya son funcionales con datos reales. Mejoras pendientes documentadas en cada README de módulo: Temporal Fusion Transformer y ensamble LightGBM en `/ml_predictions`, ventana de datos más larga para los tests econométricos en `/quant_analytics`, y evaluación de Rust/Java como alternativas en `/core_engine` y `/api` respectivamente.

---

<a name="-english"></a>
# chile-fintech-systemic-risk (English)

Polyglot architecture for systemic risk analysis of the Chilean financial market (IPSA, Central Bank rates, credit risk). Each language was chosen for the task it solves best, not for portfolio completeness.

**Project status: all 5 modules are functional and run on real data.** This README honestly documents real results, including honest-negative findings, rather than inflating them.

## Structure

| Module | Language(s) | Status |
|---|---|---|
| [`/etl`](etl) | Python + DuckDB (SQL) | ✅ Functional, real data |
| [`/ml_predictions`](ml_predictions) | Python (XGBoost, PyTorch, SHAP) | ✅ Functional (PD: documented synthetic; equity: real data) |
| [`/quant_analytics`](quant_analytics) | R + Julia | ✅ Functional, real data |
| [`/core_engine`](core_engine) | C++ (OpenMP) | ✅ Functional, real data |
| [`/api`](api) | Go + C# | ✅ Functional, real data |

## `/etl`

Two Python (Polars) ingestors against real public APIs — Chilean Central Bank indicators via [mindicador.cl](https://mindicador.cl), and daily Chile-equity OHLCV via Yahoo Finance (using `ECH`, iShares MSCI Chile ETF, as a documented proxy for `^IPSA` — yfinance's own `^IPSA` feed has a real data gap and stops returning rows after 2019-06-14 regardless of the requested range, verified 2026-08-31). Consolidated into a local DuckDB store with a SQL feature view (log returns, SMA-20, 20-day realized volatility). Verified this session: 155 indicator rows, 2,511 equity rows spanning 2016-09-01 through 2026-08-31.

![Chile equity price and realized volatility, animated](quant_analytics/figures/chile_equity_price_vol_animated.gif)
![Chile equity price and realized volatility](quant_analytics/figures/chile_equity_price_vol.png)

The animated version draws both series at the real data's pace (subsampled to ~45 frames from the 2,511 trading days) with a floating label tracking the current value on each line — a quick way to spot where volatility spikes, while the static chart below remains the reference for detailed reading.

Full `ECH` series 2016-2026: the top panel is the closing price, the bottom one the 20-day realized volatility computed in the DuckDB SQL view — volatility spikes line up with the choppier price stretches, and this is the base data everything downstream (LSTM, GARCH, clustering) builds on.

## `/ml_predictions`

- **PD (XGBoost + SHAP):** held-out AUC = 0.624 on a documented synthetic credit portfolio (no public Chilean borrower-level dataset exists). SHAP confirms `dti` and prior delinquencies dominate, as designed into the simulation.

![SHAP feature importance](ml_predictions/reports/figures/shap_importance.png)

`dti` and `num_prior_delinquencies` account for most of the mean |SHAP value| — exactly the two variables that dominate the logistic function used to generate the synthetic defaults, so this chart doubles as a sanity check that the explainability pipeline recovers real signal, not noise.

- **Directional LSTM (real data):** test accuracy = 51.2%, **below** the majority-class baseline (53.6%). Honest finding, not discarded — same pattern as `reading-market-turbulence`.

![LSTM vs. majority-class baseline](ml_predictions/reports/figures/lstm_vs_baseline.png)

The LSTM bar sits below the baseline bar — not decorative, it's the visual evidence for the honest-negative finding: always predicting the more frequent class beats the trained model here.

## `/quant_analytics`

- **R:** cointegration and Granger causality inconclusive due to the free API's short data window (~31 obs/indicator), documented honestly; GARCH(1,1) shows volatility persistence = 0.987, consistent with the LSTM not beating its baseline.
- **Julia:** K-Medoids separates real volatility regimes in Chilean equity (high-volatility cluster = negative average return) and credit-risk profiles (lower DTI = lower observed default rate).

## `/core_engine`

Monte Carlo (GBM) engine in C++20 + OpenMP, fed with real spot and volatility from Chilean equity: European call pricing and 1-day 99% VaR/ES on a long position. Real result: **1M paths in 14.4ms** with 16 threads.

![Simulated PnL distribution with VaR/ES](core_engine/figures/montecarlo_var_distribution.png)

Histogram of 20,000 1-day PnL paths (a subsample of what actually gets plotted; the VaR/ES metrics themselves are computed over the full million). The vertical lines mark where the 99% VaR and 99% Expected Shortfall fall on that distribution — the left tail is exactly what both metrics are measuring.

## `/api`

Go service (stdlib, no dependencies) serving predictions already computed by the Python/XGBoost models — verified: `/health`, `/v1/equity/snapshot`, `/v1/credit/predictions` respond with real data. C#/.NET stub for legacy banking core integration, with credit decisions based on the model's real PD scores.

## Key results (this session)

| Module | Metric | Result |
|---|---|---|
| PD (XGBoost+SHAP) | Held-out AUC | 0.624 |
| Equity direction (LSTM) | Accuracy vs. baseline | 51.2% vs. **53.6%** (doesn't beat baseline) |
| Econometrics (R, GARCH) | Volatility persistence | 0.987 |
| Clustering (Julia) | Default rate, lowest-DTI cluster | 6.1% (vs. 13.6% for the riskiest) |
| Monte Carlo engine (C++) | 1M paths | 14.4 ms, 16 threads |

## Next steps

All 5 modules are functional on real data. Documented follow-ups per module: a Temporal Fusion Transformer and a LightGBM challenger in `/ml_predictions`, a longer data window for the econometric tests in `/quant_analytics`, and evaluating Rust/Java as alternatives in `/core_engine` and `/api` respectively.
