[ 🇺🇸 [Read in English](README.md) ] | [ 🇨🇱 Español ]

# chile-fintech-systemic-risk

Arquitectura políglota para análisis de riesgo sistémico del mercado financiero chileno (IPSA, tasas del Banco Central, riesgo crediticio). Cada lenguaje se eligió por la tarea que resuelve mejor, no por completitud del portafolio — ver justificación por módulo abajo.

**Estado del proyecto: los 5 módulos son funcionales y corren con datos reales.** Este README documenta resultados reales, incluyendo hallazgos honestos-negativos, en vez de maquillarlos. Todas las cifras de este documento salen de una re-ejecución completa de los 5 módulos el 2026-09-02; donde el número cambió levemente frente a una corrida anterior (aleatoriedad de inicialización de pesos, timing de CPU) se dejó el valor de esta corrida.

## El problema

El riesgo sistémico de un mercado financiero pequeño y concentrado como el chileno no vive en una sola fuente de datos: vive en cómo un shock en una capa se propaga a las demás. Una subida de tasa del Banco Central (TPM) presiona el costo de financiamiento de los hogares endeudados, lo que sube la probabilidad de default (PD) de la cartera de crédito; al mismo tiempo, la volatilidad del mercado accionario (IPSA) tiende a subir junto con la incertidumbre de tasas, y esa volatilidad es justamente el insumo que un motor de pricing/VaR necesita para cuantificar cuánto puede perder una posición en un día malo. Ningún modelo aislado —ni el de crédito, ni el de mercado— captura esa cadena completa.

Este repo construye esa cadena de punta a punta con datos reales donde existen y con supuestos documentados donde no: **ETL** consolida series públicas del Banco Central de Chile y el precio diario de un proxy del IPSA en un store analítico; **ml_predictions** estima PD de crédito (XGBoost, con SHAP obligatorio porque un modelo de crédito no explicable no es desplegable bajo un estándar regulatorio razonable) y dirección de mercado (LSTM); **quant_analytics** aporta el aparataje econométrico clásico (cointegración, GARCH) que un equipo de riesgo esperaría ver antes de confiar en un modelo de ML, más clustering de regímenes de volatilidad y de perfiles de riesgo; **core_engine** traduce la volatilidad real observada en una métrica de VaR/ES accionable; y **api** expone todo eso como sería consumido en producción — scoring servido con baja latencia, y un punto de integración explícito hacia un core bancario legado. El hilo conductor no es "un modelo por módulo", es una sola cadena de riesgo que atraviesa cinco lenguajes porque cada eslabón de esa cadena tiene requisitos distintos (explicabilidad regulatoria, rigor econométrico, throughput de requests, latencia de cómputo numérico).

Un resultado honesto de este ejercicio: el mercado chileno, medido con los datos públicos disponibles, se comporta de forma difícil de predecir direccionalmente (el LSTM no le gana a un baseline trivial) pero con volatilidad muy persistente (GARCH ≈ 0.987) — dos caras de la misma moneda, y ambas relevantes para cómo se debería gestionar el riesgo en la práctica: no confiar en predicción direccional, sí en gestión activa de exposición a volatilidad.

## Estructura

| Módulo | Lenguaje(s) | Estado |
|---|---|---|
| [`/etl`](etl) | Python + DuckDB (SQL) | ✅ Funcional, datos reales |
| [`/ml_predictions`](ml_predictions) | Python (XGBoost, PyTorch, SHAP) | ✅ Funcional (PD: sintético documentado; equity: datos reales) |
| [`/quant_analytics`](quant_analytics) | R + Julia | ✅ Funcional, datos reales |
| [`/core_engine`](core_engine) | C++ (OpenMP) | ✅ Funcional, datos reales |
| [`/api`](api) | Go + C# | ✅ Funcional, datos reales |

## Técnicas por módulo

| Módulo | Técnica / algoritmo | Librería(s) | Por qué |
|---|---|---|---|
| `/etl` | Vista SQL sobre store columnar (retorno log, SMA-20, volatilidad realizada 20d) | Polars, DuckDB | Feature engineering reproducible y auditable en SQL puro, sin recalcular en cada consumidor |
| `/ml_predictions` (PD) | Gradient boosting (XGBoost) + explicabilidad por valores de Shapley | `xgboost`, `shap`, scikit-learn | Árboles para tabular con variables mixtas; SHAP es obligatorio para un modelo de crédito, no opcional |
| `/ml_predictions` (equity) | LSTM de una capa sobre secuencias de 20 días (retorno log, SMA-20, vol. realizada) | PyTorch | Dependencia temporal explícita, comparada contra un baseline de clase mayoritaria |
| `/quant_analytics` (R) | Cointegración Engle-Granger (ADF sobre residuos), causalidad de Granger, ARIMA(1,0,0)-GARCH(1,1) | `urca`, `vars`, `tseries`, `rugarch` | El aparataje econométrico estándar que un equipo de riesgo de mercado exige antes de confiar en un modelo de ML |
| `/quant_analytics` (Julia) | K-Medoids (distancia euclidiana) sobre features estandarizadas | `Clustering.jl`, `Distances.jl` | Cómputo matricial puro sin el overhead del GIL de Python, apto para clustering repetido sobre ventanas grandes |
| `/core_engine` | Simulación Monte Carlo de Movimiento Browniano Geométrico, paralelizada | C++20 + OpenMP | Pricing de opción europea y VaR/ES 99% necesitan millones de trayectorias; el *hot path* requiere paralelismo real, no algo que Python dé sin FFI |
| `/api` (Go) | Servicio HTTP stdlib, *batch-scored offline / served online* | `net/http` (sin dependencias) | Throughput de requests concurrentes sirviendo predicciones ya calculadas, no cómputo numérico |
| `/api` (C#) | Stub de integración con core bancario legado, decisión basada en PD real | .NET 8 | Contrato de integración documentado (no mock oculto) hacia un sistema legado típico de banca chilena |

## `/etl` — lo que ya corre

Dos ingestores en Python (Polars) contra APIs públicas reales:

- **`fetch_bcch_indicators.py`**: TPM, UF, dólar, IPC, IMACEC desde [mindicador.cl](https://mindicador.cl) (fuente pública del Banco Central de Chile).
- **`fetch_chile_equity.py`**: serie diaria OHLCV de `ECH` (iShares MSCI Chile ETF) vía Yahoo Finance. **Nota honesta:** se usa `ECH` en vez del índice nativo `^IPSA` porque la API de `yfinance` para `^IPSA` tiene un gap real de datos y deja de devolver filas después de 2019-06-14 sin importar el rango solicitado (verificado 2026-08-31) — un fallo de la fuente, documentado en vez de escondido.
- **`build_duckdb.py`**: consolida ambas fuentes en `data/chile_fintech.duckdb`, con una vista `chile_equity_features` (retorno log, SMA-20, volatilidad realizada 20d) calculada en SQL puro sobre DuckDB.

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Windows
python etl/fetch_bcch_indicators.py
python etl/fetch_chile_equity.py
python etl/build_duckdb.py
```

Resultado verificado en esta sesión (2026-09-02): 155 filas de indicadores BCCh, 2,511 filas de `chile_equity_daily` (2016-09-02 → 2026-09-01).

![Precio y volatilidad animados](quant_analytics/figures/chile_equity_price_vol_animated.gif)
![Precio y volatilidad realizada del activo chileno](quant_analytics/figures/chile_equity_price_vol.png)

La versión animada traza ambas series al ritmo real de los datos (submuestreados a ~45 frames de los 2,511 días) con una etiqueta flotante que marca el valor vigente en cada punto — útil para ver de un vistazo dónde se dispara la volatilidad, aunque el gráfico estático de abajo sigue siendo la referencia para lectura detallada.

Serie completa de `ECH` 2016-2026: el panel superior es el precio de cierre, el inferior la volatilidad realizada de 20 días calculada en la vista SQL de DuckDB — se ve cómo los picos de volatilidad coinciden con los tramos de precio más agitados, la base para todo lo que viene después (LSTM, GARCH, clustering).

## `/ml_predictions` — lo que ya corre

```bash
python ml_predictions/simulate_credit_portfolio.py   # cartera sintética documentada
python ml_predictions/train_pd_model.py               # XGBoost + SHAP
python ml_predictions/train_lstm_equity.py             # LSTM direccional sobre datos reales
python ml_predictions/make_charts.py
```

- **PD (XGBoost + SHAP):** AUC held-out = **0.6239** sobre cartera de crédito sintética documentada de 20,000 solicitudes, tasa de default = 10.01% (no hay fuente pública chilena a nivel de deudor individual). SHAP confirma que `dti` (0.271) y morosidad previa (0.269) dominan el |SHAP value| promedio, como está diseñado en la simulación.

![Importancia de features por SHAP](ml_predictions/reports/figures/shap_importance.png)
![Distribución de PD scores](ml_predictions/reports/figures/pd_score_distribution.png)

`dti` y `num_prior_delinquencies` concentran la mayor parte del |SHAP value| promedio — exactamente las dos variables con más peso en la función logística que generó los defaults sintéticos, así que este gráfico funciona como verificación de que el pipeline de explicabilidad recupera la señal real, no ruido. La distribución de PD scores muestra la separación (parcial, consistente con un AUC de 0.62) entre solicitudes que efectivamente cayeron en default y las que no.

- **LSTM direccional (datos reales):** accuracy de test = **51.0%**, por debajo del baseline de clase mayoritaria (**53.6%**). Hallazgo honesto, no descartado: con retornos diarios de un activo altamente líquido y volatilidad muy persistente (ver GARCH abajo), la dirección día-a-día se comporta cerca de un camino aleatorio — el modelo no encuentra señal explotable en las tres features usadas (retorno log, SMA-20, vol. realizada 20d).

![LSTM vs. baseline de clase mayoritaria](ml_predictions/reports/figures/lstm_vs_baseline.png)

La barra del LSTM queda por debajo de la del baseline — no es un gráfico decorativo, es la evidencia visual del hallazgo honesto-negativo: predecir siempre la clase más frecuente le gana al modelo entrenado.

## `/quant_analytics` — lo que ya corre

```bash
Rscript quant_analytics/r/macro_econometrics.R
python quant_analytics/julia/export_for_julia.py && julia quant_analytics/julia/cluster_profiles.jl
python quant_analytics/make_charts.py
```

- **R:** cointegración UF vs. dólar (Engle-Granger, ADF sobre residuos) da estadístico = **-1.851** con apenas n=22 observaciones mensuales — insuficiente para concluir, documentado honestamente en vez de forzar una lectura. Causalidad de Granger de TPM sobre retornos no es evaluable en esta ventana (solo 1 valor único de TPM en los ~30 días alineados). GARCH(1,1) sobre 2,510 retornos diarios sí es concluyente: persistencia de volatilidad (α+β) = **0.9871** — volatilidad extremadamente persistente, coherente con el LSTM que no supera su baseline.
- **Julia:** K-Medoids (k=3) separa regímenes de volatilidad reales del activo chileno — el cluster de mayor volatilidad (n=10, vol. promedio 20d = 5.05%) tiene retorno promedio **negativo** (-8.29%), los otros dos clusters (vol. 1.18% y 1.83%) tienen retorno positivo. K-Medoids (k=4) sobre perfiles de crédito: el cluster de menor DTI (0.209) tiene default rate = **6.1%**, el de mayor DTI entre los evaluados (0.348) llega a **13.6%**.

![Clusters de crédito y volatilidad en espacio de features](quant_analytics/figures/cluster_feature_space.png)

## `/core_engine` — lo que ya corre

```bash
python core_engine/cpp/export_params.py
cl /O2 /openmp /EHsc core_engine/cpp/montecarlo_var.cpp /Fe:core_engine/cpp/montecarlo_var.exe
core_engine/cpp/montecarlo_var.exe core_engine/cpp/data/market_params.csv core_engine/cpp/data/pnl_sample.csv
```

Motor Monte Carlo (GBM) en C++20 + OpenMP, alimentado con spot (40.39 CLP) y volatilidad anualizada real (18.97%) del activo chileno: pricing de opción call europea (strike 41.20, precio = 0.601 ± 0.0011 stderr) y VaR/ES 99% a 1 día de una posición larga. Resultado real de esta corrida: **1,000,000 trayectorias en 14.6 ms** con 16 hilos OpenMP; VaR 99% = 27,485, ES 99% = 31,406 (unidades del notional configurado).

![Distribución de PnL simulado con VaR/ES](core_engine/figures/montecarlo_var_distribution.png)

Histograma de una submuestra de 20,000 trayectorias de PnL a 1 día (la métrica de VaR/ES se calcula igual sobre el millón completo). Las líneas verticales marcan dónde caen el VaR 99% y el Expected Shortfall 99% sobre esa distribución — la cola izquierda es justamente lo que ambas métricas están midiendo.

## `/api` — lo que ya corre

```bash
python api/go/export_predictions.py && go run api/go/main.go
cd api/corporate_stubs/legacy_core_stub && dotnet run
```

Servicio Go (stdlib, sin dependencias) que sirve predicciones ya calculadas por los modelos Python/XGBoost — verificado en esta sesión desde la raíz del repo: `/health` → `ok`; `/v1/equity/snapshot` responde cierre=40.39, retorno log=-0.00272, vol. realizada 20d=0.0120, junto con las métricas del LSTM; `/v1/credit/predictions` sirve las 200 solicitudes con `pd_score` real (ej. applicant_id 0: PD=0.0724, DTI=0.23; applicant_id 1: PD=0.1338, DTI=0.275). Stub C#/.NET de integración con core bancario legado compiló y corrió: dos decisiones de crédito (una aprobada, una rechazada) basadas en los PD scores reales del modelo.

## Dashboard interactivo (Plotly, offline)

**[Abrir el dashboard interactivo en el navegador](https://htmlpreview.github.io/?https://github.com/Rxyxs/chile-fintech-systemic-risk/blob/main/outputs/interactive/systemic_risk_dashboard.html)**

HTML autocontenido (`outputs/interactive/systemic_risk_dashboard.html`, generado por `scripts/make_interactive_dashboard.py` con `plotly`) construido directamente desde `chile_fintech.duckdb` y desde las 200 predicciones reales servidas por el API Go — sin datos inventados. El enlace de arriba usa `htmlpreview.github.io` para renderizarlo en vivo en el navegador sin clonar el repo. Incluye:

1. Precio de cierre de `ECH` + SMA-20 (eje izquierdo) y volatilidad realizada 20d (eje derecho), con range-slider interactivo sobre los 2,509 días con feature calculada.
2. Scatter de PD score (XGBoost) vs. DTI para las 200 solicitudes servidas por el API, coloreado por si la solicitud efectivamente cayó en default — se ve la correlación positiva esperada entre DTI/PD score y default real, con ruido suficiente para explicar por qué el AUC es 0.62 y no más alto.

## Por qué esta separación de lenguajes

- **Python + SQL (DuckDB)** para ETL: ecosistema de datos maduro, DuckDB da almacenamiento columnar local sin la operación de un warehouse real, adecuado para series de tiempo financieras de este tamaño.
- **XGBoost/LightGBM + PyTorch** para modelado: árboles de gradiente para PD tabular con SHAP obligatorio (explicabilidad regulatoria), deep learning para series de tiempo donde la dependencia temporal importa más que features tabulares.
- **R** para econometría clásica (cointegración, Granger, ARIMA/GARCH): sigue siendo el ecosistema con mejor soporte y validación académica para estos tests específicos.
- **Julia** para clustering de alta velocidad: cálculo matricial sin el overhead del GIL de Python, relevante para clustering sobre ventanas rolling grandes.
- **C++** para el motor Monte Carlo de pricing/VaR: el *hot-path* del sistema necesita control de memoria y paralelismo real (OpenMP), no algo que Python pueda dar sin FFI de todas formas.
- **Go** para el API de predicciones: throughput de muchas requests concurrentes de scoring, no cómputo pesado — el modelo ya corrió offline.

## Resultados clave (re-ejecución 2026-09-02)

| Módulo | Métrica | Resultado |
|---|---|---|
| PD (XGBoost+SHAP) | AUC held-out | 0.6239 |
| Dirección equity (LSTM) | Accuracy vs. baseline | 51.0% vs. **53.6%** (no supera baseline) |
| Econometría (R, GARCH) | Persistencia de volatilidad | 0.9871 |
| Econometría (R, cointegración) | ADF sobre residuos (UF vs. USD) | -1.851 (n=22, no concluyente) |
| Clustering (Julia) | Default rate, cluster de menor DTI | 6.1% (vs. 13.6% del más riesgoso evaluado) |
| Motor Monte Carlo (C++) | 1M trayectorias | 14.6 ms, 16 hilos |
| API Go | Endpoints verificados en vivo | `/health`, `/v1/equity/snapshot`, `/v1/credit/predictions` — 200 predicciones reales |
| Stub C#/.NET | Decisiones de crédito verificadas | 1 aprobada, 1 rechazada, ambas basadas en PD real |
| Tests automatizados (pytest, CI) | 15 tests reales | `etl/build_duckdb.py` (SQL de la vista de features) y los 3 módulos de `ml_predictions/` (generador sintético, PD+SHAP, LSTM) |

## Próximos pasos

Los 5 módulos ya son funcionales con datos reales. Mejoras pendientes documentadas en cada README de módulo: Temporal Fusion Transformer y ensamble LightGBM en `/ml_predictions`, ventana de datos más larga para los tests econométricos en `/quant_analytics`, y evaluación de Rust/Java como alternativas en `/core_engine` y `/api` respectivamente.

## Autor

Pablo Reyes — Data Scientist, Santiago, Chile.

Licencia: MIT — ver [LICENSE](LICENSE).
