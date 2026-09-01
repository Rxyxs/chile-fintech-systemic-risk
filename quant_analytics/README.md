# Econometría & Clustering (R + Julia)

**Estado:** ✅ funcional, corrido de principio a fin en esta sesión.

## `/r` — Econometría macro

```bash
Rscript quant_analytics/r/macro_econometrics.R   # ejecutar desde la raíz del proyecto
```

Lee `bcch_indicators` y `chile_equity_features` directo de `data/chile_fintech.duckdb`.

**Resultados reales de esta corrida:**
- **Cointegración (Engle-Granger, UF vs USD/CLP):** ADF stat = -1.147 sobre n=21 — **no concluyente** (se necesita ~-3.5 o más negativo). La API pública de `mindicador.cl` solo entrega ~31 observaciones recientes por indicador, ventana insuficiente para un test de cointegración robusto. Documentado, no maquillado.
- **Causalidad de Granger (TPM → retornos):** se salta el test — en la ventana disponible la TPM prácticamente no varía (1 valor único tras forward-fill), así que no hay señal que testear. Mismo problema de ventana de datos.
- **ARIMA(1,0,0)-GARCH(1,1) sobre retornos del activo chileno:** persistencia de volatilidad (α+β) = **0.987** — muy alta, consistente con el hallazgo del LSTM en `/ml_predictions` (51.2% vs 53.6% baseline): los retornos se comportan casi como random walk en dirección, aunque la volatilidad esté fuertemente correlacionada en el tiempo.

**Limitación honesta:** los dos primeros tests dependen de una ventana temporal más larga de indicadores del Banco Central que la que expone la API gratuita de `mindicador.cl` hoy. Quedan implementados y correctos, pero sub-potenciados por la fuente de datos — no es un bug del código.

| Test | Resultado | Interpretación |
|---|---|---|
| Cointegración (Engle-Granger, UF vs USD) | ADF = -1.147 (n=21) | No concluyente — ventana corta |
| Granger (TPM → retornos) | omitido | TPM sin variación en la ventana |
| GARCH(1,1) persistencia (α+β) | 0.987 | Volatilidad muy persistente, retornos ~random walk |

![Precio y volatilidad animados](figures/chile_equity_price_vol_animated.gif)
![Precio y volatilidad del activo chileno](figures/chile_equity_price_vol.png)

La versión animada (`chile_equity_price_vol_animated.gif`) traza ambas series submuestreando los ~2,511 días reales a ~45 frames, con una etiqueta flotante que muestra el valor vigente en cada punto; el PNG estático de arriba queda como referencia fija para lectura detallada.

## `/julia` — Clustering de alta velocidad

```bash
python quant_analytics/julia/export_for_julia.py   # exporta CSVs desde DuckDB/parquet
julia quant_analytics/julia/cluster_profiles.jl
```

K-Medoids (`Clustering.jl` + `Distances.jl`) sobre dos problemas:

1. **Regímenes de volatilidad del activo chileno** (retorno log, SMA-20, volatilidad realizada 20d), k=3. Resultado real: el cluster de mayor volatilidad (0.0218 vs ~0.012 de los otros dos) coincide con retorno promedio negativo (-0.0083) — identifica el régimen de estrés del mercado.
2. **Perfiles de riesgo en la cartera de crédito sintética** (DTI, morosidad previa, ingreso, monto del préstamo), k=4, submuestreado a 5,000 filas (una matriz de distancia par-a-par de 20k×20k son 3.2GB en Float64 — innecesario para este benchmark). Resultado real: el cluster con menor DTI promedio (0.209) tiene la menor tasa de default observada (6.1%) — el clustering separa perfiles de riesgo de forma consistente con la lógica de generación.

| Cluster (crédito) | n | DTI promedio | Tasa de default |
|---|---|---|---|
| 1 | 1,516 | 0.464 | 11.3% |
| 2 | 748 | 0.363 | 9.0% |
| 3 | 1,459 | 0.348 | 13.6% |
| 4 | 1,277 | 0.209 | 6.1% |

Justificación de Julia: cálculo matricial de distancias par-a-par sin el overhead del GIL de Python ni el arranque de la JVM — relevante en este tipo de carga puramente numérica sobre ventanas grandes.

![Espacio de features del activo chileno, coloreado por SMA-20](figures/cluster_feature_space.png)

## Regenerar los gráficos

```bash
python quant_analytics/make_charts.py
```
