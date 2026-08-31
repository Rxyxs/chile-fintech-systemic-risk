# Econometría & Clustering (R + Julia) — PENDIENTE / PENDING

**Estado:** diseñado, no implementado todavía.

## `/r` — Econometría macro

- Cointegración (Engle-Granger / Johansen) entre TPM, UF, USD/CLP e IMACEC.
- Causalidad de Granger entre tasa de política monetaria y retornos de renta variable chilena.
- ARIMA/GARCH sobre volatilidad de `chile_equity_features` como benchmark econométrico frente a los modelos de deep learning de `/ml_predictions`.
- Fuente de datos: la misma tabla `bcch_indicators` de `data/chile_fintech.duckdb`, leída directo desde R vía el paquete `duckdb`.

## `/julia` — Clustering de alta velocidad

- K-Medoids / HDBSCAN sobre perfiles de riesgo crediticio (una vez exista el dataset simulado de `/ml_predictions`) y sobre regímenes de volatilidad del activo chileno.
- Justificación de Julia: cálculo matricial de alto desempeño sin el overhead de la JVM/Python GIL, relevante cuando el clustering corre sobre ventanas rolling grandes.

## Por qué no está aún

Depende de que `/ml_predictions` genere primero el dataset de riesgo crediticio simulado, y de priorización explícita del usuario para el resto del stack políglota.
