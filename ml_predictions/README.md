# Modelado Predictivo / Predictive Modeling — PENDIENTE / PENDING

**Estado:** diseñado, no implementado todavía. Se construye en una sesión futura sobre las features ya materializadas en `data/chile_fintech.duckdb` (ver [`/etl`](../etl)).

## Diseño

- **Probability of Default (PD):** ensamble XGBoost/LightGBM sobre variables macro (`bcch_indicators`: TPM, UF, IPC, IMACEC) y features de cartera de crédito. Requiere un dataset transaccional de cartera — no existe una fuente pública chilena a nivel de deudor individual, así que esta pieza usará simulación honesta documentada (como en `chile-credit-risk-scoring-engine` y `catching-credit-card-fraud`), nunca datos reales de personas.
- **Predicción direccional IPSA / volatilidad:** LSTM o Temporal Fusion Transformer en PyTorch sobre `chile_equity_features` (retornos log, SMA-20, volatilidad realizada 20d ya calculados en DuckDB).
- **SHAP obligatorio** sobre el modelo de PD antes de considerar el módulo completo — es un requisito, no un nice-to-have, dado el uso regulatorio.

## Por qué no está aún

El resto del proyecto (motor C++/Rust, R, Julia, Go/Java/C#) requiere priorización explícita del usuario — ver conversación de scoping inicial. Este módulo es el siguiente en la cola.
