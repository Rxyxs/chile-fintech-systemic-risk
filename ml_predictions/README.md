# Modelado Predictivo / Predictive Modeling

**Estado:** ✅ funcional, corrido de principio a fin en esta sesión.

## Probability of Default — XGBoost + SHAP

No existe una fuente pública chilena de datos de crédito a nivel de deudor individual (con razón — son datos personales). `simulate_credit_portfolio.py` genera una cartera **sintética documentada**: 20,000 solicitantes, default como función logística de DTI, morosidad previa, empleo formal, TPM de originación e ingreso, más ruido gaussiano — igual que en `chile-credit-risk-scoring-engine` y `catching-credit-card-fraud`, nunca datos reales de personas.

`train_pd_model.py` entrena XGBoost y calcula SHAP (obligatorio, no opcional, dado el uso regulatorio):

```bash
python ml_predictions/simulate_credit_portfolio.py
python ml_predictions/train_pd_model.py
```

**Resultado real de esta corrida:** AUC held-out = **0.624**. Features dominantes por SHAP: `dti` y `num_prior_delinquencies` (coherente con la lógica de generación — no es casualidad, es una verificación de que el pipeline SHAP funciona correctamente sobre una señal conocida).

| Métrica | Valor |
|---|---|
| AUC held-out | 0.624 |
| n_train / n_test | 15,000 / 5,000 |
| Tasa de default (cartera) | 10.01% |
| Feature #1 por SHAP | `dti` (0.271) |
| Feature #2 por SHAP | `num_prior_delinquencies` (0.269) |

![SHAP feature importance](reports/figures/shap_importance.png)
![PD score distribution](reports/figures/pd_score_distribution.png)

## Predicción direccional — LSTM sobre el activo chileno

`train_lstm_equity.py` lee `chile_equity_features` (retorno log, SMA-20, volatilidad realizada 20d) directo de `data/chile_fintech.duckdb` y entrena un LSTM para predecir la dirección del retorno del día siguiente.

```bash
python ml_predictions/train_lstm_equity.py
```

**Hallazgo honesto de esta corrida:** accuracy de test = **51.2%**, por debajo del baseline de clase mayoritaria (**53.6%**). El LSTM no supera a simplemente predecir la clase más frecuente — mismo patrón que en `reading-market-turbulence`, donde el baseline de persistencia también le gana a los modelos entrenados. Se documenta el resultado real, no se descarta ni se maquilla.

| Métrica | LSTM | Baseline (clase mayoritaria) |
|---|---|---|
| Accuracy (test) | 51.2% | 53.6% |
| n_train / n_test | ~1,990 / ~498 | — |

![LSTM vs baseline](reports/figures/lstm_vs_baseline.png)

## Pendiente

- Temporal Fusion Transformer (arquitectura más compleja que LSTM simple) como siguiente intento de superar el baseline.
- Ensamble LightGBM como challenger del XGBoost de PD.

## Regenerar los gráficos

```bash
python ml_predictions/make_charts.py
```
