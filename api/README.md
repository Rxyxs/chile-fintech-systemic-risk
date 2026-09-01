# Orquestación & APIs (Go + C#)

**Estado:** ✅ funcional, corrido de principio a fin en esta sesión.

## `/go` — Microservicio de predicciones en tiempo real

```bash
python api/go/export_predictions.py    # batch scoring offline con los modelos ya entrenados
go build -o api/go/predict_server.exe api/go/main.go
api/go/predict_server.exe api/go/data/predictions.json
```

Endpoints (stdlib `net/http`, sin dependencias externas):

- `GET /health`
- `GET /v1/credit/predictions` — 200 solicitudes reales de la cartera sintética, con `pd_score` calculado por el XGBoost entrenado en `/ml_predictions`.
- `GET /v1/equity/snapshot` — último cierre, retorno log, SMA-20, volatilidad realizada del activo chileno (datos reales), más las métricas del LSTM.

**Decisión de arquitectura:** el servicio Go **no** reimplementa inferencia de XGBoost/LSTM — carga predicciones producidas por un job de *batch scoring* offline en Python (`export_predictions.py`). Es el split estándar train/score-offline + serve-online: el valor de Go aquí es concurrencia de requests (goroutines, stdlib `net/http`), no cómputo numérico, que ya se resuelve mejor en Python/C++ en el resto del repo.

**Verificado esta sesión:** `/health`, `/v1/equity/snapshot` y `/v1/credit/predictions` responden con datos reales del pipeline (spot=40.50, PD scores entre ~0.07 y valores más altos según perfil de riesgo).

## `/corporate_stubs` — Integración con sistemas legados (C#/.NET)

```bash
cd api/corporate_stubs/legacy_core_stub && dotnet run
```

Stub explícito (no una integración real) que simula la forma de un core bancario chileno típico: búsqueda de titular de cuenta y decisión de crédito basada en el `pd_score` real del modelo XGBoost (el umbral de decisión usa rangos que el modelo efectivamente produce, ver `ml_predictions/reports/metrics.txt`). Sin I/O de red — es un contrato de integración documentado, no un mock que esconde un bug.

**Java** se evaluó como alternativa (Spring Boot) pero no se implementó: sería el mismo stub en un segundo lenguaje sin justificación adicional, mismo criterio que se aplicó a Rust en `/core_engine`.
