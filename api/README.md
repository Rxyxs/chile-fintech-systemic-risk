# Orquestación & APIs (Go/Java/C#) — PENDIENTE / PENDING

**Estado:** diseñado, no implementado todavía.

## `/go` — Microservicio de predicciones en tiempo real

- Sirve las predicciones de `/ml_predictions` y `/core_engine` vía HTTP, priorizando throughput con goroutines ligeras sobre threads del OS.
- Justificación de Go sobre Python para esta capa: el modelo ya corrió offline (entrenamiento en PyTorch/XGBoost); lo que este servicio necesita es concurrencia barata para muchas requests simultáneas de scoring, no cómputo numérico pesado.

## `/corporate_stubs` — Integración con sistemas legados

- Esqueleto Spring Boot (Java) o .NET Core (C#) que simula la integración con un core bancario chileno típico (SOAP/REST legado). Deliberadamente un stub — no hay un sistema bancario real al que conectarse en este proyecto, así que se documenta como simulación de interfaz, no como servicio funcional con datos reales.

## Por qué no está aún

Es la última capa de la cadena — depende de que exista un modelo real sirviendo predicciones en `/ml_predictions` antes de tener sentido construir el API que las expone.
