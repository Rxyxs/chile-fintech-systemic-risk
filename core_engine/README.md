# Motor de Latencia Crítica: Pricing & VaR (C++/Rust/C) — PENDIENTE / PENDING

**Estado:** diseñado, no implementado todavía.

## Diseño

- **Simulación Monte Carlo** para pricing de derivados (opciones sobre IPSA/ECH) y Value at Risk de una cartera sintética, en C++20 con OpenMP para paralelismo multi-hilo — el mismo patrón ya probado y benchmarkeado en [`copper-options-montecarlo-cpp`](https://github.com/Rxyxs/copper-options-montecarlo-cpp) (1M trayectorias en <250ms), adaptado a activos chilenos.
- Alternativa en evaluación: Rust (`rayon` para concurrencia segura en memoria) si el objetivo termina siendo un binding expuesto vía FFI al microservicio Go de `/api`.
- Rutinas SIMD en C puro solo si el profiling muestra que valen la pena frente a la vectorización automática de `-O3` en C++ — no se implementan preventivamente.

## Por qué no está aún

Pendiente de terminar `/ml_predictions` y `/quant_analytics`, que definen qué distribución de retornos y qué parámetros de volatilidad alimentan la simulación Monte Carlo.
