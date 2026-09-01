# Econometric analysis of Chilean macro indicators + equity market.
# Reads the same DuckDB store built by etl/build_duckdb.py (real BCCh + ECH data).
#
# Cointegration (Engle-Granger via urca::ur.ers residual test), Granger
# causality (vars::causality), and an ARIMA/GARCH volatility benchmark on
# equity log returns, to compare against the LSTM in ml_predictions/.

suppressMessages({
  library(duckdb)
  library(urca)
  library(vars)
  library(tseries)
  library(rugarch)
})

# Run from the project root: Rscript quant_analytics/r/macro_econometrics.R
root <- getwd()
db_path <- file.path(root, "data", "chile_fintech.duckdb")

con <- dbConnect(duckdb(), dbdir = db_path, read_only = TRUE)

indicators <- dbGetQuery(con, "
  SELECT fecha, indicador, valor FROM bcch_indicators ORDER BY fecha
")
equity <- dbGetQuery(con, "
  SELECT fecha, close, log_return FROM chile_equity_features
  WHERE log_return IS NOT NULL ORDER BY fecha
")
dbDisconnect(con, shutdown = TRUE)

cat(sprintf("Loaded %d indicator rows, %d equity rows\n", nrow(indicators), nrow(equity)))

## --- 1. Cointegration: UF vs dolar (both BCCh series, common macro pair) ---
uf <- indicators[indicators$indicador == "uf", c("fecha", "valor")]
dolar <- indicators[indicators$indicador == "dolar", c("fecha", "valor")]
merged <- merge(uf, dolar, by = "fecha", suffixes = c("_uf", "_dolar"))

cat("\n=== Cointegration: UF vs USD/CLP (Engle-Granger residual ADF) ===\n")
if (nrow(merged) >= 15) {
  eg_reg <- lm(valor_uf ~ valor_dolar, data = merged)
  eg_resid <- residuals(eg_reg)
  adf_test <- ur.df(eg_resid, type = "none", selectlags = "AIC")
  adf_stat <- adf_test@teststat[1]
  cat(sprintf(
    "n=%d, ADF stat on residuals = %.3f (more negative = stronger evidence of cointegration)\n",
    nrow(merged), adf_stat
  ))
} else {
  cat(sprintf("n=%d observations — too few for a reliable cointegration test with this ingestion window; skipping.\n", nrow(merged)))
}

## --- 2. Granger causality: TPM -> equity returns ---
cat("\n=== Granger causality: TPM changes -> equity log returns ===\n")
tpm <- indicators[indicators$indicador == "tpm", c("fecha", "valor")]
tpm$fecha <- as.Date(tpm$fecha)
equity$fecha <- as.Date(equity$fecha)

# TPM is published far less frequently than daily equity closes; align by
# forward-filling TPM onto equity trading days (documented, not hidden).
merged_gc <- merge(equity, tpm, by = "fecha", all.x = TRUE)
merged_gc$valor <- zoo::na.locf(merged_gc$valor, na.rm = FALSE)
merged_gc <- na.omit(merged_gc)

if (nrow(merged_gc) >= 30 && length(unique(merged_gc$valor)) > 1) {
  gc_data <- data.frame(tpm = merged_gc$valor, log_return = merged_gc$log_return)
  var_model <- VAR(gc_data, p = 2, type = "const")
  gc_result <- causality(var_model, cause = "tpm")
  cat(sprintf("Granger F-test p-value (TPM -> log_return): %.4f\n", gc_result$Granger$p.value))
} else {
  cat(sprintf(
    "n=%d aligned obs, %d unique TPM values in this window — insufficient variation for a meaningful Granger test; skipping.\n",
    nrow(merged_gc), length(unique(merged_gc$valor))
  ))
}

## --- 3. ARIMA/GARCH volatility benchmark on equity returns ---
cat("\n=== ARIMA(1,0,0)-GARCH(1,1) on equity log returns ===\n")
returns <- equity$log_return * 100  # scale for numerical stability, standard practice
spec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(1, 0), include.mean = TRUE),
  distribution.model = "std"
)
fit <- ugarchfit(spec, returns, solver = "hybrid")
persistence <- sum(coef(fit)[c("alpha1", "beta1")])
cat(sprintf("n=%d returns, GARCH persistence (alpha1+beta1) = %.4f\n", length(returns), persistence))
cat(sprintf(
  "Interpretation: %s\n",
  if (persistence > 0.9) "high volatility persistence (shocks decay slowly) — consistent with the LSTM in ml_predictions failing to beat the majority-class baseline, since near-random-walk-like returns are hard to predict directionally even with strong volatility clustering."
  else "moderate volatility persistence."
))

out_dir <- file.path(root, "quant_analytics", "r", "reports")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
writeLines(
  c(
    sprintf("cointegration_adf_stat=%s", if (exists("adf_stat")) round(adf_stat, 4) else "NA_insufficient_data"),
    sprintf("granger_tpm_to_returns_pvalue=%s", if (exists("gc_result")) round(gc_result$Granger$p.value, 4) else "NA_insufficient_data"),
    sprintf("garch_persistence=%.4f", persistence)
  ),
  file.path(out_dir, "metrics.txt")
)
cat(sprintf("\n-> %s\n", file.path("quant_analytics", "r", "reports", "metrics.txt")))
