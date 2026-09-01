#=
K-Medoids clustering: volatility regimes of Chilean equity returns, and
credit-risk borrower profiles. Reads the CSVs written by export_for_julia.py.

Why Julia: this is pure matrix distance computation over rolling windows —
no ML training loop, no need for Python's ecosystem, and Julia's native
performance on this kind of numeric workload avoids both the Python GIL and
the JVM startup overhead a Java equivalent would carry.
=#

using CSV
using DataFrames
using Clustering
using Distances
using Statistics

const DATA_DIR = joinpath(@__DIR__, "data")
const REPORTS_DIR = joinpath(@__DIR__, "reports")
mkpath(REPORTS_DIR)

function zscore_cols(X::Matrix{Float64})
    mu = mean(X, dims = 1)
    sigma = std(X, dims = 1)
    sigma[sigma .== 0] .= 1.0
    return (X .- mu) ./ sigma
end

# --- 1. Volatility regimes of Chile equity ---
println("=== K-Medoids: Chile equity volatility regimes ===")
equity = CSV.read(joinpath(DATA_DIR, "chile_equity_features.csv"), DataFrame)
eq_features = Matrix{Float64}(equity[:, [:log_return, :sma_20, :realized_vol_20d]])
eq_scaled = zscore_cols(eq_features)
eq_dist = pairwise(Euclidean(), eq_scaled')

k_equity = 3
eq_result = kmedoids(eq_dist, k_equity)
println("n=$(nrow(equity)), k=$k_equity, total cost=$(round(eq_result.totalcost, digits=2))")
for c in 1:k_equity
    idx = findall(==(c), eq_result.assignments)
    avg_vol = mean(equity.realized_vol_20d[idx])
    avg_ret = mean(equity.log_return[idx])
    println("  cluster $c: n=$(length(idx)), avg realized_vol_20d=$(round(avg_vol, digits=5)), avg log_return=$(round(avg_ret, digits=5))")
end

# --- 2. Credit risk borrower profiles ---
println("\n=== K-Medoids: synthetic credit portfolio risk profiles ===")
credit = CSV.read(joinpath(DATA_DIR, "credit_portfolio_synthetic.csv"), DataFrame)
credit_cols = [:dti, :num_prior_delinquencies, :income_clp, :loan_amount_clp]
credit_features = Matrix{Float64}(credit[:, credit_cols])
credit_scaled = zscore_cols(credit_features)

# Subsample for the pairwise distance matrix (20k x 20k is 3.2GB in Float64 — unnecessary for this benchmark)
using Random
Random.seed!(42)
sample_idx = randperm(nrow(credit))[1:5000]
credit_dist = pairwise(Euclidean(), credit_scaled[sample_idx, :]')

k_credit = 4
credit_result = kmedoids(credit_dist, k_credit)
println("n=$(length(sample_idx)) (subsampled from $(nrow(credit))), k=$k_credit, total cost=$(round(credit_result.totalcost, digits=2))")
sampled_default = credit.default[sample_idx]
for c in 1:k_credit
    idx = findall(==(c), credit_result.assignments)
    avg_dti = mean(credit.dti[sample_idx][idx])
    default_rate = mean(sampled_default[idx])
    println("  cluster $c: n=$(length(idx)), avg dti=$(round(avg_dti, digits=3)), default_rate=$(round(default_rate, digits=3))")
end

open(joinpath(REPORTS_DIR, "metrics.txt"), "w") do io
    println(io, "equity_kmedoids_total_cost=$(round(eq_result.totalcost, digits=2))")
    println(io, "credit_kmedoids_total_cost=$(round(credit_result.totalcost, digits=2))")
    println(io, "credit_cluster_default_rates=", join(round.([mean(sampled_default[findall(==(c), credit_result.assignments)]) for c in 1:k_credit], digits=3), ","))
end
println("\n-> quant_analytics/julia/reports/metrics.txt")
