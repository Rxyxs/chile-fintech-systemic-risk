// Monte Carlo engine for European option pricing (GBM) and portfolio VaR/ES.
// Multi-threaded with OpenMP. Parameters (spot, realized vol) are read from a
// small CSV exported by core_engine/cpp/export_params.py, sourced from the
// same real chile_equity_features view used across this project.
//
// Build (MSVC, from a Developer Command Prompt / after vcvars64.bat):
//   cl /O2 /openmp /EHsc montecarlo_var.cpp /Fe:montecarlo_var.exe
// Build (g++):
//   g++ -O3 -fopenmp montecarlo_var.cpp -o montecarlo_var

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

struct MarketParams {
    double spot = 0.0;
    double annualized_vol = 0.0;
    double risk_free_rate = 0.05;  // approx. Chilean TPM-adjacent real rate, documented assumption
};

MarketParams read_params(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("could not open params file: " + path);
    }
    MarketParams params;
    std::string line;
    std::getline(file, line);  // header
    std::getline(file, line);  // single data row: spot,annualized_vol
    std::stringstream ss(line);
    std::string field;
    std::getline(ss, field, ',');
    params.spot = std::stod(field);
    std::getline(ss, field, ',');
    params.annualized_vol = std::stod(field);
    return params;
}

// Prices a European call via GBM Monte Carlo and computes 1-day 99% VaR/ES
// for a long position of `n_shares` in the underlying.
struct SimResult {
    double call_price = 0.0;
    double call_stderr = 0.0;
    double var_99 = 0.0;
    double es_99 = 0.0;
    long paths = 0;
    double seconds = 0.0;
};

SimResult run_simulation(const MarketParams& p, double strike, double maturity_years,
                          long n_paths, double position_notional) {
    const double drift = (p.risk_free_rate - 0.5 * p.annualized_vol * p.annualized_vol) * maturity_years;
    const double diffusion = p.annualized_vol * std::sqrt(maturity_years);
    const double daily_vol = p.annualized_vol / std::sqrt(252.0);

    std::vector<double> payoffs(n_paths);
    std::vector<double> pnl_1d(n_paths);

    auto t0 = std::chrono::high_resolution_clock::now();

#pragma omp parallel
    {
        unsigned seed = 42u;
#ifdef _OPENMP
        seed += static_cast<unsigned>(omp_get_thread_num());
#endif
        std::mt19937_64 rng(seed);
        std::normal_distribution<double> norm(0.0, 1.0);

#pragma omp for
        for (long i = 0; i < n_paths; ++i) {
            double z_terminal = norm(rng);
            double s_terminal = p.spot * std::exp(drift + diffusion * z_terminal);
            payoffs[i] = std::max(s_terminal - strike, 0.0) * std::exp(-p.risk_free_rate * maturity_years);

            double z_1d = norm(rng);
            double s_1d = p.spot * std::exp(-0.5 * daily_vol * daily_vol + daily_vol * z_1d);
            pnl_1d[i] = position_notional * (s_1d / p.spot - 1.0);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    double mean_payoff = 0.0;
    for (double v : payoffs) mean_payoff += v;
    mean_payoff /= static_cast<double>(n_paths);

    double sq_sum = 0.0;
    for (double v : payoffs) sq_sum += (v - mean_payoff) * (v - mean_payoff);
    double stderr_payoff = std::sqrt(sq_sum / static_cast<double>(n_paths)) / std::sqrt(static_cast<double>(n_paths));

    std::vector<double> sorted_pnl = pnl_1d;
    std::sort(sorted_pnl.begin(), sorted_pnl.end());
    long var_idx = static_cast<long>(0.01 * static_cast<double>(n_paths));
    double var_99 = -sorted_pnl[var_idx];

    double es_sum = 0.0;
    for (long i = 0; i <= var_idx; ++i) es_sum += sorted_pnl[i];
    double es_99 = -(es_sum / static_cast<double>(var_idx + 1));

    SimResult result;
    result.call_price = mean_payoff;
    result.call_stderr = stderr_payoff;
    result.var_99 = var_99;
    result.es_99 = es_99;
    result.paths = n_paths;
    result.seconds = std::chrono::duration<double>(t1 - t0).count();
    return result;
}

int main(int argc, char** argv) {
    std::string params_path = "core_engine/cpp/data/market_params.csv";
    if (argc > 1) params_path = argv[1];

    long n_paths = 1'000'000;
    double maturity_years = 30.0 / 365.0;  // 30-day option, matches the LSTM's 30s->daily horizon framing loosely
    double position_notional = 1'000'000.0;  // CLP-equivalent notional for the VaR leg

    try {
        MarketParams params = read_params(params_path);
        double strike = params.spot * 1.02;  // 2% OTM call, illustrative

        std::cout << "spot=" << params.spot << " annualized_vol=" << params.annualized_vol
                  << " strike=" << strike << " paths=" << n_paths << "\n";

        SimResult r = run_simulation(params, strike, maturity_years, n_paths, position_notional);

        std::cout << "call_price=" << r.call_price << " (stderr=" << r.call_stderr << ")\n";
        std::cout << "var_99_1d=" << r.var_99 << " es_99_1d=" << r.es_99 << "\n";
        std::cout << "elapsed_seconds=" << r.seconds << " paths=" << r.paths << "\n";
#ifdef _OPENMP
        std::cout << "openmp_threads=" << omp_get_max_threads() << "\n";
#else
        std::cout << "openmp_threads=1 (built without OpenMP)\n";
#endif
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
