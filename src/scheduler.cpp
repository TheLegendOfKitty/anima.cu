#include "scheduler.h"
#include <cmath>
#include <algorithm>
#include <numeric>

float time_snr_shift(float alpha, float t) {
    if (alpha == 1.0f) return t;
    return (t * alpha) / (t * (alpha - 1.0f) + 1.0f);
}

std::vector<float> build_normal_sigmas(int num_steps, int num_train_timesteps, float shift) {
    float multiplier = ANIMA_SAMPLING_MULT;

    // Build base sigmas
    std::vector<float> base_sigmas(num_train_timesteps);
    for (int i = 0; i < num_train_timesteps; i++) {
        float t = ((float)(i + 1) / (float)num_train_timesteps) * multiplier;
        base_sigmas[i] = time_snr_shift(shift, t / multiplier);
    }

    float sigma_min = base_sigmas[0];
    float sigma_max = base_sigmas[num_train_timesteps - 1];
    float start = sigma_max * multiplier;
    float end = sigma_min * multiplier;

    bool append_zero = true;
    float sigma_at_end = time_snr_shift(shift, end / multiplier);
    if (std::abs(sigma_at_end) < 1e-5f) {
        num_steps += 1;
        append_zero = false;
    }

    std::vector<float> sigmas(num_steps);
    for (int i = 0; i < num_steps; i++) {
        float t_val = start + (end - start) * (float)i / (float)(num_steps - 1);
        sigmas[i] = time_snr_shift(shift, t_val / multiplier);
    }

    if (append_zero) {
        sigmas.push_back(0.0f);
    }
    return sigmas;
}

std::vector<float> build_simple_sigmas(int num_steps, int num_train_timesteps, float shift) {
    float multiplier = ANIMA_SAMPLING_MULT;

    std::vector<float> base_sigmas(num_train_timesteps);
    for (int i = 0; i < num_train_timesteps; i++) {
        float t = ((float)(i + 1) / (float)num_train_timesteps) * multiplier;
        base_sigmas[i] = time_snr_shift(shift, t);
    }

    float stride = (float)base_sigmas.size() / (float)num_steps;
    std::vector<float> sigmas;
    for (int i = 0; i < num_steps; i++) {
        int idx = (int)(base_sigmas.size() - 1 - (int)(i * stride));
        idx = std::max(0, std::min(idx, (int)base_sigmas.size() - 1));
        sigmas.push_back(base_sigmas[idx]);
    }
    sigmas.push_back(0.0f);
    return sigmas;
}

// Beta inverse CDF approximation (simplified for practical use)
static double beta_ppf(double p, double a, double b) {
    // Newton's method on the regularized incomplete beta function
    // For the common case a=b=0.6, this converges quickly
    if (p <= 0.0) return 0.0;
    if (p >= 1.0) return 1.0;

    // Initial guess using approximation
    double x = p;
    if (a > 1.0 && b > 1.0) {
        // Use normal approximation for initial guess
        double mu = a / (a + b);
        x = mu;
    }

    // Simple bisection on regularized incomplete beta function
    // Use substitution t = u^(1/a) near 0 to handle singularity for a < 1
    double beta_ab = std::tgamma(a) * std::tgamma(b) / std::tgamma(a + b);

    auto incomplete_beta = [&](double x) -> double {
        if (x <= 0.0) return 0.0;
        if (x >= 1.0) return 1.0;
        // Midpoint rule with many points, skip singularities
        int N = 1000;
        double integral = 0.0;
        for (int i = 0; i < N; i++) {
            double t = x * ((double)i + 0.5) / (double)N;
            double dt = x / (double)N;
            double f = std::pow(t, a - 1.0) * std::pow(1.0 - t, b - 1.0);
            if (std::isfinite(f)) integral += f * dt;
        }
        return integral / beta_ab;
    };

    double lo = 0.0, hi = 1.0;
    for (int iter = 0; iter < 80; iter++) {
        double mid = (lo + hi) / 2.0;
        double cdf = incomplete_beta(mid);
        if (cdf < p) lo = mid;
        else hi = mid;
    }
    return (lo + hi) / 2.0;
}

std::vector<float> build_beta_sigmas(int num_steps, int num_train_timesteps, float shift,
                                      float beta_alpha, float beta_beta) {
    float multiplier = ANIMA_SAMPLING_MULT;

    std::vector<float> base_sigmas(num_train_timesteps);
    for (int i = 0; i < num_train_timesteps; i++) {
        float t = ((float)(i + 1) / (float)num_train_timesteps) * multiplier;
        base_sigmas[i] = time_snr_shift(shift, t);
    }

    int total_timesteps = (int)base_sigmas.size() - 1;

    std::vector<float> sigmas;

    for (int i = 0; i < num_steps; i++) {
        double ts = 1.0 - (double)i / (double)num_steps;
        double mapped = beta_ppf(ts, (double)beta_alpha, (double)beta_beta) * (double)total_timesteps;

        if (std::isnan(mapped)) mapped = 0.0;
        mapped = std::max(0.0, std::min(mapped, (double)total_timesteps));
        int index = (int)std::round(mapped);
        index = std::max(0, std::min(index, total_timesteps));

        sigmas.push_back(base_sigmas[index]);
    }

    sigmas.push_back(0.0f);
    return sigmas;
}

std::vector<float> build_uniform_sigmas(int num_steps, int num_train_timesteps, float shift) {
    // Simple uniform spacing
    std::vector<float> sigmas(num_steps + 1);
    for (int i = 0; i < num_steps; i++) {
        float t = 1.0f - (float)i / (float)num_steps;
        sigmas[i] = time_snr_shift(shift, t);
    }
    sigmas[num_steps] = 0.0f;
    return sigmas;
}
