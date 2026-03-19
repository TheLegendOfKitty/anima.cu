#include "spectrum.h"
#include "cuda_utils.cuh"

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cstring>
#include <cmath>

#define CUSOLVER_CHECK(call)                                                    \
    do {                                                                        \
        cusolverStatus_t st = (call);                                           \
        if (st != CUSOLVER_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuSOLVER error at %s:%d: %d\n", __FILE__,         \
                    __LINE__, (int)st);                                         \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// Lazily-initialized file-static cuBLAS handle for Spectrum operations
static cublasHandle_t s_spectrum_cublas = nullptr;

static cublasHandle_t get_spectrum_cublas() {
    if (!s_spectrum_cublas) {
        CUBLAS_CHECK(cublasCreate(&s_spectrum_cublas));
    }
    return s_spectrum_cublas;
}

// ================================================================
// CUDA Kernels
// ================================================================

// Map step indices to tau in [-1, 1] via affine transform
__global__ void normalize_tau_kernel(float* tau, const float* steps, int K,
                                     float t_min, float t_max) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    float t = steps[k];
    if (t_max - t_min < 1e-6f) {
        tau[k] = 0.0f;
    } else {
        float mid = 0.5f * (t_min + t_max);
        float range = t_max - t_min;
        tau[k] = (t - mid) * 2.0f / range;
    }
}

// Build Chebyshev polynomial basis matrix [K, M+1] using 3-term recurrence
__global__ void chebyshev_design_kernel(float* design, const float* tau, int K, int M) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    float tk = tau[k];
    // T_0(tau) = 1
    design[k * (M + 1) + 0] = 1.0f;
    if (M == 0) return;

    // T_1(tau) = tau
    design[k * (M + 1) + 1] = tk;

    // T_m(tau) = 2 * tau * T_{m-1}(tau) - T_{m-2}(tau)
    float T_prev2 = 1.0f;
    float T_prev1 = tk;
    for (int m = 2; m <= M; m++) {
        float T_m = 2.0f * tk * T_prev1 - T_prev2;
        design[k * (M + 1) + m] = T_m;
        T_prev2 = T_prev1;
        T_prev1 = T_m;
    }
}

// Build single prediction row [P] for target tau
__global__ void build_chebyshev_row_kernel(float* design_row, float tau, int M) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    design_row[0] = 1.0f;
    if (M == 0) return;

    design_row[1] = tau;
    float T_prev2 = 1.0f;
    float T_prev1 = tau;
    for (int m = 2; m <= M; m++) {
        float T_m = 2.0f * tau * T_prev1 - T_prev2;
        design_row[m] = T_m;
        T_prev2 = T_prev1;
        T_prev1 = T_m;
    }
}

// Add ridge regularization lambda to diagonal of P×P matrix
__global__ void add_diagonal_kernel(float* matrix, float value, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        matrix[i * n + i] += value;
    }
}

// Predict using fitted Chebyshev coefficients: C^T * design_row -> [F]
// C: [P, F], design_row: [P], out: [F]
__global__ void chebyshev_predict_kernel(const float* C, const float* design_row,
                                          float* out, int P, int F) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= F) return;

    float sum = 0.0f;
    for (int p = 0; p < P; p++) {
        sum += C[p * F + f] * design_row[p];
    }
    out[f] = sum;
}

// Newton forward difference extrapolation using last 2-4 features
__global__ void taylor_predict_kernel(
    const float* H, const float* steps, float* out,
    int K, int F, int target_step, int order)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= F) return;

    if (K < 2) {
        out[f] = H[(K - 1) * F + f];
        return;
    }

    float h_i = H[(K - 1) * F + f];
    float h_im1 = H[(K - 2) * F + f];
    float t_i = steps[K - 1];
    float t_im1 = steps[K - 2];

    float dh1 = h_i - h_im1;
    float dt_last = fmaxf(t_i - t_im1, 1e-6f);
    float k_step = (float)(target_step - (int)t_i) / dt_last;

    float pred = h_i + k_step * dh1;

    // Second order
    if (order >= 2 && K >= 3) {
        float h_im2 = H[(K - 3) * F + f];
        float d2 = h_i - 2.0f * h_im1 + h_im2;
        pred += 0.5f * k_step * (k_step - 1.0f) * d2;
    }

    // Third order
    if (order >= 3 && K >= 4) {
        float h_im2 = H[(K - 3) * F + f];
        float h_im3 = H[(K - 4) * F + f];
        float d3 = h_i - 3.0f * h_im1 + 3.0f * h_im2 - h_im3;
        pred += (k_step * (k_step - 1.0f) * (k_step - 2.0f) / 6.0f) * d3;
    }

    out[f] = pred;
}

// Blend: out = (1-w)*taylor + w*cheb
__global__ void blend_predictions_kernel(
    const float* cheb_pred, const float* taylor_pred, float* out,
    float blend_weight, int F)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= F) return;

    out[f] = (1.0f - blend_weight) * taylor_pred[f] + blend_weight * cheb_pred[f];
}

// Element-wise subtraction: c = a - b
__global__ void vector_sub_kernel(const float* a, const float* b, float* c, int64_t N) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) c[i] = a[i] - b[i];
}

// Element-wise addition: c = a + b (c may alias a)
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int64_t N) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) c[i] = a[i] + b[i];
}

// ================================================================
// Ridge Regression: (X^T X + lambda*I) C = X^T H
// X: [K, P] row-major, H: [K, F] row-major, C: [P, F] row-major
// ================================================================

static void spectrum_fit_ridge_regression(
    const float* X, const float* H, float* C,
    int K, int P, int F, float lambda, cudaStream_t stream)
{
    cublasHandle_t cublas = get_spectrum_cublas();
    CUBLAS_CHECK(cublasSetStream(cublas, stream));

    float* XtX = nullptr;
    CUDA_CHECK(cudaMallocAsync(&XtX, (size_t)P * P * sizeof(float), stream));

    const float one = 1.0f;
    const float zero = 0.0f;

    // X is row-major [K, P] = column-major [P, K]
    // X^T X: cublasSgemm(N, T, P, P, K, ..., X, P, X, P, ..., XtX, P)
    CUBLAS_CHECK(cublasSgemm(
        cublas, CUBLAS_OP_N, CUBLAS_OP_T,
        P, P, K,
        &one, X, P, X, P,
        &zero, XtX, P));

    // Add ridge regularization
    add_diagonal_kernel<<<1, 32, 0, stream>>>(XtX, lambda, P);

    // C is row-major [P, F] = column-major [F, P]
    // Compute (X^T H)^T = H^T X into C as column-major [F, P]
    CUBLAS_CHECK(cublasSgemm(
        cublas, CUBLAS_OP_N, CUBLAS_OP_T,
        F, P, K,
        &one, H, F, X, P,
        &zero, C, F));

    // Cholesky factorization of XtX (P×P)
    cusolverDnHandle_t solver;
    CUSOLVER_CHECK(cusolverDnCreate(&solver));
    CUSOLVER_CHECK(cusolverDnSetStream(solver, stream));

    int workspace_size = 0;
    CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(
        solver, CUBLAS_FILL_MODE_UPPER, P, XtX, P, &workspace_size));

    float* workspace = nullptr;
    int* info = nullptr;
    CUDA_CHECK(cudaMallocAsync(&workspace, (size_t)workspace_size * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&info, sizeof(int), stream));

    CUSOLVER_CHECK(cusolverDnSpotrf(
        solver, CUBLAS_FILL_MODE_UPPER, P, XtX, P, workspace, workspace_size, info));

    int host_info = 0;
    CUDA_CHECK(cudaMemcpyAsync(&host_info, info, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (host_info != 0) {
        fprintf(stderr, "[spectrum] ridge Cholesky failed (info=%d)\n", host_info);
        exit(1);
    }

    // Triangular solves: (R^T R) C = X^T H
    // Two-step: R^T Y = X^T H, then R C = Y
    CUBLAS_CHECK(cublasStrsm(
        cublas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
        F, P, &one, XtX, P, C, F));
    CUBLAS_CHECK(cublasStrsm(
        cublas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
        F, P, &one, XtX, P, C, F));

    CUSOLVER_CHECK(cusolverDnDestroy(solver));
    CUDA_CHECK(cudaFreeAsync(workspace, stream));
    CUDA_CHECK(cudaFreeAsync(info, stream));
    CUDA_CHECK(cudaFreeAsync(XtX, stream));
}

// ================================================================
// High-level Spectrum functions
// ================================================================

void spectrum_init(SpectrumState& state, int feature_dim, int spatial_size,
                   int max_buffer_size, int max_order, bool use_residual,
                   cudaStream_t stream) {
    state.feature_dim = feature_dim;
    state.spatial_size = spatial_size;
    state.max_buffer_size = max_buffer_size;
    state.max_order = max_order;
    state.buffer_size = 0;
    state.coeffs_valid = false;
    state.last_actual_step = -1;
    state.consecutive_cached = 0;
    state.use_residual = use_residual;
    state.has_last_actual = false;
    state.last_quality = 1.0f;
    state.force_actual = false;

    int64_t F = (int64_t)feature_dim * spatial_size;
    int P = max_order + 1;

    CUDA_CHECK(cudaMallocAsync(&state.feature_buffer, (size_t)max_buffer_size * F * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&state.step_buffer, (size_t)max_buffer_size * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&state.cheb_coeffs, (size_t)P * F * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&state.design_matrix, (size_t)max_buffer_size * P * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&state.tau_buffer, (size_t)max_buffer_size * sizeof(float), stream));

    if (use_residual) {
        CUDA_CHECK(cudaMallocAsync(&state.last_actual_feature, F * sizeof(float), stream));
        CUDA_CHECK(cudaMallocAsync(&state.delta_tmp, F * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(state.last_actual_feature, 0, F * sizeof(float), stream));
    }

    CUDA_CHECK(cudaMemsetAsync(state.feature_buffer, 0, (size_t)max_buffer_size * F * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(state.step_buffer, 0, (size_t)max_buffer_size * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(state.cheb_coeffs, 0, (size_t)P * F * sizeof(float), stream));

    fprintf(stderr, "[spectrum] init: feature_dim=%d, spatial_size=%d, F=%ld, max_buffer=%d, order=%d, residual=%s\n",
            feature_dim, spatial_size, (long)F, max_buffer_size, max_order,
            use_residual ? "yes" : "no");
}

void spectrum_reset(SpectrumState& state) {
    state.buffer_size = 0;
    state.coeffs_valid = false;
    state.last_actual_step = -1;
    state.consecutive_cached = 0;
    state.has_last_actual = false;
    state.last_quality = 1.0f;
    state.force_actual = false;
}

void SpectrumState::free_all() {
    if (feature_buffer) { cudaFree(feature_buffer); feature_buffer = nullptr; }
    if (step_buffer) { cudaFree(step_buffer); step_buffer = nullptr; }
    if (cheb_coeffs) { cudaFree(cheb_coeffs); cheb_coeffs = nullptr; }
    if (design_matrix) { cudaFree(design_matrix); design_matrix = nullptr; }
    if (tau_buffer) { cudaFree(tau_buffer); tau_buffer = nullptr; }
    if (last_actual_feature) { cudaFree(last_actual_feature); last_actual_feature = nullptr; }
    if (delta_tmp) { cudaFree(delta_tmp); delta_tmp = nullptr; }
}

void spectrum_update(SpectrumState& state, const float* feature, int step,
                     cudaStream_t stream) {
    int64_t F = (int64_t)state.feature_dim * state.spatial_size;
    float step_f = (float)step;

    // In residual mode, the first call just saves the reference — no buffer entry
    if (state.use_residual && !state.has_last_actual) {
        CUDA_CHECK(cudaMemcpyAsync(state.last_actual_feature, feature,
                                    F * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        state.has_last_actual = true;
        state.last_actual_step = step;
        state.consecutive_cached = 0;
        state.coeffs_valid = false;
        return;
    }

    // Determine what to store: delta (residual mode) or absolute feature
    const float* to_store = feature;
    if (state.use_residual) {
        int block = 256;
        int grid = ((int)F + block - 1) / block;
        vector_sub_kernel<<<grid, block, 0, stream>>>(
            feature, state.last_actual_feature, state.delta_tmp, F);
        to_store = state.delta_tmp;
    }

    // Buffer management: append or shift
    if (state.buffer_size < state.max_buffer_size) {
        CUDA_CHECK(cudaMemcpyAsync(
            state.step_buffer + state.buffer_size,
            &step_f, sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(
            state.feature_buffer + (size_t)state.buffer_size * F,
            to_store, F * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        state.buffer_size++;
    } else {
        // Shift buffer left by one and append at end
        size_t shift_features = (size_t)(state.max_buffer_size - 1) * F * sizeof(float);
        size_t shift_steps = (size_t)(state.max_buffer_size - 1) * sizeof(float);

        // Use memmove-style: copy [1..end] to [0..end-1] via temp buffers
        float* temp_feature = nullptr;
        float* temp_step = nullptr;
        CUDA_CHECK(cudaMallocAsync(&temp_feature, shift_features, stream));
        CUDA_CHECK(cudaMallocAsync(&temp_step, shift_steps, stream));

        CUDA_CHECK(cudaMemcpyAsync(temp_feature, state.feature_buffer + F,
                                    shift_features, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(temp_step, state.step_buffer + 1,
                                    shift_steps, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(state.feature_buffer, temp_feature,
                                    shift_features, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(state.step_buffer, temp_step,
                                    shift_steps, cudaMemcpyDeviceToDevice, stream));

        CUDA_CHECK(cudaFreeAsync(temp_feature, stream));
        CUDA_CHECK(cudaFreeAsync(temp_step, stream));

        CUDA_CHECK(cudaMemcpyAsync(
            state.step_buffer + state.max_buffer_size - 1,
            &step_f, sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(
            state.feature_buffer + (size_t)(state.max_buffer_size - 1) * F,
            to_store, F * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    }

    // Update reference for residual mode
    if (state.use_residual) {
        CUDA_CHECK(cudaMemcpyAsync(state.last_actual_feature, feature,
                                    F * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    }

    state.last_actual_step = step;
    state.consecutive_cached = 0;
    state.coeffs_valid = false;
}

void spectrum_predict(SpectrumState& state, float* output, int target_step,
                      float blend_weight, cudaStream_t stream) {
    int K = state.buffer_size;
    int64_t F = (int64_t)state.feature_dim * state.spatial_size;

    if (K < 2) {
        fprintf(stderr, "[spectrum] insufficient history (K=%d) for prediction\n", K);
        return;
    }

    // Clamp effective order to prevent overfitting: need K >= P+2 for stable fit
    int max_safe_order = (K >= 4) ? K - 3 : 0;
    int effective_order = (state.max_order < max_safe_order) ? state.max_order : max_safe_order;
    int P = effective_order + 1;

    // If not enough data for Chebyshev, use Taylor-only
    if (effective_order < 1) {
        float* d_taylor_pred = nullptr;
        CUDA_CHECK(cudaMallocAsync(&d_taylor_pred, F * sizeof(float), stream));
        int block_size = 256;
        int grid = ((int)F + block_size - 1) / block_size;
        taylor_predict_kernel<<<grid, block_size, 0, stream>>>(
            state.feature_buffer, state.step_buffer, d_taylor_pred, K, (int)F, target_step, 1);
        CUDA_CHECK(cudaMemcpyAsync(output, d_taylor_pred, F * sizeof(float),
                                    cudaMemcpyDeviceToDevice, stream));
        // In residual mode, add back the last actual feature
        if (state.use_residual && state.has_last_actual) {
            vector_add_kernel<<<grid, block_size, 0, stream>>>(
                output, state.last_actual_feature, output, F);
        }
        CUDA_CHECK(cudaFreeAsync(d_taylor_pred, stream));
        return;
    }

    int block_size = 256;
    int grid;

    // Get tau range from step buffer (filled in order: [0] is min, [K-1] is max)
    float t_min, t_max;
    CUDA_CHECK(cudaMemcpyAsync(&t_min, state.step_buffer, sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&t_max, state.step_buffer + (K - 1), sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Step 1: Normalize tau values
    grid = (K + block_size - 1) / block_size;
    normalize_tau_kernel<<<grid, block_size, 0, stream>>>(
        state.tau_buffer, state.step_buffer, K, t_min, t_max);

    // Step 2: Build Chebyshev design matrix [K, P]
    chebyshev_design_kernel<<<grid, block_size, 0, stream>>>(
        state.design_matrix, state.tau_buffer, K, P - 1);

    // Step 3: Fit Chebyshev coefficients via ridge regression
    spectrum_fit_ridge_regression(
        state.design_matrix, state.feature_buffer, state.cheb_coeffs,
        K, P, (int)F, state.ridge_lambda, stream);
    state.coeffs_valid = true;

    // Step 4: Build design row for target step
    float tau_target;
    {
        float mid = 0.5f * (t_min + t_max);
        float range = t_max - t_min;
        tau_target = (range < 1e-6f) ? 0.0f : ((float)target_step - mid) * 2.0f / range;
    }

    float* d_design_row = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_design_row, P * sizeof(float), stream));
    build_chebyshev_row_kernel<<<1, 1, 0, stream>>>(d_design_row, tau_target, P - 1);

    // Step 5: Chebyshev prediction
    float* d_cheb_pred = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_cheb_pred, F * sizeof(float), stream));
    grid = ((int)F + block_size - 1) / block_size;
    chebyshev_predict_kernel<<<grid, block_size, 0, stream>>>(
        state.cheb_coeffs, d_design_row, d_cheb_pred, P, (int)F);

    // Step 6: Taylor prediction
    float* d_taylor_pred = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_taylor_pred, F * sizeof(float), stream));
    taylor_predict_kernel<<<grid, block_size, 0, stream>>>(
        state.feature_buffer, state.step_buffer, d_taylor_pred, K, (int)F, target_step, 1);

    // Step 7: Blend predictions
    blend_predictions_kernel<<<grid, block_size, 0, stream>>>(
        d_cheb_pred, d_taylor_pred, output, blend_weight, (int)F);

    // Step 8: In residual mode, add back the last actual feature to get absolute prediction
    if (state.use_residual && state.has_last_actual) {
        vector_add_kernel<<<grid, block_size, 0, stream>>>(
            output, state.last_actual_feature, output, F);
    }

    CUDA_CHECK(cudaFreeAsync(d_design_row, stream));
    CUDA_CHECK(cudaFreeAsync(d_cheb_pred, stream));
    CUDA_CHECK(cudaFreeAsync(d_taylor_pred, stream));
}

// ================================================================
// Cosine similarity via cuBLAS dot products
// ================================================================

float spectrum_cosine_similarity(const float* a, const float* b, int64_t N,
                                  cudaStream_t stream) {
    cublasHandle_t cublas = get_spectrum_cublas();
    CUBLAS_CHECK(cublasSetStream(cublas, stream));

    float dot_ab = 0.0f, dot_aa = 0.0f, dot_bb = 0.0f;
    int n = (int)N;

    CUBLAS_CHECK(cublasSdot(cublas, n, a, 1, b, 1, &dot_ab));
    CUBLAS_CHECK(cublasSdot(cublas, n, a, 1, a, 1, &dot_aa));
    CUBLAS_CHECK(cublasSdot(cublas, n, b, 1, b, 1, &dot_bb));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float denom = sqrtf(dot_aa * dot_bb);
    return (denom > 1e-8f) ? dot_ab / denom : 0.0f;
}
