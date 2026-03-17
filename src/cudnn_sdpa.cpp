#include "cudnn_sdpa.h"
#include "cuda_utils.cuh"

#include <cudnn_frontend.h>
#include <unordered_map>
#include <cstdio>

namespace fe = cudnn_frontend;

// ---- Plan cache keyed by (B, H, S_q, S_kv, HD) ----

struct PlanKey {
    int B, H, S_q, S_kv, HD;
    float scale;
    bool operator==(const PlanKey& o) const {
        return B==o.B && H==o.H && S_q==o.S_q && S_kv==o.S_kv && HD==o.HD && scale==o.scale;
    }
};
struct PlanKeyHash {
    size_t operator()(const PlanKey& k) const {
        size_t h = 0;
        auto mix = [&](int v) { h ^= std::hash<int>()(v) + 0x9e3779b9 + (h<<6) + (h>>2); };
        mix(k.B); mix(k.H); mix(k.S_q); mix(k.S_kv); mix(k.HD);
        mix(*(int*)&k.scale);
        return h;
    }
};

struct CudnnSDPA::CachedPlan {
    fe::graph::Graph graph;
    std::shared_ptr<fe::graph::Tensor_attributes> Q_attr, K_attr, V_attr, O_attr;
    int64_t workspace_size = 0;
};

struct CudnnSDPA::PlanCache {
    std::unordered_map<PlanKey, CachedPlan, PlanKeyHash> plans;
};

// ---- Lifecycle ----

void CudnnSDPA::init(cudnnHandle_t handle) {
    handle_ = handle;
    if (!cache_) cache_ = new PlanCache();
}

CudnnSDPA::~CudnnSDPA() {
    if (workspace_) cudaFree(workspace_);
    delete cache_;
}

void CudnnSDPA::ensure_workspace(size_t needed) {
    if (needed > workspace_bytes_) {
        if (workspace_) CUDA_CHECK(cudaFree(workspace_));
        CUDA_CHECK(cudaMalloc(&workspace_, needed));
        workspace_bytes_ = needed;
    }
}

// ---- Build or retrieve a cached plan ----

CudnnSDPA::CachedPlan* CudnnSDPA::get_plan(int B, int H, int S_q, int S_kv, int HD, float attn_scale) {
    PlanKey key{B, H, S_q, S_kv, HD, attn_scale};
    auto it = cache_->plans.find(key);
    if (it != cache_->plans.end()) return &it->second;

    fprintf(stderr, "[sdpa] building cuDNN flash attention graph: B=%d H=%d S_q=%d S_kv=%d HD=%d\n",
            B, H, S_q, S_kv, HD);

    CachedPlan plan;
    plan.graph.set_io_data_type(fe::DataType_t::BFLOAT16)
              .set_intermediate_data_type(fe::DataType_t::FLOAT)
              .set_compute_data_type(fe::DataType_t::FLOAT);

    // Q: [B, H, S_q, HD] with strides [H*S_q*HD, S_q*HD, HD, 1]
    plan.Q_attr = plan.graph.tensor(fe::graph::Tensor_attributes()
        .set_name("Q")
        .set_dim({(int64_t)B, (int64_t)H, (int64_t)S_q, (int64_t)HD})
        .set_stride({(int64_t)H*S_q*HD, (int64_t)S_q*HD, (int64_t)HD, 1LL})
        .set_uid(1)
        .set_data_type(fe::DataType_t::BFLOAT16));

    // K: [B, H, S_kv, HD]
    plan.K_attr = plan.graph.tensor(fe::graph::Tensor_attributes()
        .set_name("K")
        .set_dim({(int64_t)B, (int64_t)H, (int64_t)S_kv, (int64_t)HD})
        .set_stride({(int64_t)H*S_kv*HD, (int64_t)S_kv*HD, (int64_t)HD, 1LL})
        .set_uid(2)
        .set_data_type(fe::DataType_t::BFLOAT16));

    // V: [B, H, S_kv, HD]
    plan.V_attr = plan.graph.tensor(fe::graph::Tensor_attributes()
        .set_name("V")
        .set_dim({(int64_t)B, (int64_t)H, (int64_t)S_kv, (int64_t)HD})
        .set_stride({(int64_t)H*S_kv*HD, (int64_t)S_kv*HD, (int64_t)HD, 1LL})
        .set_uid(3)
        .set_data_type(fe::DataType_t::BFLOAT16));

    // SDPA attributes
    auto sdpa_opts = fe::graph::SDPA_attributes()
        .set_name("cosmos_sdpa")
        .set_attn_scale(attn_scale)
        .set_generate_stats(false);

    auto [O, Stats] = plan.graph.sdpa(plan.Q_attr, plan.K_attr, plan.V_attr, sdpa_opts);

    O->set_output(true)
      .set_dim({(int64_t)B, (int64_t)H, (int64_t)S_q, (int64_t)HD})
      .set_stride({(int64_t)H*S_q*HD, (int64_t)S_q*HD, (int64_t)HD, 1LL})
      .set_uid(4)
      .set_data_type(fe::DataType_t::BFLOAT16);
    plan.O_attr = O;

    // Build the graph
    auto status = plan.graph.validate();
    if (!status.is_good()) {
        fprintf(stderr, "[sdpa] graph validation failed: %s\n", status.get_message().c_str());
        fprintf(stderr, "[sdpa] falling back to cuBLAS attention\n");
        return nullptr;
    }

    status = plan.graph.build_operation_graph(handle_);
    if (!status.is_good()) {
        fprintf(stderr, "[sdpa] operation graph build failed: %s\n", status.get_message().c_str());
        return nullptr;
    }

    status = plan.graph.create_execution_plans({fe::HeurMode_t::A});
    if (!status.is_good()) {
        fprintf(stderr, "[sdpa] execution plan creation failed: %s\n", status.get_message().c_str());
        return nullptr;
    }

    status = plan.graph.check_support(handle_);
    if (!status.is_good()) {
        fprintf(stderr, "[sdpa] plan not supported on this device: %s\n", status.get_message().c_str());
        return nullptr;
    }

    status = plan.graph.build_plans(handle_, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);
    if (!status.is_good()) {
        fprintf(stderr, "[sdpa] plan build failed: %s\n", status.get_message().c_str());
        return nullptr;
    }

    plan.workspace_size = plan.graph.get_workspace_size();
    fprintf(stderr, "[sdpa] graph built: workspace=%ld bytes\n", (long)plan.workspace_size);

    auto [ins, ok] = cache_->plans.emplace(key, std::move(plan));
    return &ins->second;
}

// ---- Execute ----

void CudnnSDPA::forward(const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
                         __nv_bfloat16* O,
                         int B, int H, int S_q, int S_kv, int HD,
                         float attn_scale,
                         cudaStream_t stream) {
    auto* plan = get_plan(B, H, S_q, S_kv, HD, attn_scale);
    if (!plan) {
        fprintf(stderr, "[sdpa] FATAL: no plan available, cannot run attention\n");
        exit(1);
    }

    ensure_workspace(plan->workspace_size);

    // Build variant pack mapping tensor attributes to device pointers
    std::unordered_map<int64_t, void*> variant_pack = {
        {plan->Q_attr->get_uid(), const_cast<__nv_bfloat16*>(Q)},
        {plan->K_attr->get_uid(), const_cast<__nv_bfloat16*>(K)},
        {plan->V_attr->get_uid(), const_cast<__nv_bfloat16*>(V)},
        {plan->O_attr->get_uid(), O},
    };

    cudnnSetStream(handle_, stream);
    auto status = plan->graph.execute(handle_, variant_pack, workspace_);
    if (!status.is_good()) {
        fprintf(stderr, "[sdpa] execution failed: %s\n", status.get_message().c_str());
        exit(1);
    }
}
