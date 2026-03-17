#include "cudnn_sdpa.h"
#include "cuda_utils.cuh"

#include <cudnn_frontend.h>
#include <unordered_map>
#include <cstdio>
#include <cstring>

namespace fe = cudnn_frontend;

// ---- Plan cache keyed by (B, H, S_q, S_kv, HD, strides) ----

struct PlanKey {
    int B, H, S_q, S_kv, HD;
    int64_t q_stride[4], kv_stride[4], o_stride[4];
    float scale;
    bool operator==(const PlanKey& o) const {
        return B==o.B && H==o.H && S_q==o.S_q && S_kv==o.S_kv && HD==o.HD
            && scale==o.scale
            && memcmp(q_stride, o.q_stride, sizeof(q_stride)) == 0
            && memcmp(kv_stride, o.kv_stride, sizeof(kv_stride)) == 0
            && memcmp(o_stride, o.o_stride, sizeof(o_stride)) == 0;
    }
};
struct PlanKeyHash {
    size_t operator()(const PlanKey& k) const {
        size_t h = 0;
        auto mix = [&](int64_t v) { h ^= std::hash<int64_t>()(v) + 0x9e3779b9 + (h<<6) + (h>>2); };
        mix(k.B); mix(k.H); mix(k.S_q); mix(k.S_kv); mix(k.HD);
        mix(*(int64_t*)&k.scale);
        for (int i = 0; i < 4; i++) { mix(k.q_stride[i]); mix(k.kv_stride[i]); mix(k.o_stride[i]); }
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

CudnnSDPA::CachedPlan* CudnnSDPA::get_plan(
    int B, int H, int S_q, int S_kv, int HD,
    const int64_t* q_stride, const int64_t* kv_stride, const int64_t* o_stride,
    float attn_scale)
{
    PlanKey key;
    key.B = B; key.H = H; key.S_q = S_q; key.S_kv = S_kv; key.HD = HD;
    key.scale = attn_scale;
    memcpy(key.q_stride, q_stride, 4 * sizeof(int64_t));
    memcpy(key.kv_stride, kv_stride, 4 * sizeof(int64_t));
    memcpy(key.o_stride, o_stride, 4 * sizeof(int64_t));

    auto it = cache_->plans.find(key);
    if (it != cache_->plans.end()) return &it->second;

    fprintf(stderr, "[sdpa] building cuDNN flash attention graph: B=%d H=%d S_q=%d S_kv=%d HD=%d"
            " q_stride=[%ld,%ld,%ld,%ld]\n",
            B, H, S_q, S_kv, HD,
            (long)q_stride[0], (long)q_stride[1], (long)q_stride[2], (long)q_stride[3]);

    CachedPlan plan;
    plan.graph.set_io_data_type(fe::DataType_t::BFLOAT16)
              .set_intermediate_data_type(fe::DataType_t::FLOAT)
              .set_compute_data_type(fe::DataType_t::FLOAT);

    // Q: [B, H, S_q, HD] with caller-specified strides
    plan.Q_attr = plan.graph.tensor(fe::graph::Tensor_attributes()
        .set_name("Q")
        .set_dim({(int64_t)B, (int64_t)H, (int64_t)S_q, (int64_t)HD})
        .set_stride({q_stride[0], q_stride[1], q_stride[2], q_stride[3]})
        .set_uid(1)
        .set_data_type(fe::DataType_t::BFLOAT16));

    // K: [B, H, S_kv, HD]
    plan.K_attr = plan.graph.tensor(fe::graph::Tensor_attributes()
        .set_name("K")
        .set_dim({(int64_t)B, (int64_t)H, (int64_t)S_kv, (int64_t)HD})
        .set_stride({kv_stride[0], kv_stride[1], kv_stride[2], kv_stride[3]})
        .set_uid(2)
        .set_data_type(fe::DataType_t::BFLOAT16));

    // V: [B, H, S_kv, HD]
    plan.V_attr = plan.graph.tensor(fe::graph::Tensor_attributes()
        .set_name("V")
        .set_dim({(int64_t)B, (int64_t)H, (int64_t)S_kv, (int64_t)HD})
        .set_stride({kv_stride[0], kv_stride[1], kv_stride[2], kv_stride[3]})
        .set_uid(3)
        .set_data_type(fe::DataType_t::BFLOAT16));

    auto sdpa_opts = fe::graph::SDPA_attributes()
        .set_name("cosmos_sdpa")
        .set_attn_scale(attn_scale)
        .set_generate_stats(false);

    auto [O, Stats] = plan.graph.sdpa(plan.Q_attr, plan.K_attr, plan.V_attr, sdpa_opts);

    O->set_output(true)
      .set_dim({(int64_t)B, (int64_t)H, (int64_t)S_q, (int64_t)HD})
      .set_stride({o_stride[0], o_stride[1], o_stride[2], o_stride[3]})
      .set_uid(4)
      .set_data_type(fe::DataType_t::BFLOAT16);
    plan.O_attr = O;

    auto status = plan.graph.validate();
    if (!status.is_good()) {
        fprintf(stderr, "[sdpa] graph validation failed: %s\n", status.get_message().c_str());
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
                         const int64_t* q_stride, const int64_t* kv_stride, const int64_t* o_stride,
                         float attn_scale,
                         cudaStream_t stream) {
    auto* plan = get_plan(B, H, S_q, S_kv, HD, q_stride, kv_stride, o_stride, attn_scale);
    if (!plan) {
        fprintf(stderr, "[sdpa] FATAL: no plan available, cannot run attention\n");
        exit(1);
    }

    ensure_workspace(plan->workspace_size);

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
