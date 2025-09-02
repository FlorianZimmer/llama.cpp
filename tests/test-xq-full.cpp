#include "../src/llama-impl.h"
#include "../src/llama-memory-xquant.h"
#include "../src/llama-model.h"
#include "ggml.h"

#include <vector>
#include <cstring>

struct llama_model_stub {
    llm_type                 type = LLM_TYPE_UNKNOWN;
    llm_arch                 arch = LLM_ARCH_UNKNOWN;
    std::string              name;
    llama_hparams            hparams;
    std::vector<llama_layer> layers;
};

static void fill_identity_f32(ggml_tensor * A) {
    int64_t d0 = A->ne[0];
    int64_t d1 = A->ne[1];
    GGML_ASSERT(A->type == GGML_TYPE_F32 && d0 == d1);
    for (int64_t i = 0; i < d1; ++i) {
        float * row = (float *) ((char *) A->data + i * A->nb[1]);
        for (int64_t j = 0; j < d0; ++j) {
            row[j] = i == j ? 1.0f : 0.0f;
        }
    }
}

static ggml_tensor * normalize_to_dm_by_elements(ggml_context * ctx, ggml_tensor * t, int64_t d_model) {
    int64_t elems = ggml_nelements(t);
    GGML_ASSERT(elems % d_model == 0);
    int64_t cols = elems / d_model;
    if (t->ne[0] != d_model || t->ne[1] != cols) {
        t = ggml_reshape_2d(ctx, t, d_model, cols);
    }
    return t;
}

static ggml_tensor * xq_build_full_x_test(
    ggml_context * ctx,
    const llama_memory_xquant & mem,
    const std::vector<llama_memory_xquant_context::pending_write> & pending,
    int32_t il,
    int64_t d_model) {

    ggml_tensor * cur = nullptr;

    if (mem.layer_data.size() > (size_t) il) {
        for (const auto & blk : mem.layer_data[il]) {
            ggml_tensor * qt = ggml_new_tensor_2d(ctx, blk.type, d_model, blk.ne1);
            memcpy(qt->data, blk.data.data(), blk.data.size());
            ggml_tensor * deq = ggml_cast(ctx, qt, GGML_TYPE_F32);
            deq = normalize_to_dm_by_elements(ctx, deq, d_model);
            if (!ggml_is_contiguous(deq)) {
                deq = ggml_cont(ctx, deq);
            }
            cur = cur ? ggml_concat(ctx, cur, deq, 1) : deq;
            cur = normalize_to_dm_by_elements(ctx, cur, d_model);
        }
    }

    for (const auto & pw : pending) {
        if (pw.il != il) continue;
        ggml_tensor * deq_full = ggml_cast(ctx, pw.q, GGML_TYPE_F32);
        deq_full = normalize_to_dm_by_elements(ctx, deq_full, d_model);
        ggml_tensor * deq_cont = ggml_cont(ctx, deq_full);
        const int64_t cols_full = ggml_nelements(deq_cont) / d_model;
        const int64_t cols_take = pw.n_tokens <= cols_full ? pw.n_tokens : cols_full;
        ggml_tensor * deq = ggml_view_2d(ctx, deq_cont, d_model, cols_take, deq_cont->nb[1], 0);
        deq = normalize_to_dm_by_elements(ctx, deq, d_model);
        cur = cur ? ggml_concat(ctx, cur, deq, 1) : deq;
        cur = normalize_to_dm_by_elements(ctx, cur, d_model);
    }

    if (cur) {
        cur = normalize_to_dm_by_elements(ctx, cur, d_model);
    }
    return cur;
}

static uint32_t count_tokens_for_layer(const llama_memory_xquant & mem,
                                       const std::vector<llama_memory_xquant_context::pending_write> & pending,
                                       int32_t il) {
    uint32_t n = 0;
    if (mem.layer_data.size() > (size_t) il) {
        for (const auto & blk : mem.layer_data[il]) {
            n += (uint32_t) blk.ne1;
        }
    }
    for (const auto & pw : pending) {
        if (pw.il == il) {
            n += (uint32_t) pw.n_tokens;
        }
    }
    return n;
}

int main() {
    const int64_t d_model = 8;

    llama_backend_init();
    ggml_init_params ip = { 64u * 1024u * 1024u, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        llama_backend_free();
        return 1;
    }

    llama_model_stub stub;
    stub.hparams.n_embd = d_model;
    stub.hparams.n_layer = 1;
    stub.hparams.n_rot = d_model;
    stub.hparams.n_embd_head_k = d_model;
    stub.hparams.n_embd_head_v = d_model;
    stub.hparams.n_head_kv_arr[0] = 1;
    stub.layers.resize(1);
    stub.layers[0].wk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, d_model);
    stub.layers[0].wv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, d_model);
    fill_identity_f32(stub.layers[0].wk);
    fill_identity_f32(stub.layers[0].wv);

    // 1. token accounting
    {
        llama_memory_xquant mem(*reinterpret_cast<llama_model*>(&stub));
        mem.layer_data.resize(1);
        llama_memory_xquant::xq_block blk;
        blk.type = GGML_TYPE_F32;
        blk.ne0  = d_model;
        blk.ne1  = 5;
        blk.data.resize((size_t) d_model * blk.ne1 * sizeof(float));
        mem.layer_data[0].push_back(std::move(blk));

        using pending_write = llama_memory_xquant_context::pending_write;
        std::vector<pending_write> pend;
        ggml_tensor * q1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, 2);
        pend.push_back({0, q1, 2});
        ggml_tensor * q2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, 3);
        pend.push_back({0, q2, 3});

        GGML_ASSERT(count_tokens_for_layer(mem, pend, 0) == 10);
        ggml_tensor * X = xq_build_full_x_test(ctx, mem, pend, 0, d_model);
        GGML_ASSERT(X && X->ne[1] == 10);
    }

    // 2. clamping pending slice
    {
        llama_memory_xquant mem(*reinterpret_cast<llama_model*>(&stub));
        std::vector<llama_memory_xquant_context::pending_write> pend;
        ggml_tensor * q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, 8);
        pend.push_back({0, q, 3});
        ggml_tensor * X = xq_build_full_x_test(ctx, mem, pend, 0, d_model);
        GGML_ASSERT(X && X->ne[1] == 3);
    }

    // 3. projection shape
    {
        llama_memory_xquant mem(*reinterpret_cast<llama_model*>(&stub));
        std::vector<llama_memory_xquant_context::pending_write> pend;
        ggml_tensor * q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, 4);
        pend.push_back({0, q, 4});
        ggml_tensor * X = xq_build_full_x_test(ctx, mem, pend, 0, d_model);
        ggml_tensor * K_lin = ggml_mul_mat(ctx, stub.layers[0].wk, X);
        ggml_tensor * V_lin = ggml_mul_mat(ctx, stub.layers[0].wv, X);
        ggml_tensor * K = ggml_reshape_3d(ctx, K_lin, d_model, 1, 4);
        ggml_tensor * V = ggml_reshape_3d(ctx, V_lin, d_model, 1, 4);
        GGML_ASSERT(K->ne[0] == d_model && K->ne[1] == 1 && K->ne[2] == 4);
        GGML_ASSERT(V->ne[0] == d_model && V->ne[1] == 1 && V->ne[2] == 4);
    }

    // 4. rope regression
    {
        llama_memory_xquant mem(*reinterpret_cast<llama_model*>(&stub));
        std::vector<llama_memory_xquant_context::pending_write> pend;
        ggml_tensor * q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, 6);
        pend.push_back({0, q, 6});
        ggml_tensor * X = xq_build_full_x_test(ctx, mem, pend, 0, d_model);
        ggml_tensor * K_lin = ggml_mul_mat(ctx, stub.layers[0].wk, X);
        ggml_tensor * K = ggml_reshape_3d(ctx, K_lin, d_model, 1, 6);
        ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, K->ne[2]);
        for (int i = 0; i < K->ne[2]; ++i) {
            ((int32_t *) pos->data)[i] = i;
        }
        ggml_tensor * rope = ggml_rope(ctx, K, pos, d_model, 0);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, rope);
        ggml_graph_compute_with_ctx(ctx, gf, 1);
    }

    ggml_free(ctx);
    llama_backend_free();
    return 0;
}
