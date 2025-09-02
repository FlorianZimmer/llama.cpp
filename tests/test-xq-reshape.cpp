#include "ggml.h"
#include "../src/llama-memory-xquant.h"

#include <vector>
#include <cstring>

static ggml_tensor * normalize(ggml_context * ctx, ggml_tensor * t, int64_t d_model) {
    int64_t elems = ggml_nelements(t);
    GGML_ASSERT(elems % d_model == 0);
    int64_t cols = elems / d_model;
    if (t->ne[0] != d_model || t->ne[1] != cols) {
        t = ggml_reshape_2d(ctx, t, d_model, cols);
    }
    return t;
}

static ggml_tensor * xq_dequant_concat_test(ggml_context * ctx,
        const std::vector<llama_memory_xquant::xq_block> & qs,
        const std::vector<llama_memory_xquant_context::pending_write> & pending,
        int32_t il, int64_t d_model) {
    ggml_tensor * cur = nullptr;
    for (const auto & blk : qs) {
        ggml_tensor * qt = ggml_new_tensor_2d(ctx, blk.type, d_model, blk.ne1);
        memcpy(qt->data, blk.data.data(), blk.data.size());
        ggml_tensor * deq = ggml_cast(ctx, qt, GGML_TYPE_F32);
        deq = normalize(ctx, deq, d_model);
        if (!cur) {
            cur = deq;
        } else {
            cur = ggml_concat(ctx, cur, deq, 1);
            cur = normalize(ctx, cur, d_model);
        }
    }
    for (const auto & pw : pending) {
        if (pw.il != il) continue;
        ggml_tensor * deq = ggml_cast(ctx, pw.q, GGML_TYPE_F32);
        deq = normalize(ctx, deq, d_model);
        if (!cur) {
            cur = deq;
        } else {
            cur = ggml_concat(ctx, cur, deq, 1);
            cur = normalize(ctx, cur, d_model);
        }
    }
    return cur;
}

int main() {
    ggml_init_params ip = { 16u * 1024u * 1024u, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return 1;

    const int64_t d_model = 8;

    // 1. Concat Normalization
    {
        ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, 3);
        ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, 5);
        ggml_tensor * cur = ggml_concat(ctx, A, B, 1);
        cur = normalize(ctx, cur, d_model);
        GGML_ASSERT(cur->ne[0] == d_model);
        GGML_ASSERT(cur->ne[1] == 8);
    }

    // 2. Pending + Cached Mix
    {
        using xq_block = llama_memory_xquant::xq_block;
        using pending_write = llama_memory_xquant_context::pending_write;
        std::vector<xq_block> qs;
        std::vector<pending_write> pend;
        for (int n : {2,4}) {
            xq_block blk;
            blk.type = GGML_TYPE_F32;
            blk.ne0 = d_model;
            blk.ne1 = n;
            blk.data.resize((size_t)d_model * n * sizeof(float));
            qs.push_back(std::move(blk));
        }
        ggml_tensor * q_pending = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, 3);
        pend.push_back({0, q_pending});
        ggml_tensor * cur = xq_dequant_concat_test(ctx, qs, pend, 0, d_model);
        cur = normalize(ctx, cur, d_model);
        GGML_ASSERT(cur->ne[0] == d_model);
        GGML_ASSERT(cur->ne[1] == 9);
    }

    // 3. Idempotence
    {
        ggml_tensor * T = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, 7);
        T = normalize(ctx, T, d_model);
        int64_t ne0 = T->ne[0];
        int64_t ne1 = T->ne[1];
        T = normalize(ctx, T, d_model);
        GGML_ASSERT(T->ne[0] == ne0 && T->ne[1] == ne1);
    }

    // 4. Matmul Precondition
    {
        ggml_tensor * wk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, d_model);
        ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, 5);
        ggml_tensor * cur = ggml_concat(ctx, A, A, 1);
        cur = normalize(ctx, cur, d_model);
        ggml_tensor * prod = ggml_mul_mat(ctx, wk, cur);
        GGML_ASSERT(prod->ne[0] == d_model);
        GGML_ASSERT(prod->ne[1] == 10);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, prod);
        ggml_graph_compute_with_ctx(ctx, gf, 1);
    }

    ggml_free(ctx);
    return 0;
}
