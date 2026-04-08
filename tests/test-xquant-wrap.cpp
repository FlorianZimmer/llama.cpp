#include "llama.h"
#include "ggml.h"

#include "../src/llama-memory-xquant.h"
#include "../src/llama-memory-xquant-wrap.h"

#include <vector>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>

// Fill [d,d] F16 identity
static void fill_identity_f16(ggml_tensor * A) {
    const int64_t d0 = A->ne[0];
    const int64_t d1 = A->ne[1];
    GGML_ASSERT(A->type == GGML_TYPE_F16 && d0 == d1);

    for (int64_t i = 0; i < d1; ++i) {
        auto * row = (ggml_fp16_t *)((char *)A->data + i * A->nb[1]);
        for (int64_t j = 0; j < d0; ++j) {
            row[j] = ggml_fp32_to_fp16(i == j ? 1.0f : 0.0f);
        }
    }
}

int main() {
    const char * model_path = std::getenv("LLAMA_TEST_MODEL");
    if (!model_path || !*model_path) {
        std::fprintf(stderr, "[xquant test] SKIP: set LLAMA_TEST_MODEL to a .gguf path\n");
        return 0; // skip cleanly
    }

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    llama_model * mdl = llama_model_load_from_file(model_path, mp); // updated API
    if (!mdl) {
        std::fprintf(stderr, "[xquant test] FAIL: cannot load model: %s\n", model_path);
        return 1;
    }

    const int32_t n_embd_full = llama_model_n_embd(mdl);
    const int32_t blck        = ggml_blck_size(GGML_TYPE_Q4_0); // usually 32

    // XQuant store expects rows with the model's full hidden size
    const int32_t d      = (n_embd_full / blck) * blck;  // typically equals n_embd_full
    const int32_t T      = 7;
    const int32_t il     = 0;

    // XQuant store (wrapper helpers will also accept the store directly)
    llama_memory_ptr store = llama_memory_make_xquant(mdl, /*n_ctx_tokens*/ T);

    // Build X[T, d] in fp32
    std::vector<float> X((size_t)T * d);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> U(-2.5f, 2.5f);
    for (auto & v : X) v = U(rng);

    // Append rows to layer 0
    bool ok = llama_xquant_wrap_append_prefill_rows(
        store.get(), il, X.data(), /*n_tokens*/ T, /*n_embd*/ d, /*is_fp16*/ false);
    if (!ok) {
        std::fprintf(stderr, "[xquant test] FAIL: append_prefill_rows returned false\n");
        llama_model_free(mdl);
        llama_backend_free();
        return 1;
    }

    // ggml context for tiny graph (128 MiB headroom)
    ggml_init_params ip = {};
    ip.mem_size   = 128u * 1024u * 1024u;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = false;

    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        std::fprintf(stderr, "[xquant test] FAIL: ggml_init\n");
        llama_model_free(mdl);
        llama_backend_free();
        return 1;
    }

    // Identity Wk, Wv : [d, d] F16
    ggml_tensor * Wk = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, d, d);
    ggml_tensor * Wv = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, d, d);
    fill_identity_f16(Wk);
    fill_identity_f16(Wv);

    // Rematerialize K, V for [0, T)
    auto R = llama_xquant_wrap_remat_kv(store.get(), ctx, il, /*t0*/ 0, /*t1*/ T, Wk, Wv);
    if (!R.ok || !R.K || !R.V) {
        std::fprintf(stderr, "[xquant test] FAIL: remat_kv failed\n");
        ggml_free(ctx);
        llama_model_free(mdl);
        llama_backend_free();
        return 1;
    }

    // Normalize shapes to [T, d] first (some backends may yield [d, T])
    ggml_tensor * Knorm = R.K;
    ggml_tensor * Vnorm = R.V;
    if (!(Knorm->ne[0] == d && Knorm->ne[1] == T)) {
        Knorm = ggml_transpose(ctx, Knorm); // [d,T] -> [T,d]
    }
    if (!(Vnorm->ne[0] == d && Vnorm->ne[1] == T)) {
        Vnorm = ggml_transpose(ctx, Vnorm); // [d,T] -> [T,d]
    }
    // Now make contiguous for easy row-wise reading
    ggml_tensor * Kc = ggml_cont(ctx, Knorm);
    ggml_tensor * Vc = ggml_cont(ctx, Vnorm);

    // Single graph, single compute
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, Kc);
    ggml_build_forward_expand(gf, Vc);
    ggml_graph_compute_with_ctx(ctx, gf, /*n_threads*/ 1);

    // Read back K as either F16 or F32 and compare to original X
    std::vector<float> Kf((size_t)T * d);
    if (Kc->type == GGML_TYPE_F16) {
        for (int t = 0; t < T; ++t) {
            const char * base = (const char *)Kc->data + (size_t)t * Kc->nb[1];
            for (int i = 0; i < d; ++i) {
                const ggml_fp16_t * cell = (const ggml_fp16_t *)(base + (size_t)i * Kc->nb[0]);
                Kf[(size_t)t * d + i] = ggml_fp16_to_fp32(*cell);
            }
        }
    } else if (Kc->type == GGML_TYPE_F32) {
        for (int t = 0; t < T; ++t) {
            const char * base = (const char *)Kc->data + (size_t)t * Kc->nb[1];
            for (int i = 0; i < d; ++i) {
                const float * cell = (const float *)(base + (size_t)i * Kc->nb[0]);
                Kf[(size_t)t * d + i] = *cell;
            }
        }
    } else {
        std::fprintf(stderr, "[xquant test] FAIL: unsupported dtype for K: %d\n", (int)Kc->type);
        ggml_free(ctx);
        llama_model_free(mdl);
        llama_backend_free();
        return 1;
    }

    double se = 0.0, ve = 0.0;
    for (size_t i = 0; i < Kf.size(); ++i) {
        const double e = (double)X[i] - (double)Kf[i];
        se += e*e; ve += (double)X[i]*(double)X[i];
    }
    const double rmse  = std::sqrt(se / Kf.size());
    const double nrmse = rmse / std::sqrt(ve / Kf.size());

    std::printf("[xquant test] K vs X: RMSE=%.6f NRMSE=%.6f (d=%d, T=%d)\n", rmse, nrmse, d, T);

    const bool pass = (nrmse < 0.12); // Q4_0 sanity tolerance
    if (!pass) {
        std::fprintf(stderr, "[xquant test] FAIL: NRMSE too high (%.4f)\n", nrmse);
    }

    ggml_free(ctx);
    llama_model_free(mdl);
    llama_backend_free();

    return pass ? 0 : 1;
}