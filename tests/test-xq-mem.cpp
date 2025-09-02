#include "../src/llama-memory-xquant.h"
#include "../src/llama-model.h"
#include "ggml.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void fill_identity_f16(ggml_tensor * A) {
    const int64_t d0 = A->ne[0];
    const int64_t d1 = A->ne[1];
    GGML_ASSERT(A->type == GGML_TYPE_F16 && d0 == d1);
    for (int64_t i = 0; i < d1; ++i) {
        auto * row = (ggml_fp16_t *) ((char *) A->data + i * A->nb[1]);
        for (int64_t j = 0; j < d0; ++j) {
            row[j] = ggml_fp32_to_fp16(i == j ? 1.0f : 0.0f);
        }
    }
}

int main() {
    const char * model_path = std::getenv("LLAMA_TEST_MODEL");
    if (!model_path || !*model_path) {
        std::fprintf(stderr, "[xq mem test] SKIP: set LLAMA_TEST_MODEL to a .gguf path\n");
        return 0;
    }

    llama_backend_init();

    llama_model_params mp  = llama_model_default_params();
    llama_model *      mdl = llama_model_load_from_file(model_path, mp);
    if (!mdl) {
        std::fprintf(stderr, "[xq mem test] FAIL: cannot load model: %s\n", model_path);
        return 1;
    }

    const int32_t d = llama_model_n_embd(mdl);
    const int32_t T = 7;

    llama_memory_xquant mem(*mdl);
    auto                mctx   = mem.init_full();
    auto *              xq_ctx = static_cast<llama_memory_xquant_context *>(mctx.get());

    // replace layer0 Wk/Wv with identity so K/V should match input X
    ggml_tensor * wk = mdl->layers[0].wk;
    ggml_tensor * wv = mdl->layers[0].wv;
    if (wk->type != GGML_TYPE_F16 || wv->type != GGML_TYPE_F16) {
        std::fprintf(stderr, "[xq mem test] SKIP: wk/wv must be F16 (got %d/%d)\n", (int) wk->type, (int) wv->type);
        llama_model_free(mdl);
        llama_backend_free();
        return 0;
    }
    fill_identity_f16(wk);
    fill_identity_f16(wv);

    ggml_init_params ip  = { 128u * 1024u * 1024u, nullptr, false };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) {
        std::fprintf(stderr, "[xq mem test] FAIL: ggml_init\n");
        llama_model_free(mdl);
        llama_backend_free();
        return 1;
    }

    // build random X[d,T]
    std::vector<float>                    X((size_t) d * T);
    std::mt19937                          rng(42);
    std::uniform_real_distribution<float> dist(-2.5f, 2.5f);
    for (auto & v : X) {
        v = dist(rng);
    }
    ggml_tensor * Xt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, T);
    memcpy(Xt->data, X.data(), X.size() * sizeof(float));

    int           bits = 4;
    ggml_tensor * q    = xq_ctx->write(ctx, Xt, 0, bits);
    GGML_ASSERT(q->type == llama_xq_bits_to_type(bits));
    ggml_tensor * K  = xq_ctx->get_k(ctx, 0);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, K);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    // read back K
    std::vector<float> Kf((size_t) d * T);
    if (K->type == GGML_TYPE_F16) {
        for (int t = 0; t < T; ++t) {
            const char * base = (const char *) K->data + (size_t) t * K->nb[1];
            for (int i = 0; i < d; ++i) {
                const ggml_fp16_t * cell = (const ggml_fp16_t *) (base + (size_t) i * K->nb[0]);
                Kf[(size_t) t * d + i]   = ggml_fp16_to_fp32(*cell);
            }
        }
    } else if (K->type == GGML_TYPE_F32) {
        memcpy(Kf.data(), K->data, Kf.size() * sizeof(float));
    } else {
        std::fprintf(stderr, "[xq mem test] FAIL: unsupported dtype %d\n", (int) K->type);
        ggml_free(ctx);
        llama_model_free(mdl);
        llama_backend_free();
        return 1;
    }

    double se = 0.0, ve = 0.0;
    for (size_t i = 0; i < Kf.size(); ++i) {
        double e = (double) X[i] - (double) Kf[i];
        se += e * e;
        ve += (double) X[i] * (double) X[i];
    }
    double rmse  = std::sqrt(se / Kf.size());
    double nrmse = rmse / std::sqrt(ve / Kf.size());

    std::printf("[xq mem test] RMSE=%.6f NRMSE=%.6f\n", rmse, nrmse);

    ggml_free(ctx);
    llama_model_free(mdl);
    llama_backend_free();

    if (nrmse >= 0.12) {
        std::fprintf(stderr, "[xq mem test] FAIL: NRMSE too high\n");
        return 1;
    }
    return 0;
}
