#define private public
#include "../src/llama-memory-xquant.h"
#undef private
#include "../src/llama-model.h"
#include <ggml.h>
#include <ggml-backend.h>
#include <cstdio>
#include <vector>

int main() {
    llama_backend_init();

    const int64_t d_model = 4;
    const int64_t actual_tokens = 3;
    const int64_t claimed_tokens = 5;

    llama_model_params mp = llama_model_default_params();
    llama_model model(mp);
    model.hparams.n_embd = d_model;
    model.hparams.n_layer = 1;

    llama_memory_xquant mem(model);
    auto mctx = mem.init_full();
    auto * ctx = static_cast<llama_memory_xquant_context *>(mctx.get());

    ggml_init_params ip = { 16*1024, nullptr, true };
    ggml_context * gctx = ggml_init(ip);
    if (!gctx) {
        fprintf(stderr, "ggml_init failed\n");
        return 1;
    }

    ggml_tensor * q = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, d_model, actual_tokens);
    size_t nbytes = ggml_nbytes(q);
    std::vector<uint8_t> storage(nbytes);
    ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr(storage.data(), nbytes);
    ggml_backend_tensor_alloc(buf, q, storage.data());

    ctx->pending.push_back({0, q, claimed_tokens});
    ctx->apply();

    ggml_backend_buffer_free(buf);
    ggml_free(gctx);
    llama_backend_free();

    if (mem.layer_data.empty() || mem.layer_data[0].empty()) {
        fprintf(stderr, "block was skipped\n");
        return 1;
    }

    const auto & blk = mem.layer_data[0][0];
    if (blk.ne1 != actual_tokens) {
        fprintf(stderr, "token count mismatch: %lld vs %lld\n", (long long) blk.ne1, (long long) actual_tokens);
        return 1;
    }

    return 0;
}
