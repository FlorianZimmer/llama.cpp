#include "../src/llama-xq-quant.h"
#include "ggml.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

int main() {
    ggml_init_params   params = { 16 * 1024 * 1024, NULL, false };
    ggml_context *     ctx    = ggml_init(params);
    const int          n      = 128;
    std::vector<float> src(n);
    for (int i = 0; i < n; ++i) {
        src[i] = (float) i - 64.f;
    }
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, 1);
    std::memcpy(t->data, src.data(), n * sizeof(float));

    ggml_tensor * q   = llama_xq_quantize(ctx, t, 4);
    ggml_tensor * deq = ggml_cast(ctx, q, GGML_TYPE_F32);

    float         max_err = 0.f;
    const float * d       = (const float *) deq->data;
    for (int i = 0; i < n; ++i) {
        float err = std::fabs(src[i] - d[i]);
        if (err > max_err) {
            max_err = err;
        }
    }
    ggml_free(ctx);
    assert(max_err < 1.0f);
    return 0;
}
