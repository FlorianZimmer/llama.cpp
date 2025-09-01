#pragma once

#include "ggml.h"
#include "../ggml/src/ggml-quants.h"

// map bit width to ggml quantization type
inline ggml_type llama_xq_bits_to_type(int bits) {
    switch (bits) {
        case 8: return GGML_TYPE_Q8_0;
        case 3: // fallthrough
        case 4: return GGML_TYPE_Q4_0;
        case 2: return GGML_TYPE_Q2_K;
        default: return GGML_TYPE_Q4_0;
    }
}

inline void llama_xq_quantize_row(const float * src, void * dst, int64_t k, int bits) {
    switch (bits) {
        case 8: quantize_row_q8_0_ref(src, (block_q8_0 *) dst, k); break;
        case 3: // fallthrough
        case 4: quantize_row_q4_0_ref(src, (block_q4_0 *) dst, k); break;
        case 2: quantize_row_q2_K_ref(src, (block_q2_K *) dst, k); break;
        default: quantize_row_q4_0_ref(src, (block_q4_0 *) dst, k); break;
    }
}

inline ggml_tensor * llama_xq_quantize(ggml_context * ctx, ggml_tensor * src, int bits) {
    ggml_type t = llama_xq_bits_to_type(bits);
    ggml_tensor * dst = ggml_new_tensor(ctx, t, GGML_MAX_DIMS, src->ne);

    const int64_t nrow = ggml_nrows(src);
    const int64_t nper = src->ne[0];
    const float * x = (const float *) src->data;
    char * y = (char *) dst->data;
    const size_t row_size = ggml_row_size(t, nper);

    for (int64_t i = 0; i < nrow; ++i) {
        llama_xq_quantize_row(x + i * nper, y + i * row_size, nper, bits);
    }

    return dst;
}
