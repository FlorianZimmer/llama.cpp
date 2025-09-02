#pragma once

#include "ggml.h"

// map bit width to ggml quantization type
inline ggml_type llama_xq_bits_to_type(int bits) {
    switch (bits) {
        case 8:
            return GGML_TYPE_Q8_0;
        case 3:  // fallthrough
        case 4:
            return GGML_TYPE_Q4_0;
        case 2:
            return GGML_TYPE_Q2_K;
        default:
            return GGML_TYPE_Q4_0;
    }
}

// runtime quantization helper
// NOTE: using ggml_cast ensures that quantization happens as part of the
// compute graph instead of during graph construction. The previous
// implementation invoked the low-level quantize_row_* routines immediately
// which attempted to read from `src->data` before it was populated, leading to
// segmentation faults when XQuant was enabled.
inline ggml_tensor * llama_xq_quantize(ggml_context * ctx, ggml_tensor * src, int bits) {
    ggml_type t = llama_xq_bits_to_type(bits);
    return ggml_cast(ctx, src, t);
}
