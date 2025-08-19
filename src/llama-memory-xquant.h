#pragma once

#include "llama-memory.h"
#include "ggml.h"
#include <vector>
#include <cstdint>
#include <memory>

// MVP constants (hardcoded knobs)
#ifndef LLAMA_XQ_GGML_TYPE
#define LLAMA_XQ_GGML_TYPE GGML_TYPE_Q4_0    // 4-bit, block=32; reuses tested ggml kernels
#endif

struct llama_model;

// Factory – create XQuant memory (parallel to other memory types)
llama_memory_ptr llama_memory_make_xquant(const llama_model * mdl, int32_t n_ctx);

// Append post-norm X rows during prefill (one call per layer+ubatch)
bool llama_xquant_append_prefill_rows(
    llama_memory_i * mem,
    int32_t il,
    const void * x,     // [n_tokens, n_embd] contiguous; fp16 or fp32
    int32_t n_tokens,
    int32_t n_embd,
    bool     is_fp16);

// Rematerialize pre-RoPE K,V for [t0, t1)
struct llama_xq_remat_result {
    ggml_tensor * K = nullptr;  // [T, d]
    ggml_tensor * V = nullptr;  // [T, d]
    bool ok = false;
};

llama_xq_remat_result llama_xquant_remat_kv(
    llama_memory_i * mem,
    ggml_context   * ctx,
    int32_t          il,
    int32_t          t0,
    int32_t          t1,
    ggml_tensor    * Wk,   // [d, d]
    ggml_tensor    * Wv);  // [d, d]
