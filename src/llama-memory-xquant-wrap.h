#pragma once
#include "llama-memory.h"
#include "llama-memory-xquant.h"  // <- brings the ONE true llama_xq_remat_result and base helpers
// Process-wide flag: true once the XQuant wrapper is attached in this process.
// Implemented in llama-memory-xquant-wrap.cpp
bool llama_xquant_runtime_active();

// forward decls
struct llama_model;

// Factory
llama_memory_ptr llama_memory_make_xquant_wrap(
    const llama_model * mdl,
    llama_memory_ptr    base_kv,       // takes ownership
    int32_t             n_ctx_tokens);

// Query helper (used by layer code to detect XQuant)
bool llama_memory_is_xquant_enabled(const llama_memory_i * mem);

// Wrapper-friendly helper names (distinct from base helpers)
bool llama_xquant_wrap_append_prefill_rows(
    llama_memory_i * mem, int32_t il,
    const void * x, int32_t n_tokens, int32_t n_embd, bool is_fp16);

llama_xq_remat_result llama_xquant_wrap_remat_kv(
    llama_memory_i * mem, ggml_context * ctx, int32_t il,
    int32_t t0, int32_t t1, ggml_tensor * Wk, ggml_tensor * Wv);