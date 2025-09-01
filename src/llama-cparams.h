#pragma once

#include "llama.h"

#include <cstdint>
#include <string>

#define LLAMA_MAX_SEQ 64

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool no_perf;
    bool warmup;
    bool op_offload;
    bool kv_unified;

    bool xquant;
    bool xquant_cl;
    int32_t xq_bits;
    int32_t xq_group;
    int32_t xq_base_layers;
    bool xq_gqa_svd;
    int32_t xq_svd_rank;
    std::string xq_svd_path;

    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
};
