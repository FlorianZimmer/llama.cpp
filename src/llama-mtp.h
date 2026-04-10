#pragma once

#include "llama-batch.h"
#include "llama.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdint>
#include <map>
#include <vector>

struct llama_model;
struct ggml_context;
struct ggml_tensor;

struct llama_mtp_desc {
    bool     supported            = false;
    uint32_t n_predict            = 0;
    uint32_t n_draft              = 0;
    bool     dedicated_embeddings = false;
};

enum llama_mtp_seed_mode : uint8_t {
    LLAMA_MTP_SEED_MODE_NONE    = 0,
    LLAMA_MTP_SEED_MODE_HOST    = 1,
    LLAMA_MTP_SEED_MODE_BACKEND = 2,
};

struct llama_mtp_backend_seed_state {
    ggml_backend_t             backend        = nullptr;
    ggml_backend_buffer_type_t buft           = nullptr;
    uint32_t                   n_embd         = 0;
    uint64_t                   generation     = 0;
    ggml_tensor *              seed_cache_dev = nullptr;
    ggml_tensor *              seed_batch_dev = nullptr;

    ggml_context_ptr        ctx_roots;
    ggml_context_ptr        ctx_views;
    ggml_backend_buffer_ptr buf;

    std::vector<ggml_tensor *>      seed_cache_rows;
    std::vector<ggml_tensor *>      seed_batch_rows;
    std::vector<ggml_context_ptr>   capture_ctxs;

    bool ready() const;
    bool matches(ggml_backend_t backend, ggml_backend_buffer_type_t buft, uint32_t n_embd) const;
    void clear();
    void clear_capture_views();
};

struct llama_mtp_state {
    llama_mtp_desc desc;
    uint32_t       n_embd = 0;
    uint32_t       n_pos_per_embd = 0;

    std::vector<float>       seed_embd;
    std::vector<llama_token> accepted;
    std::vector<llama_token> draft;
    std::vector<float>       seed_by_seq;
    std::vector<uint32_t>    seed_epoch_by_seq;
    uint32_t                 seed_epoch = 1;
    llama_mtp_seed_mode      seed_mode = LLAMA_MTP_SEED_MODE_NONE;
    uint64_t                 backend_seed_generation_next = 1;
    llama_mtp_backend_seed_state seed_backend;

    std::vector<llama_token>    ubatch_token;
    std::vector<llama_pos>      ubatch_pos;
    std::vector<int32_t>        ubatch_n_seq_id;
    std::vector<llama_seq_id *> ubatch_seq_id;
    std::vector<llama_seq_id>   ubatch_seq_id_unq;
    std::vector<int32_t>        ubatch_seq_idx;
    std::vector<int8_t>         ubatch_output;
    std::vector<llama_seq_id>   temp_seq_ids;
    std::vector<uint8_t>        temp_seq_used;

    void clear();
    void reserve(uint32_t n_embd, uint32_t n_pos_per_embd);
    void next_seed_epoch();
    void set_seed_mode(llama_mtp_seed_mode mode);
    bool ensure_backend_seed_storage(ggml_backend_t backend, ggml_backend_buffer_type_t buft);
    void clear_backend_seed_storage();
    void clear_backend_capture_views();

    float * seed_row(llama_seq_id seq_id);
    const float * seed_row(llama_seq_id seq_id) const;
    bool has_seed(llama_seq_id seq_id) const;
    void mark_seed(llama_seq_id seq_id);
    llama_ubatch ubatch_reserve(uint32_t n_seq);

    bool enabled() const {
        return desc.supported && desc.n_draft > 0;
    }
};

llama_mtp_desc llama_mtp_init_desc(const llama_model & model);
