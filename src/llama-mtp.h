#pragma once

#include "llama-batch.h"
#include "llama.h"

#include <cstdint>
#include <map>
#include <vector>

struct llama_model;

struct llama_mtp_desc {
    bool     supported            = false;
    uint32_t n_predict            = 0;
    uint32_t n_draft              = 0;
    bool     dedicated_embeddings = false;
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
