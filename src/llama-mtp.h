#pragma once

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

    std::vector<float>       seed_embd;
    std::vector<llama_token> accepted;
    std::vector<llama_token> draft;
    std::map<llama_seq_id, std::vector<float>> seed_by_seq;

    void clear();
    void reserve(uint32_t n_embd);

    bool enabled() const {
        return desc.supported && desc.n_draft > 0;
    }
};

llama_mtp_desc llama_mtp_init_desc(const llama_model & model);
