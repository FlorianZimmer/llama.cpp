#pragma once

#include "ggml.h"
#include "llama-batch.h"
#include "llama-memory.h"
#include "llama-xq-quant.h"

#include <cstdint>
#include <string>
#include <vector>

struct llama_model;
struct llama_context;

struct llama_xq_svd_layer {
    uint32_t rank_k;
    uint32_t rank_v;
};

class llama_memory_xquant : public llama_memory_i {
  public:
    llama_memory_xquant(const llama_model & model) : model(model) {}

    ~llama_memory_xquant() override = default;

    struct xq_block {
        ggml_type            type;
        int64_t              ne0;
        int64_t              ne1;  // tokens (logical)
        std::vector<uint8_t> data;
    };

    std::vector<std::vector<xq_block>> layer_data;

    bool load_svd(const std::string & path, const llama_model & model);

    llama_memory_context_ptr init_batch(llama_batch_allocr &, uint32_t, bool) override;
    llama_memory_context_ptr init_full() override;
    llama_memory_context_ptr init_update(llama_context *, bool) override;

    bool get_can_shift() const override { return false; }

    void clear(bool) override {}

    bool seq_rm(llama_seq_id, llama_pos, llama_pos) override { return true; }

    void seq_cp(llama_seq_id, llama_seq_id, llama_pos, llama_pos) override {}

    void seq_keep(llama_seq_id) override {}

    void seq_add(llama_seq_id, llama_pos, llama_pos, llama_pos) override {}

    void seq_div(llama_seq_id, llama_pos, llama_pos, int) override {}

    llama_pos seq_pos_min(llama_seq_id) const override { return 0; }

    llama_pos seq_pos_max(llama_seq_id) const override { return 0; }

    void state_write(llama_io_write_i &, llama_seq_id, llama_state_seq_flags) const override {}

    void state_read(llama_io_read_i &, llama_seq_id, llama_state_seq_flags) override {}

  private:
    friend class llama_memory_xquant_context;

    const llama_model & model;

    bool                            svd_loaded = false;
    std::vector<llama_xq_svd_layer> svd_layers;
};

class llama_memory_xquant_cl : public llama_memory_xquant {
  public:
    llama_memory_xquant_cl(const llama_model & model) : llama_memory_xquant(model) {}
};

class llama_memory_xquant_context : public llama_memory_context_i {
  public:
    struct pending_write {
        int32_t       il;
        ggml_tensor * q;
        int64_t       n_tokens;  // logical token count for this write
    };

    llama_memory_xquant_context(llama_memory_xquant & mem) : mem(mem) {}

    ~llama_memory_xquant_context() override = default;

    bool next() override { return processed ? false : (processed = true); }

    bool apply() override;

    const llama_ubatch & get_ubatch() const override { return dummy; }

    llama_memory_status get_status() const override { return LLAMA_MEMORY_STATUS_SUCCESS; }

    ggml_tensor * write(ggml_context * ctx, ggml_tensor * x_cur, int32_t il, int32_t bits);

    uint32_t      get_n_kv() const;
    ggml_tensor * get_k(ggml_context * ctx, int32_t il);
    ggml_tensor * get_v(ggml_context * ctx, int32_t il);

  private:
    llama_memory_xquant &      mem;
    std::vector<pending_write> pending;
    mutable llama_ubatch       dummy{};
    bool                       processed = false;
};
