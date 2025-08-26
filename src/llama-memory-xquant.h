#pragma once

#include "llama-memory.h"

#include <memory>

struct llama_xq_params {
    int  bits        = 4;
    int  group_size  = 128;
    int  base_layers = 3;
    bool use_cl      = false;
    bool gqa_svd     = false;
};

class llama_memory_xquant;

class llama_memory_xquant_ctx : public llama_memory_context_i {
public:
    explicit llama_memory_xquant_ctx(std::unique_ptr<llama_memory_context_i> base);
    ~llama_memory_xquant_ctx() override = default;

    bool next() override;
    bool apply() override;

    const llama_ubatch & get_ubatch() const override;
    llama_memory_status  get_status() const override;

private:
    std::unique_ptr<llama_memory_context_i> base_;
};

class llama_memory_xquant : public llama_memory_i {
public:
    llama_memory_xquant(const llama_xq_params & params, std::unique_ptr<llama_memory_i> base);
    ~llama_memory_xquant() override = default;

    llama_memory_context_ptr init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) override;
    llama_memory_context_ptr init_full() override;
    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags) override;

private:
    llama_xq_params p_;
    std::unique_ptr<llama_memory_i> base_;
};

