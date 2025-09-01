#pragma once

#include "llama-memory.h"

struct llama_model;
struct llama_context;

class llama_memory_xquant : public llama_memory_i {
public:
    llama_memory_xquant(const llama_model & model) { (void) model; }
    ~llama_memory_xquant() override = default;

    llama_memory_context_ptr init_batch(llama_batch_allocr &, uint32_t, bool) override { return nullptr; }
    llama_memory_context_ptr init_full() override { return nullptr; }
    llama_memory_context_ptr init_update(llama_context *, bool) override { return nullptr; }

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
};

class llama_memory_xquant_cl : public llama_memory_xquant {
public:
    llama_memory_xquant_cl(const llama_model & model) : llama_memory_xquant(model) {}
};

