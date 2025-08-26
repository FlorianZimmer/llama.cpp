#include "llama-memory-xquant.h"

#include "llama-batch.h"

llama_memory_xquant_ctx::llama_memory_xquant_ctx(llama_memory_status status, llama_memory_xquant *)
    : status_(status) {
}

bool llama_memory_xquant_ctx::next() {
    return false;
}

bool llama_memory_xquant_ctx::apply() {
    return false;
}

const llama_ubatch & llama_memory_xquant_ctx::get_ubatch() const {
    static llama_ubatch dummy;
    return dummy;
}

llama_memory_status llama_memory_xquant_ctx::get_status() const {
    return status_;
}

llama_memory_xquant::llama_memory_xquant(const llama_xq_params & params)
    : p_(params) {
}

llama_memory_context_ptr llama_memory_xquant::init_batch(llama_batch_allocr &, uint32_t, bool) {
    return llama_memory_context_ptr(new llama_memory_xquant_ctx(LLAMA_MEMORY_STATUS_NO_UPDATE, this));
}

llama_memory_context_ptr llama_memory_xquant::init_full() {
    return llama_memory_context_ptr(new llama_memory_xquant_ctx(LLAMA_MEMORY_STATUS_NO_UPDATE, this));
}

llama_memory_context_ptr llama_memory_xquant::init_update(llama_context *, bool) {
    return llama_memory_context_ptr(new llama_memory_xquant_ctx(LLAMA_MEMORY_STATUS_NO_UPDATE, this));
}

bool llama_memory_xquant::get_can_shift() const {
    return false;
}

void llama_memory_xquant::clear(bool) {
}

bool llama_memory_xquant::seq_rm(llama_seq_id, llama_pos, llama_pos) {
    return false;
}

void llama_memory_xquant::seq_cp(llama_seq_id, llama_seq_id, llama_pos, llama_pos) {
}

void llama_memory_xquant::seq_keep(llama_seq_id) {
}

void llama_memory_xquant::seq_add(llama_seq_id, llama_pos, llama_pos, llama_pos) {
}

void llama_memory_xquant::seq_div(llama_seq_id, llama_pos, llama_pos, int) {
}

llama_pos llama_memory_xquant::seq_pos_min(llama_seq_id) const {
    return 0;
}

llama_pos llama_memory_xquant::seq_pos_max(llama_seq_id) const {
    return 0;
}

void llama_memory_xquant::state_write(llama_io_write_i &, llama_seq_id, llama_state_seq_flags) const {
}

void llama_memory_xquant::state_read(llama_io_read_i &, llama_seq_id, llama_state_seq_flags) {
}

