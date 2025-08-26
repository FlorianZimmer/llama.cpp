#include "llama-memory-xquant.h"

#include "llama-batch.h"

llama_memory_xquant_ctx::llama_memory_xquant_ctx(std::unique_ptr<llama_memory_context_i> base)
    : base_(std::move(base)) {
}

bool llama_memory_xquant_ctx::next() {
    return base_ ? base_->next() : false;
}

bool llama_memory_xquant_ctx::apply() {
    return base_ ? base_->apply() : false;
}

const llama_ubatch & llama_memory_xquant_ctx::get_ubatch() const {
    if (base_) {
        return base_->get_ubatch();
    }
    static llama_ubatch dummy;
    return dummy;
}

llama_memory_status llama_memory_xquant_ctx::get_status() const {
    return base_ ? base_->get_status() : LLAMA_MEMORY_STATUS_NO_UPDATE;
}

llama_memory_xquant::llama_memory_xquant(const llama_xq_params & params,
                                         std::unique_ptr<llama_memory_i> base)
    : p_(params), base_(std::move(base)) {
}

llama_memory_context_ptr llama_memory_xquant::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    auto ctx = base_->init_batch(balloc, n_ubatch, embd_all);
    return std::make_unique<llama_memory_xquant_ctx>(std::move(ctx));
}

llama_memory_context_ptr llama_memory_xquant::init_full() {
    auto ctx = base_->init_full();
    return std::make_unique<llama_memory_xquant_ctx>(std::move(ctx));
}

llama_memory_context_ptr llama_memory_xquant::init_update(llama_context * lctx, bool optimize) {
    auto ctx = base_->init_update(lctx, optimize);
    return std::make_unique<llama_memory_xquant_ctx>(std::move(ctx));
}

bool llama_memory_xquant::get_can_shift() const {
    return base_->get_can_shift();
}

void llama_memory_xquant::clear(bool data) {
    base_->clear(data);
}

bool llama_memory_xquant::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    return base_->seq_rm(seq_id, p0, p1);
}

void llama_memory_xquant::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    base_->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_xquant::seq_keep(llama_seq_id seq_id) {
    base_->seq_keep(seq_id);
}

void llama_memory_xquant::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    base_->seq_add(seq_id, p0, p1, shift);
}

void llama_memory_xquant::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    base_->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_xquant::seq_pos_min(llama_seq_id seq_id) const {
    return base_->seq_pos_min(seq_id);
}

llama_pos llama_memory_xquant::seq_pos_max(llama_seq_id seq_id) const {
    return base_->seq_pos_max(seq_id);
}

void llama_memory_xquant::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    base_->state_write(io, seq_id, flags);
}

void llama_memory_xquant::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    base_->state_read(io, seq_id, flags);
}

