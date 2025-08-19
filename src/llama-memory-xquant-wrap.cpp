#include "llama-memory-xquant-wrap.h"
#include "llama-memory-xquant.h"   // your existing store
#include "llama-batch.h"
#include "llama.h"
#include "ggml.h"
#define LLAMA_API_INTERNAL
#include "llama-impl.h"
#include "llama.h"
#include "ggml.h"
#include <mutex>
#include <memory>
#include <utility>
#include <mutex>

static std::once_flag g_xq_once;

namespace {
class llama_memory_xquant_wrap final : public llama_memory_i {
public:
    llama_memory_xquant_wrap(const llama_model * mdl, llama_memory_ptr base_kv, int32_t n_ctx)
    : mdl_(mdl)
    , base_(std::move(base_kv))
    , store_(llama_memory_make_xquant(mdl, n_ctx)) {
        std::call_once(g_xq_once, []{
            LLAMA_LOG_INFO("[xquant] wrapper active (capturing post-norm X, rematerializing K/V)\n");
        });
    }

    // delegate everything to base memory (no behavior change)
    llama_memory_context_ptr init_batch(llama_batch_allocr & a, uint32_t n_ubatch, bool embd_all) override {
        return base_->init_batch(a, n_ubatch, embd_all);
    }
    llama_memory_context_ptr init_full() override {
        return base_->init_full();
    }
    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override {
        return base_->init_update(lctx, optimize);
    }

    bool get_can_shift() const override { return base_->get_can_shift(); }
    void clear(bool data) override { base_->clear(data); }

    bool  seq_rm  (llama_seq_id s, llama_pos p0, llama_pos p1) override { return base_->seq_rm  (s,p0,p1); }
    void  seq_cp  (llama_seq_id s, llama_seq_id d, llama_pos p0, llama_pos p1) override { base_->seq_cp  (s,d,p0,p1); }
    void  seq_keep(llama_seq_id s) override { base_->seq_keep(s); }
    void  seq_add (llama_seq_id s, llama_pos p0, llama_pos p1, llama_pos shift) override { base_->seq_add (s,p0,p1,shift); }
    void  seq_div (llama_seq_id s, llama_pos p0, llama_pos p1, int d) override { base_->seq_div (s,p0,p1,d); }

    llama_pos seq_pos_min(llama_seq_id s) const override { return base_->seq_pos_min(s); }
    llama_pos seq_pos_max(llama_seq_id s) const override { return base_->seq_pos_max(s); }

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override { base_->state_write(io, seq_id, flags); }
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override { base_->state_read (io, seq_id, flags); }

    // accessors for helpers
    llama_memory_i * base()  const { return base_.get(); }
    llama_memory_i * store() const { return store_.get(); }

private:
    const llama_model *       mdl_;
    llama_memory_ptr          base_;
    llama_memory_ptr          store_; // your existing xquant store
};

static llama_memory_xquant_wrap * as_wrap(llama_memory_i * m) {
    return dynamic_cast<llama_memory_xquant_wrap *>(m);
}
static const llama_memory_xquant_wrap * as_wrap(const llama_memory_i * m) {
    return dynamic_cast<const llama_memory_xquant_wrap *>(m);
}

} // anon

llama_memory_ptr llama_memory_make_xquant_wrap(
    const llama_model * mdl,
    llama_memory_ptr    base_kv,
    int32_t             n_ctx_tokens) {
    return llama_memory_ptr(new llama_memory_xquant_wrap(mdl, std::move(base_kv), n_ctx_tokens));
}

bool llama_memory_is_xquant_enabled(const llama_memory_i * mem) {
    return as_wrap(mem) != nullptr;
}

// The helpers simply forward to the inner store you already implemented:

bool llama_xquant_wrap_append_prefill_rows(
    llama_memory_i * mem,
    int32_t il,
    const void * x,
    int32_t n_tokens,
    int32_t n_embd,
    bool     is_fp16) {

    if (auto * w = as_wrap(mem)) {
        return llama_xquant_append_prefill_rows(w->store(), il, x, n_tokens, n_embd, is_fp16);
    }
    // also allow direct store usage (dev/testing)
    return llama_xquant_append_prefill_rows(mem, il, x, n_tokens, n_embd, is_fp16);
}

llama_xq_remat_result llama_xquant_wrap_remat_kv(
    llama_memory_i * mem,
    ggml_context   * ctx,
    int32_t          il,
    int32_t          t0,
    int32_t          t1,
    ggml_tensor    * Wk,
    ggml_tensor    * Wv) {

    if (auto * w = as_wrap(mem)) {
        return llama_xquant_remat_kv(w->store(), ctx, il, t0, t1, Wk, Wv);
    }
    return llama_xquant_remat_kv(mem, ctx, il, t0, t1, Wk, Wv);
}
