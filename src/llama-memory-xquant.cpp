#include "llama-memory-xquant.h"
#include "llama.h"
#include "llama-batch.h"
#include "ggml.h"

#include <cassert>
#include <cstring>
#include <vector>
#include <cstdio>
#include <climits>

namespace {

struct xq_layer_buf {
    // model dims
    int32_t n_embd         = 0;      // columns per row (d)
    // storage
    size_t  row_size_bytes = 0;      // bytes per quantized row (ggml_row_size)
    int32_t n_written      = 0;      // rows written (tokens)
    std::vector<uint8_t> qrows;      // n_ctx * row_size_bytes opaque bytes
};

class llama_memory_xquant final : public llama_memory_i {
public:
    llama_memory_xquant(const llama_model * mdl, int32_t n_ctx)
    : mdl_(mdl)
    , n_ctx_(n_ctx) {
        // Fetch as 64-bit, then narrow safely.
        const int64_t nl64 = llama_model_n_layer(mdl_);
        const int64_t d64  = llama_model_n_embd(mdl_);
        GGML_ASSERT(nl64 >= 0 && nl64 <= INT32_MAX);
        GGML_ASSERT(d64  >  0 && d64  <= INT32_MAX);
        const int32_t nl = (int32_t) nl64;
        const int32_t d  = (int32_t) d64;

        const char * tname = ggml_type_name(LLAMA_XQ_GGML_TYPE);
        // cast to (int) for MSVC's printf family to avoid width warnings
        std::fprintf(stderr, "[xquant] new memory (%s) d=%d n_ctx=%d layers=%d\n",
            tname ? tname : "Q4_0", (int)d, (int)n_ctx_, (int)nl);

        // sanity: embedding must align to quant block size
        const int64_t blck64 = ggml_blck_size(LLAMA_XQ_GGML_TYPE);
        GGML_ASSERT(blck64 > 0 && blck64 <= INT32_MAX);
        const int32_t blck = (int32_t) blck64;

        GGML_ASSERT(blck > 0 && d % blck == 0 && "n_embd must be multiple of quant block size");

        const size_t row_size = ggml_row_size(LLAMA_XQ_GGML_TYPE, d);

        layers_.resize((size_t)nl);
        for (auto & L : layers_) {
            L.n_embd         = d;
            L.row_size_bytes = row_size;
            L.qrows.resize((size_t)n_ctx_ * row_size);
            L.n_written      = 0;
        }
    }

    // ---- memory_i interface ----
    llama_memory_context_ptr init_batch(llama_batch_allocr &, uint32_t, bool) override {
        return make_noop_ctx();
    }

    llama_memory_context_ptr init_full() override {
        return make_noop_ctx();
    }

    llama_memory_context_ptr init_update(llama_context *, bool) override {
        return make_noop_ctx(LLAMA_MEMORY_STATUS_NO_UPDATE);
    }

    bool get_can_shift() const override { return false; }

    void clear(bool data) override {
        for (auto & L : layers_) {
            if (data) std::fill(L.qrows.begin(), L.qrows.end(), uint8_t{0});
            L.n_written = 0;
        }
        pos_min_ = 0; pos_max_ = 0;
    }

    bool seq_rm  (llama_seq_id, llama_pos, llama_pos) override { return false; }
    void seq_cp  (llama_seq_id, llama_seq_id, llama_pos, llama_pos) override {}
    void seq_keep(llama_seq_id) override {}
    void seq_add (llama_seq_id, llama_pos, llama_pos, llama_pos) override {}
    void seq_div (llama_seq_id, llama_pos, llama_pos, int) override {}

    llama_pos seq_pos_min(llama_seq_id) const override { return pos_min_; }
    llama_pos seq_pos_max(llama_seq_id) const override { return pos_max_; }

    void state_write(llama_io_write_i &, llama_seq_id, llama_state_seq_flags) const override {}
    void state_read (llama_io_read_i  &, llama_seq_id, llama_state_seq_flags) override {}

    // ---- MVP helpers ----

    bool append_rows(int32_t il, const void * x, int32_t n_tokens, int32_t n_embd, bool is_fp16) {
        if (il < 0 || il >= (int)layers_.size()) return false;
        auto & L = layers_[il];

        if (n_embd != L.n_embd) return false;
        if (L.n_written + n_tokens > n_ctx_) return false;

        std::vector<float> row_fp32((size_t)L.n_embd);

        // public quantization hooks
        const auto * tt = ggml_get_type_traits(LLAMA_XQ_GGML_TYPE);
        if (!tt || !tt->from_float_ref) return false;
        auto q_from = tt->from_float_ref; // f32 -> quant

        const size_t src_stride = (size_t)L.n_embd * (is_fp16 ? sizeof(ggml_fp16_t) : sizeof(float));

        for (int32_t t = 0; t < n_tokens; ++t) {
            const uint8_t * src = (const uint8_t *)x + (size_t)t * src_stride;

            if (is_fp16) {
                const ggml_fp16_t * s = (const ggml_fp16_t *)src;
                for (int i = 0; i < L.n_embd; ++i) row_fp32[i] = ggml_fp16_to_fp32(s[i]);
            } else {
                std::memcpy(row_fp32.data(), src, (size_t)L.n_embd * sizeof(float));
            }

            void * dst_row = (void *)(L.qrows.data() +
                              (size_t)L.n_written * L.row_size_bytes);

            GGML_ASSERT(L.n_embd <= INT_MAX);
            q_from(/*src f32*/ row_fp32.data(),
                   /*dst q  */ dst_row,
                   /*k     */ (int)L.n_embd);

            ++L.n_written;
        }

        pos_max_ = L.n_written; // single-seq assumption
        return true;
    }

    ggml_tensor * dequant_window_fp16(ggml_context * ctx, int32_t il, int32_t t0, int32_t t1, int32_t n_embd) {
        if (il < 0 || il >= (int)layers_.size()) return nullptr;
        auto & L = layers_[il];
        if (n_embd != L.n_embd) return nullptr;
        if (t0 < 0 || t1 > L.n_written || t0 >= t1) return nullptr;

        const int32_t T = t1 - t0;

        // Xt: [n_embd, T]  (ne0=n_embd, ne1=T) — ggml takes int64 dims
        ggml_tensor * Xt = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, (int64_t)L.n_embd, (int64_t)T);

        std::vector<float> row_fp32((size_t)L.n_embd);

        const auto * tt = ggml_get_type_traits(LLAMA_XQ_GGML_TYPE);
        if (!tt || !tt->to_float) return nullptr;
        auto q_to = tt->to_float; // quant -> f32

        for (int32_t t = 0; t < T; ++t) {
            const void * src_row = (const void *)(L.qrows.data() +
                                   (size_t)(t0 + t) * L.row_size_bytes);

            GGML_ASSERT(L.n_embd <= INT_MAX);
            q_to(/*src q */ src_row,
                 /*dst f */ row_fp32.data(),
                 /*k     */ (int)L.n_embd);

            ggml_fp32_to_fp16_row(row_fp32.data(),
                (ggml_fp16_t *)((char*)Xt->data + (size_t)t * Xt->nb[1]),
                (int)L.n_embd);
        }

        return Xt;
    }

    int32_t n_layer() const { return (int32_t)layers_.size(); }
    int32_t n_ctx()   const { return n_ctx_; }

private:
    class ctx_noop final : public llama_memory_context_i {
    public:
        explicit ctx_noop(llama_memory_status s = LLAMA_MEMORY_STATUS_NO_UPDATE) : status_(s) {}
        bool next() override { return false; }
        bool apply() override { return true; }
        const llama_ubatch & get_ubatch() const override {
            static llama_ubatch dummy{};
            return dummy;
        }
        llama_memory_status get_status() const override { return status_; }
    private:
        llama_memory_status status_;
    };

    llama_memory_context_ptr make_noop_ctx(llama_memory_status s = LLAMA_MEMORY_STATUS_SUCCESS) {
        return llama_memory_context_ptr(new ctx_noop(s));
    }

private:
    const llama_model * mdl_;
    int32_t n_ctx_;
    std::vector<xq_layer_buf> layers_;
    llama_pos pos_min_ = 0, pos_max_ = 0;
};

static llama_memory_xquant * as_xq(llama_memory_i * mem) {
    return dynamic_cast<llama_memory_xquant *>(mem);
}

} // anon

llama_memory_ptr llama_memory_make_xquant(const llama_model * mdl, int32_t n_ctx) {
    return llama_memory_ptr(new llama_memory_xquant(mdl, n_ctx));
}

bool llama_xquant_append_prefill_rows(
    llama_memory_i * mem,
    int32_t il,
    const void * x,
    int32_t n_tokens,
    int32_t n_embd,
    bool     is_fp16) {

    auto * m = as_xq(mem);
    if (!m) return false;
    return m->append_rows(il, x, n_tokens, n_embd, is_fp16);
}

llama_xq_remat_result llama_xquant_remat_kv(
    llama_memory_i * mem, ggml_context * ctx,
    int32_t il, int32_t t0, int32_t t1,
    ggml_tensor * Wk, ggml_tensor * Wv) {

    llama_xq_remat_result R{};
    auto * m = as_xq(mem);
    if (!m) return R;

    const int32_t d = (int32_t) Wk->ne[0];
    const int32_t T = t1 - t0;

    ggml_tensor * Xt = m->dequant_window_fp16(ctx, il, t0, t1, d); // [d, T] in our construction
    if (!Xt) return R;

    ggml_tensor * Ktmp = ggml_mul_mat(ctx, Wk, Xt);
    ggml_tensor * Vtmp = ggml_mul_mat(ctx, Wv, Xt);

    auto to_Td = [&](ggml_tensor *M) -> ggml_tensor * {
        // Some backends can yield [d,T] or [T,d]. Normalize to [T,d].
        if (M->ne[0] == d && M->ne[1] == T) {
            return ggml_transpose(ctx, M); // [d,T] -> [T,d]
        } else if (M->ne[0] == T && M->ne[1] == d) {
            return M; // already [T,d]
        } else {
            // Fallback: transpose; tests will catch if this isn’t right
            return ggml_transpose(ctx, M);
        }
    };

    ggml_tensor * K = to_Td(Ktmp);
    ggml_tensor * V = to_Td(Vtmp);

    R.K = K; R.V = V; R.ok = true;
    return R;
}
