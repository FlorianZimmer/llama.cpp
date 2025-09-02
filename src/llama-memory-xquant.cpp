#include "llama-memory-xquant.h"

#include "llama-impl.h"
#include "llama-model.h"

#include <cstring>
#include <fstream>

struct xq_svd_header {
    char     magic[6];
    uint32_t version;
    uint32_t n_layer;
    uint32_t d_model;
};

bool llama_memory_xquant::load_svd(const std::string & path, const llama_model & model) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        return false;
    }

    xq_svd_header hdr{};
    fin.read(reinterpret_cast<char *>(&hdr), sizeof(hdr));
    if (!fin.good()) {
        return false;
    }

    if (std::memcmp(hdr.magic, "XQSV1", 6) != 0 || hdr.version != 1) {
        return false;
    }

    if (hdr.n_layer != static_cast<uint32_t>(model.hparams.n_layer)) {
        return false;
    }

    svd_layers.resize(hdr.n_layer);
    for (uint32_t i = 0; i < hdr.n_layer; ++i) {
        llama_xq_svd_layer layer{};
        fin.read(reinterpret_cast<char *>(&layer.rank_k), sizeof(layer.rank_k));
        fin.read(reinterpret_cast<char *>(&layer.rank_v), sizeof(layer.rank_v));
        if (!fin.good()) {
            return false;
        }
        svd_layers[i] = layer;
    }

    svd_loaded = true;
    return true;
}

ggml_tensor * llama_memory_xquant_context::write(ggml_context * ctx, ggml_tensor * x_cur, int32_t il) {
    if (mem.layer_data.size() <= static_cast<size_t>(il)) {
        mem.layer_data.resize(il + 1);
    }
    const int64_t d_model = mem.model.hparams.n_embd;

    // `x_cur` may arrive in a few different layouts depending on whether the
    // graph is executing a prefill or a decode step.  Normalize everything to
    // a 2-D view of shape [d_model, n_tokens] before quantization so that
    // subsequent concatenation logic can rely on a consistent representation.
    if (ggml_n_dims(x_cur) == 1) {
        // decode path: [d_model] -> [d_model, 1]
        x_cur = ggml_reshape_2d(ctx, x_cur, d_model, 1);
    } else if (x_cur->ne[0] != d_model) {
        // prefill path with tokens leading: transpose to [d_model, n_tokens]
        x_cur = ggml_transpose(ctx, x_cur);
    }
    if (!ggml_is_contiguous(x_cur)) {
        x_cur = ggml_cont(ctx, x_cur);
    }

    const int64_t n_tokens = x_cur->ne[1];

    ggml_tensor * q = llama_xq_quantize(ctx, x_cur, 4);
    LLAMA_LOG_DEBUG("xq_quantize: qtype=%d ne=(%lld,%lld,%lld,%lld) nbytes=%zu tokens=%lld\n",
                    (int) q->type,
                    (long long) q->ne[0],
                    (long long) q->ne[1],
                    (long long) q->ne[2],
                    (long long) q->ne[3],
                    ggml_nbytes(q),
                    (long long) n_tokens);
    LLAMA_LOG_DEBUG("xq write: il=%d n_tokens=%lld\n", il, (long long) n_tokens);
    pending.push_back({ il, q, n_tokens });
    return q;
}

bool llama_memory_xquant_context::apply() {
    const int64_t              d_model = mem.model.hparams.n_embd;
    std::vector<pending_write> remaining;
    remaining.reserve(pending.size());
    for (const auto & pw : pending) {
        if (pw.q->buffer == nullptr) {
            remaining.push_back(pw);
            continue;
        }

        if (mem.layer_data.size() <= (size_t) pw.il) {
            mem.layer_data.resize(pw.il + 1);
        }

        llama_memory_xquant::xq_block blk{};
        blk.type = pw.q->type;
        blk.ne0  = d_model;
        blk.ne1  = pw.n_tokens;  // authoritative token count

        const size_t bytes = ggml_nbytes(pw.q);
        blk.data.resize(bytes);
        ggml_backend_tensor_get(pw.q, blk.data.data(), 0, bytes);

        const size_t row_b      = ggml_row_size(blk.type, d_model);
        const size_t expected_b = row_b * (size_t) pw.n_tokens;

        if (row_b == 0 || bytes % row_b != 0) {
            LLAMA_LOG_ERROR(
                "xq apply: qtype=%d d_model=%lld bytes=%zu row_b=%zu tokens(write)=%lld -- incompatible backend output\n",
                (int) blk.type,
                (long long) d_model,
                bytes,
                row_b,
                (long long) pw.n_tokens);
            continue;
        }

        const size_t tokens_bytes = bytes / row_b;
        if (bytes != expected_b) {
            LLAMA_LOG_WARN(
                "xq apply: backend returned %zu bytes (%zu tokens) but expected %zu bytes for %lld tokens\n",
                bytes,
                tokens_bytes,
                expected_b,
                (long long) pw.n_tokens);
            blk.ne1 = tokens_bytes;
        }

        mem.layer_data[pw.il].push_back(std::move(blk));
    }
    pending = std::move(remaining);
    return true;
}

static uint32_t count_tokens_for_layer(const llama_memory_xquant &                                     mem,
                                       const std::vector<llama_memory_xquant_context::pending_write> & pending,
                                       int32_t                                                         il) {
    uint32_t n = 0;
    if (mem.layer_data.size() > (size_t) il) {
        for (const auto & blk : mem.layer_data[il]) {
            n += (uint32_t) blk.ne1;
        }
    }
    for (const auto & pw : pending) {
        if (pw.il == il) {
            n += (uint32_t) pw.n_tokens;
        }
    }
    return n;
}

uint32_t llama_memory_xquant_context::get_n_kv() const {
    return count_tokens_for_layer(mem, pending, 0);
}

// helper: dequantize and concatenate cached X for layer il
static ggml_tensor * normalize_to_dm_by_elements(ggml_context * ctx, ggml_tensor * t, int64_t d_model) {
    int64_t elems = ggml_nelements(t);
    GGML_ASSERT(elems % d_model == 0);
    int64_t cols = elems / d_model;
    if (t->ne[0] != d_model || t->ne[1] != cols) {
        t = ggml_reshape_2d(ctx, t, d_model, cols);
    }
    return t;
}

static ggml_tensor * xq_build_full_x(
    ggml_context * ctx,
    const llama_memory_xquant & mem,
    const std::vector<llama_memory_xquant_context::pending_write> & pending,
    int32_t il,
    int64_t d_model) {

    ggml_tensor * cur = nullptr;

    // A) Stored past blocks
    if (mem.layer_data.size() > (size_t) il) {
        for (const auto & blk : mem.layer_data[il]) {
            ggml_tensor * qt = ggml_new_tensor_2d(ctx, blk.type, d_model, blk.ne1);
            memcpy(qt->data, blk.data.data(), blk.data.size());
            ggml_tensor * deq = ggml_cast(ctx, qt, GGML_TYPE_F32);
            deq = normalize_to_dm_by_elements(ctx, deq, d_model);
            if (!ggml_is_contiguous(deq)) {
                deq = ggml_cont(ctx, deq);
            }
            cur = cur ? ggml_concat(ctx, cur, deq, 1) : deq;
            cur = normalize_to_dm_by_elements(ctx, cur, d_model);
        }
    }

    // B) Pending writes
    for (const auto & pw : pending) {
        if (pw.il != il) continue;

        ggml_tensor * deq_full = ggml_cast(ctx, pw.q, GGML_TYPE_F32);
        deq_full = normalize_to_dm_by_elements(ctx, deq_full, d_model);
        ggml_tensor * deq_cont = ggml_cont(ctx, deq_full);

        const int64_t cols_full = ggml_nelements(deq_cont) / d_model;
        const int64_t cols_take = pw.n_tokens <= cols_full ? pw.n_tokens : cols_full;

        ggml_tensor * deq = ggml_view_2d(ctx,
                                        deq_cont,
                                        /*ne0*/ d_model,
                                        /*ne1*/ cols_take,
                                        /*nb1*/ deq_cont->nb[1],
                                        /*offs*/ 0);
        deq = normalize_to_dm_by_elements(ctx, deq, d_model);

        cur = cur ? ggml_concat(ctx, cur, deq, 1) : deq;
        cur = normalize_to_dm_by_elements(ctx, cur, d_model);
    }

    if (cur) {
        cur = normalize_to_dm_by_elements(ctx, cur, d_model);
    }

    return cur;
}

ggml_tensor * llama_memory_xquant_context::get_k(ggml_context * ctx, int32_t il) {
    const int64_t d_model = mem.model.hparams.n_embd;

    ggml_tensor * x = xq_build_full_x(ctx, mem, pending, il, d_model);
    if (!x) {
        return nullptr;
    }

    x = normalize_to_dm_by_elements(ctx, x, d_model);
    const int64_t n_tok      = ggml_nelements(x) / d_model;
    const uint32_t n_tok_cnt = count_tokens_for_layer(mem, pending, il);
    LLAMA_LOG_DEBUG("xq layer %d: n_tok(built)=%lld n_tok(counted)=%u\n", il, (long long) n_tok, n_tok_cnt);
    GGML_ASSERT((uint64_t) n_tok == (uint64_t) n_tok_cnt);

    const auto & hp   = mem.model.hparams;
    const int64_t out_k = hp.n_embd_head_k * hp.n_head_kv(il);

    ggml_tensor * k_lin = ggml_mul_mat(ctx, mem.model.layers[il].wk, x);
    GGML_ASSERT(ggml_nelements(k_lin) == out_k * n_tok);

    ggml_tensor * k = ggml_reshape_3d(ctx, k_lin, hp.n_embd_head_k, hp.n_head_kv(il), n_tok);
    return k;
}

ggml_tensor * llama_memory_xquant_context::get_v(ggml_context * ctx, int32_t il) {
    const int64_t d_model = mem.model.hparams.n_embd;

    ggml_tensor * x = xq_build_full_x(ctx, mem, pending, il, d_model);
    if (!x) {
        return nullptr;
    }

    x = normalize_to_dm_by_elements(ctx, x, d_model);
    const int64_t n_tok      = ggml_nelements(x) / d_model;
    const uint32_t n_tok_cnt = count_tokens_for_layer(mem, pending, il);
    LLAMA_LOG_DEBUG("xq layer %d: n_tok(built)=%lld n_tok(counted)=%u\n", il, (long long) n_tok, n_tok_cnt);
    GGML_ASSERT((uint64_t) n_tok == (uint64_t) n_tok_cnt);

    const auto & hp   = mem.model.hparams;
    const int64_t out_v = hp.n_embd_head_v * hp.n_head_kv(il);

    ggml_tensor * v_lin = ggml_mul_mat(ctx, mem.model.layers[il].wv, x);
    GGML_ASSERT(ggml_nelements(v_lin) == out_v * n_tok);

    ggml_tensor * v = ggml_reshape_3d(ctx, v_lin, hp.n_embd_head_v, hp.n_head_kv(il), n_tok);
    return v;
}

llama_memory_context_ptr llama_memory_xquant::init_batch(llama_batch_allocr &, uint32_t, bool) {
    return std::make_unique<llama_memory_xquant_context>(*this);
}

llama_memory_context_ptr llama_memory_xquant::init_full() {
    return std::make_unique<llama_memory_xquant_context>(*this);
}

llama_memory_context_ptr llama_memory_xquant::init_update(llama_context *, bool) {
    return std::make_unique<llama_memory_xquant_context>(*this);
}
