#include "llama-memory-xquant.h"

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
        x_cur = ggml_cont(ctx, ggml_transpose(ctx, x_cur));
    }

    ggml_tensor * q = llama_xq_quantize(ctx, x_cur, 4);
    pending.push_back({ il, q });
    return q;
}


bool llama_memory_xquant_context::apply() {
    for (const auto & pw : pending) {
        if (mem.layer_data.size() <= (size_t) pw.il) {
            mem.layer_data.resize(pw.il + 1);
        }

        llama_memory_xquant::xq_block blk{};
        blk.type = pw.q->type;
        // the cached layout is always [d_model, n_tokens]
        blk.ne0  = mem.model.hparams.n_embd;
        blk.ne1  = pw.q->ne[1];
        blk.data.resize(ggml_nbytes(pw.q));
        ggml_backend_tensor_get(pw.q, blk.data.data(), 0, blk.data.size());
        mem.layer_data[pw.il].push_back(std::move(blk));
    }
    pending.clear();
    return true;
}

uint32_t llama_memory_xquant_context::get_n_kv() const {
    if (mem.layer_data.empty()) {
        return 0;
    }

    uint32_t n = 0;
    for (const auto & blk : mem.layer_data[0]) {
        n += blk.ne1;
    }
    for (const auto & pw : pending) {
        if (pw.il == 0) {
            n += pw.q->ne[1];
        }
    }
    return n;
}

// helper: dequantize and concatenate cached X for layer il
static ggml_tensor * xq_dequant_concat(ggml_context * ctx,
        const std::vector<llama_memory_xquant::xq_block> & qs,
        const std::vector<llama_memory_xquant_context::pending_write> & pending,
        int32_t il, int64_t d_model) {
    ggml_tensor * cur = nullptr;

    // 1) dequantize pre-existing blocks
    for (const auto & blk : qs) {
        ggml_tensor * qt  = ggml_new_tensor_2d(ctx, blk.type, d_model, blk.ne1);
        // copy raw bytes into the tensor buffer
        memcpy(qt->data, blk.data.data(), blk.data.size());

        ggml_tensor * deq = ggml_cast(ctx, qt, GGML_TYPE_F32);
        // ggml_cast may promote tensors to higher dimensions depending on the
        // backend.  Force a clean 2-D view with a fixed leading dimension so
        // that all subsequent concatenations are well-defined.
        deq = ggml_reshape_2d(ctx, deq, d_model, ggml_nelements(deq) / d_model);

        if (!cur) {
            cur = deq;
        } else {
            cur = ggml_concat(ctx, cur, deq, 1);
            // ggml_concat may promote to >2-D; fold it back explicitly
            int64_t cur_ne0 = cur->ne[0];
            int64_t cur_ne1 = ggml_nelements(cur) / cur_ne0;
            cur = ggml_reshape_2d(ctx, cur, cur_ne0, cur_ne1);
        }
    }

    // 2) append any pending writes for this layer
    for (const auto & pw : pending) {
        if (pw.il != il) {
            continue;
        }
        ggml_tensor * deq = ggml_cast(ctx, pw.q, GGML_TYPE_F32);
        deq = ggml_reshape_2d(ctx, deq, d_model, ggml_nelements(deq) / d_model);

        if (!cur) {
            cur = deq;
        } else {
            cur = ggml_concat(ctx, cur, deq, 1);
            cur = ggml_reshape_2d(ctx, cur, d_model, ggml_nelements(cur) / d_model);
        }
    }

    return cur;
}

ggml_tensor * llama_memory_xquant_context::get_k(ggml_context * ctx, int32_t il) {
    if (mem.layer_data.size() <= (size_t) il) {
        return nullptr;
    }

    ggml_tensor * x = xq_dequant_concat(ctx, mem.layer_data[il], pending, il, mem.model.hparams.n_embd);
    if (!x) {
        return nullptr;
    }

    const auto & hp = mem.model.hparams;
    ggml_tensor * k = ggml_mul_mat(ctx, mem.model.layers[il].wk, x);
    k = ggml_reshape_3d(ctx, k, hp.n_embd_head_k, hp.n_head_kv(il), get_n_kv());
    return k;
}

ggml_tensor * llama_memory_xquant_context::get_v(ggml_context * ctx, int32_t il) {
    if (mem.layer_data.size() <= (size_t) il) {
        return nullptr;
    }

    ggml_tensor * x = xq_dequant_concat(ctx, mem.layer_data[il], pending, il, mem.model.hparams.n_embd);
    if (!x) {
        return nullptr;
    }

    const auto & hp = mem.model.hparams;
    ggml_tensor * v = ggml_mul_mat(ctx, mem.model.layers[il].wv, x);
    v = ggml_reshape_3d(ctx, v, hp.n_embd_head_v, hp.n_head_kv(il), get_n_kv());
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
