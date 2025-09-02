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
    if (mem.layer_data.size() <= (size_t) il) {
        mem.layer_data.resize(il + 1);
    }
    ggml_tensor * q = llama_xq_quantize(ctx, x_cur, 4);
    mem.layer_data[il].push_back(q);
    return q;
}

uint32_t llama_memory_xquant_context::get_n_kv() const {
    if (mem.layer_data.empty()) {
        return 0;
    }

    uint32_t n = 0;
    for (ggml_tensor * t : mem.layer_data[0]) {
        n += t->ne[1];
    }
    return n;
}

// helper: dequantize and concatenate cached X for layer il
static ggml_tensor * xq_dequant_concat(ggml_context * ctx, const std::vector<ggml_tensor *> & qs) {
    ggml_tensor * cur = nullptr;
    for (ggml_tensor * q : qs) {
        ggml_tensor * deq = ggml_cast(ctx, q, GGML_TYPE_F32);
        // ensure the tensor is 2-D so future concats see matching shapes
        deq = ggml_reshape_2d(ctx, deq, deq->ne[0], deq->ne[1]);
        if (!cur) {
            cur = deq;
        } else {
            cur = ggml_concat(ctx, cur, deq, 1);
            // ggml_concat may promote the tensor to 4-D; fold back to 2-D to
            // avoid dimension mismatches on subsequent concatenations
            cur = ggml_reshape_2d(ctx, cur, cur->ne[0], cur->ne[1]);
        }
    }
    return cur;
}

ggml_tensor * llama_memory_xquant_context::get_k(ggml_context * ctx, int32_t il) {
    if (mem.layer_data.size() <= (size_t) il) {
        return nullptr;
    }

    ggml_tensor * x = xq_dequant_concat(ctx, mem.layer_data[il]);
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

    ggml_tensor * x = xq_dequant_concat(ctx, mem.layer_data[il]);
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
