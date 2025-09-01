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

llama_memory_context_ptr llama_memory_xquant::init_batch(llama_batch_allocr &, uint32_t, bool) {
    return std::make_unique<llama_memory_xquant_context>(*this);
}

llama_memory_context_ptr llama_memory_xquant::init_full() {
    return std::make_unique<llama_memory_xquant_context>(*this);
}

llama_memory_context_ptr llama_memory_xquant::init_update(llama_context *, bool) {
    return std::make_unique<llama_memory_xquant_context>(*this);
}
