#include "llama-memory-xquant.h"

#include "llama-batch.h"
#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>

#if defined(__linux__) || defined(__FreeBSD__) || defined(_AIX) || defined(__OpenBSD__)
#include <pwd.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

struct xq_k_block {
    uint32_t block_size = 0;
    uint32_t rank       = 0;
    std::vector<uint8_t> data;
    std::vector<llama::xquant::block_qparams> qparams;
};

class llama_memory_context_xquant final : public llama_memory_context_i {
public:
    explicit llama_memory_context_xquant(llama_memory_status status)
        : status(status) {
    }

    llama_memory_context_xquant(
            llama_memory_xquant & memory,
            std::vector<llama_ubatch> ubatches)
        : status(LLAMA_MEMORY_STATUS_SUCCESS),
          memory(&memory),
          ubatches(std::move(ubatches)) {
    }

    bool next() override {
        if (status != LLAMA_MEMORY_STATUS_SUCCESS) {
            return false;
        }
        if (ubatches.empty()) {
            return false;
        }
        prepared = false;
        if (i_cur + 1 >= ubatches.size()) {
            return false;
        }
        ++i_cur;
        return true;
    }

    bool apply() override {
        if (status != LLAMA_MEMORY_STATUS_SUCCESS) {
            return false;
        }
        prepared = true;
        return true;
    }

    const llama_ubatch & get_ubatch() const override {
        if (ubatches.empty()) {
            GGML_ABORT("llama_memory_context_xquant: no ubatches available");
        }
        return ubatches[i_cur];
    }

    llama_memory_status get_status() const override {
        return status;
    }

    void after_graph(const llm_graph_result * res) override {
        if (!prepared || !memory || !res) {
            return;
        }
        if (i_cur >= ubatches.size()) {
            return;
        }
        memory->ingest_post_ln(ubatches[i_cur], res->get_xquant_taps());
    }

private:
    llama_memory_status status = LLAMA_MEMORY_STATUS_FAILED_PREPARE;
    llama_memory_xquant * memory = nullptr;
    std::vector<llama_ubatch> ubatches;
    size_t i_cur = 0;
    bool prepared = false;
};

namespace {

constexpr size_t XQ_LATENT_BLOCK_TOKENS = 128;

constexpr char     SVD_MAGIC[]        = {'X', 'Q', 'S', 'V', '1', '\0'};
constexpr uint32_t SVD_VERSION        = 1;
constexpr uint32_t SVD_FLAG_HAS_UKV   = 1u << 0;

struct xq_svd_blob {
    struct layer_factors {
        uint32_t rank_k  = 0;
        uint32_t rank_v  = 0;
        uint32_t rank_kv = 0;
        std::vector<float> uk;
        std::vector<float> uv;
        std::vector<float> ukv;
        std::vector<float> skbt;
        std::vector<float> svbt;
    };

    uint32_t dim_model = 0;
    uint32_t dim_k     = 0;
    uint32_t dim_v     = 0;
    bool     has_ukv   = false;
    std::string source_path;
    std::vector<layer_factors> layers;
};

[[nodiscard]] static size_t safe_mul(uint32_t a, uint32_t b, const fs::path & path) {
    if (a == 0 || b == 0) {
        return 0;
    }
    const size_t sa = static_cast<size_t>(a);
    const size_t sb = static_cast<size_t>(b);
    if (sa > std::numeric_limits<size_t>::max() / sb) {
        throw std::runtime_error("XQuant SVD file '" + path.string() + "' is too large to load (overflow)");
    }
    return sa * sb;
}

[[nodiscard]] static uint32_t read_u32(std::ifstream & fin, const fs::path & path) {
    uint32_t value = 0;
    fin.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!fin) {
        throw std::runtime_error("Unexpected EOF while reading XQuant SVD file '" + path.string() + "'");
    }
    return value;
}

[[nodiscard]] static std::vector<float> read_f32_array(std::ifstream & fin, size_t count, const fs::path & path) {
    std::vector<float> data(count);
    if (count == 0) {
        return data;
    }
    fin.read(reinterpret_cast<char *>(data.data()), sizeof(float) * static_cast<std::streamsize>(count));
    if (!fin) {
        throw std::runtime_error("Truncated XQuant SVD data in '" + path.string() + "'");
    }
    return data;
}

[[nodiscard]] static bool file_present(const fs::path & candidate) {
    std::error_code ec;
    return fs::exists(candidate, ec) && fs::is_regular_file(candidate, ec);
}

[[nodiscard]] static fs::path cache_directory() {
    if (const char * env = std::getenv("LLAMA_CACHE")) {
        if (*env) {
            return fs::path(env);
        }
    }

#if defined(__linux__) || defined(__FreeBSD__) || defined(_AIX) || defined(__OpenBSD__)
    if (const char * xdg = std::getenv("XDG_CACHE_HOME")) {
        if (*xdg) {
            return fs::path(xdg) / "llama.cpp";
        }
    }
    if (const char * home = std::getenv("HOME")) {
        if (*home) {
            return fs::path(home) / ".cache" / "llama.cpp";
        }
    }
#if defined(__linux__)
    if (struct passwd * pw = getpwuid(getuid()); pw && pw->pw_dir) {
        return fs::path(pw->pw_dir) / ".cache" / "llama.cpp";
    }
#endif
#elif defined(__APPLE__)
    if (const char * home = std::getenv("HOME")) {
        if (*home) {
            return fs::path(home) / "Library" / "Caches" / "llama.cpp";
        }
    }
#elif defined(_WIN32)
    if (const char * local_app = std::getenv("LOCALAPPDATA")) {
        if (*local_app) {
            return fs::path(local_app) / "llama.cpp";
        }
    }
#endif

    return {};
}

[[nodiscard]] static std::string sanitize_name(std::string source) {
    if (source.empty()) {
        return "model";
    }
    for (char & ch : source) {
        if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '-' && ch != '_' && ch != '.') {
            ch = '_';
        }
    }
    return source;
}

[[nodiscard]] static std::string default_svd_filename(const llama_model & model) {
    if (!model.model_path.empty()) {
        fs::path gguf_path(model.model_path);
        if (!gguf_path.stem().empty()) {
            return gguf_path.stem().string() + ".xqsvd";
        }
    }
    if (!model.name.empty()) {
        return sanitize_name(model.name) + ".xqsvd";
    }
    return "model.xqsvd";
}

[[nodiscard]] static std::unique_ptr<xq_svd_blob> parse_svd_file(const fs::path & path) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        throw std::runtime_error("Failed to open XQuant SVD file '" + path.string() + "'");
    }

    std::array<char, sizeof(SVD_MAGIC)> magic{};
    fin.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!fin) {
        throw std::runtime_error("XQuant SVD file '" + path.string() + "' is invalid");
    }
    if (std::strncmp(magic.data(), SVD_MAGIC, magic.size()) != 0) {
        throw std::runtime_error("XQuant SVD file '" + path.string() + "' has unexpected magic header");
    }

    const uint32_t version = read_u32(fin, path);
    if (version != SVD_VERSION) {
        throw std::runtime_error("XQuant SVD file '" + path.string() + "' has unsupported version " + std::to_string(version));
    }

    const uint32_t n_layers  = read_u32(fin, path);
    const uint32_t dim_model = read_u32(fin, path);
    const uint32_t dim_k     = read_u32(fin, path);
    const uint32_t dim_v     = read_u32(fin, path);
    const uint32_t flags     = read_u32(fin, path);

    auto blob       = std::make_unique<xq_svd_blob>();
    blob->dim_model = dim_model;
    blob->dim_k     = dim_k;
    blob->dim_v     = dim_v;
    blob->has_ukv   = (flags & SVD_FLAG_HAS_UKV) != 0;
    blob->source_path = path.string();
    blob->layers.resize(n_layers);

    for (uint32_t il = 0; il < n_layers; ++il) {
        auto & layer = blob->layers[il];
        layer.rank_k  = read_u32(fin, path);
        layer.rank_v  = read_u32(fin, path);
        layer.rank_kv = read_u32(fin, path);
    }

    for (uint32_t il = 0; il < n_layers; ++il) {
        auto & layer = blob->layers[il];
        layer.uk   = read_f32_array(fin, safe_mul(blob->dim_model, layer.rank_k, path), path);
        layer.uv   = read_f32_array(fin, safe_mul(blob->dim_model, layer.rank_v, path), path);
        if (blob->has_ukv && layer.rank_kv > 0) {
            layer.ukv = read_f32_array(fin, safe_mul(blob->dim_model, layer.rank_kv, path), path);
        } else {
            layer.ukv.clear();
        }
        layer.skbt = read_f32_array(fin, safe_mul(layer.rank_k, blob->dim_k, path), path);
        layer.svbt = read_f32_array(fin, safe_mul(layer.rank_v, blob->dim_v, path), path);
    }

    return blob;
}

static void validate_svd_blob(const xq_svd_blob & blob, const llama_model & model) {
    if (blob.layers.size() != model.layers.size()) {
        throw std::runtime_error("XQuant SVD file '" + blob.source_path + "' has " +
                                 std::to_string(blob.layers.size()) + " layers, expected " +
                                 std::to_string(model.layers.size()));
    }

    if (blob.dim_model != model.hparams.n_embd) {
        throw std::runtime_error("XQuant SVD file '" + blob.source_path + "' mismatches model embedding dim");
    }

    if (blob.dim_k != model.hparams.n_embd_k_gqa() || blob.dim_v != model.hparams.n_embd_v_gqa()) {
        throw std::runtime_error("XQuant SVD file '" + blob.source_path + "' mismatches model attention dims");
    }
}

[[nodiscard]] static fs::path resolve_svd_path(const llama_model & model, const llama_cparams & cparams, const std::string & default_file) {
    if (!cparams.xq_svd_path.empty()) {
        fs::path user_path(cparams.xq_svd_path);
        std::error_code ec;
        if (fs::is_directory(user_path, ec)) {
            user_path /= default_file;
        }
        if (!file_present(user_path)) {
            throw std::runtime_error("XQuant SVD file '" + user_path.string() + "' does not exist");
        }
        return user_path;
    }

    std::vector<fs::path> candidates;
    if (!model.model_path.empty()) {
        fs::path gguf_path(model.model_path);
        fs::path sibling = gguf_path;
        sibling.replace_extension(".xqsvd");
        candidates.push_back(sibling);
        if (!gguf_path.parent_path().empty()) {
            candidates.push_back(gguf_path.parent_path() / default_file);
        }
    }

    if (fs::path cache = cache_directory(); !cache.empty()) {
        candidates.push_back(cache / default_file);
    }

    for (const auto & candidate : candidates) {
        if (!candidate.empty() && file_present(candidate)) {
            return candidate;
        }
    }

    return {};
}

static llama_memory_context_ptr make_unimplemented_context() {
    return std::make_unique<llama_memory_context_xquant>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

static void log_unimplemented(const char * func) {
    LLAMA_LOG_ERROR("%s: XQuant memory backend is not implemented yet\n", func);
}

} // namespace

llama_memory_xquant::llama_memory_xquant(const llama_model & model, const llama_cparams & cparams)
    : model(model),
      cparams(cparams),
      dim_model(static_cast<size_t>(model.hparams.n_embd)) {
    if (model.arch != LLM_ARCH_LLAMA) {
        throw std::runtime_error("XQuant currently supports only llama architecture models");
    }

    if (cparams.xq_gqa_svd) {
        const std::string default_file = default_svd_filename(model);
        const fs::path resolved = resolve_svd_path(model, cparams, default_file);
        if (resolved.empty()) {
            throw std::runtime_error(
                "Unable to locate XQuant SVD factors. Provide --xq-svd-path or place '" + default_file +
                "' next to the model.");
        }
        svd_blob = parse_svd_file(resolved);
        validate_svd_blob(*svd_blob, model);
        LLAMA_LOG_INFO("%s: loaded XQuant SVD factors from '%s'\n", __func__, resolved.string().c_str());
    }
    zero_buffer.assign(dim_model, 0.0f);
    init_layer_configs();
    LLAMA_LOG_INFO("%s: initialized XQuant memory scaffolding\n", __func__);
}

llama_memory_xquant::~llama_memory_xquant() = default;

uint32_t llama_memory_xquant::layer_bits(int32_t il) const {
    if (il < static_cast<int32_t>(cparams.xq_base_layers)) {
        return 4;
    }
    return cparams.xq_bits;
}

void llama_memory_xquant::init_layer_configs() {
    const size_t n_layers = model.layers.size();
    const size_t dim_model = static_cast<size_t>(model.hparams.n_embd);
    if (dim_model == 0) {
        throw std::runtime_error("XQuant: invalid model dimension");
    }
    if (cparams.xq_group_size == 0) {
        throw std::runtime_error("XQuant: group size must be greater than zero");
    }

    layer_cfgs.resize(n_layers);
    for (size_t il = 0; il < n_layers; ++il) {
        auto & cfg = layer_cfgs[il];
        cfg.spec.bits       = layer_bits(static_cast<int32_t>(il));
        cfg.spec.group_size = cparams.xq_group_size;
        cfg.dim             = dim_model;
        cfg.store_delta     = cparams.xquant_cl;
        cfg.block_sizes.clear();
        cfg.block_offsets.clear();
        cfg.block_nbytes.clear();
        cfg.bytes_per_token = 0;

        size_t offset = 0;
        while (offset < dim_model) {
            const size_t block = std::min<size_t>(cfg.spec.group_size, dim_model - offset);
            cfg.block_sizes.push_back(block);
            cfg.block_offsets.push_back(cfg.bytes_per_token);
            const size_t block_bytes = llama::xquant::block_data_bytes(cfg.spec.bits, block);
            cfg.block_nbytes.push_back(block_bytes);
            cfg.bytes_per_token += block_bytes;
            offset += block;
        }

        if (cparams.xq_gqa_svd && svd_blob) {
            init_svd_layer(cfg, il);
        } else {
            cfg.rank_k = cfg.rank_v = cfg.rank_kv = 0;
            cfg.latent_v_block_sizes.clear();
            cfg.latent_v_block_offsets.clear();
            cfg.latent_v_block_nbytes.clear();
            cfg.latent_v_bytes_per_token = 0;
            cfg.dim_k = model.hparams.n_embd_k_gqa(il);
            cfg.dim_v = model.hparams.n_embd_v_gqa(il);
            cfg.uk = cfg.uv = cfg.ukv = nullptr;
            cfg.skbt = cfg.svbt = nullptr;
        }
    }
}

void llama_memory_xquant::init_svd_layer(xq_layer_config & cfg, size_t il) {
    if (!svd_blob || il >= svd_blob->layers.size()) {
        cfg.rank_k = cfg.rank_v = cfg.rank_kv = 0;
        cfg.latent_v_block_sizes.clear();
        cfg.latent_v_block_offsets.clear();
        cfg.latent_v_block_nbytes.clear();
        cfg.latent_v_bytes_per_token = 0;
        cfg.uk = cfg.uv = cfg.ukv = nullptr;
        cfg.skbt = cfg.svbt = nullptr;
        cfg.dim_k = model.hparams.n_embd_k_gqa(il);
        cfg.dim_v = model.hparams.n_embd_v_gqa(il);
        return;
    }

    const auto & fac = svd_blob->layers[il];
    cfg.rank_k  = fac.rank_k;
    cfg.rank_v  = fac.rank_v;
    cfg.rank_kv = fac.rank_kv;
    cfg.uk  = fac.uk.empty()  ? nullptr : fac.uk.data();
    cfg.uv  = fac.uv.empty()  ? nullptr : fac.uv.data();
    cfg.ukv = fac.ukv.empty() ? nullptr : fac.ukv.data();
    cfg.skbt = fac.skbt.empty() ? nullptr : fac.skbt.data();
    cfg.svbt = fac.svbt.empty() ? nullptr : fac.svbt.data();
    cfg.dim_k = model.hparams.n_embd_k_gqa(il);
    cfg.dim_v = model.hparams.n_embd_v_gqa(il);

    cfg.latent_v_block_sizes.clear();
    cfg.latent_v_block_offsets.clear();
    cfg.latent_v_block_nbytes.clear();
    cfg.latent_v_bytes_per_token = 0;

    size_t offset = 0;
    while (offset < cfg.rank_v) {
        const size_t block = std::min<size_t>(cfg.spec.group_size, cfg.rank_v - offset);
        cfg.latent_v_block_sizes.push_back(block);
        cfg.latent_v_block_offsets.push_back(cfg.latent_v_bytes_per_token);
        const size_t block_bytes = llama::xquant::block_data_bytes(cfg.spec.bits, block);
        cfg.latent_v_block_nbytes.push_back(block_bytes);
        cfg.latent_v_bytes_per_token += block_bytes;
        offset += block;
    }
}

void llama_memory_xquant::tensor_to_host(ggml_tensor * tensor, std::vector<float> & out) const {
    const size_t n = ggml_nelements(tensor);
    out.resize(n);

    switch (tensor->type) {
        case GGML_TYPE_F32:
            ggml_backend_tensor_get(tensor, out.data(), 0, n * sizeof(float));
            break;
        case GGML_TYPE_F16:
            {
                std::vector<ggml_fp16_t> tmp(n);
                ggml_backend_tensor_get(tensor, tmp.data(), 0, n * sizeof(ggml_fp16_t));
                for (size_t i = 0; i < n; ++i) {
                    out[i] = ggml_fp16_to_fp32(tmp[i]);
                }
            } break;
        case GGML_TYPE_BF16:
            {
                std::vector<ggml_bf16_t> tmp(n);
                ggml_backend_tensor_get(tensor, tmp.data(), 0, n * sizeof(ggml_bf16_t));
                for (size_t i = 0; i < n; ++i) {
                    out[i] = ggml_bf16_to_fp32(tmp[i]);
                }
            } break;
        default:
            throw std::runtime_error("XQuant: unsupported tensor type for capture");
    }
}

void llama_memory_xquant::compute_latents(const xq_layer_config & cfg,
        const float * token,
        std::vector<float> & out_k,
        std::vector<float> & out_v,
        std::vector<float> & out_kv) const {
    auto project = [&](size_t rank, const float * proj, std::vector<float> & dst) {
        if (rank == 0 || proj == nullptr) {
            dst.clear();
            return;
        }
        dst.resize(rank);
        std::fill(dst.begin(), dst.end(), 0.0f);
        for (size_t i = 0; i < dim_model; ++i) {
            const float xi = token[i];
            const float * row = proj + i*rank;
            for (size_t r = 0; r < rank; ++r) {
                dst[r] += xi * row[r];
            }
        }
    };

    project(cfg.rank_k,  cfg.uk,  out_k);
    project(cfg.rank_v,  cfg.uv,  out_v);
    project(cfg.rank_kv, cfg.ukv, out_kv);
}

llama_memory_xquant::xq_token_state & llama_memory_xquant::ensure_token_state(llama_seq_id seq_id, llama_pos pos) {
    GGML_ASSERT(seq_id >= 0 && seq_id < LLAMA_MAX_SEQ);
    auto & seq = seq_states[seq_id];
    auto [it, inserted] = seq.tokens.try_emplace(pos);
    auto & token = it->second;
    if (inserted) {
        token.pos = pos;
    }
    if (token.layers.size() != layer_cfgs.size()) {
        token.layers.resize(layer_cfgs.size());
    }
    if (cparams.xquant_cl && token.hat_layers.size() != layer_cfgs.size()) {
        token.hat_layers.resize(layer_cfgs.size());
    }
    return token;
}

const std::vector<float> & llama_memory_xquant::get_hat_prev(const xq_token_state & token, int32_t il) const {
    if (!cparams.xquant_cl || il <= 0) {
        return zero_buffer;
    }
    if (token.hat_layers.empty()) {
        return zero_buffer;
    }
    const auto & prev = token.hat_layers[il - 1];
    if (prev.size() != dim_model) {
        return zero_buffer;
    }
    return prev;
}

std::vector<float> & llama_memory_xquant::get_hat_slot(xq_token_state & token, int32_t il) {
    if (token.hat_layers.size() != layer_cfgs.size()) {
        token.hat_layers.resize(layer_cfgs.size());
    }
    auto & cur = token.hat_layers[il];
    if (cur.size() != dim_model) {
        cur.assign(dim_model, 0.0f);
    }
    return cur;
}

void llama_memory_xquant::append_latent_v(xq_layer_payload & payload, const xq_layer_config & cfg, const std::vector<float> & latent_v) {
    if (latent_v.empty()) {
        payload.latent_v.kind = xq_layer_payload::storage_kind::none;
        payload.latent_v.data.clear();
        payload.latent_v.qparams.clear();
        payload.latent_v.float_data.clear();
        return;
    }

    if (cfg.latent_v_bytes_per_token == 0) {
        payload.latent_v.kind = xq_layer_payload::storage_kind::floating;
        payload.latent_v.data.clear();
        payload.latent_v.qparams.clear();
        payload.latent_v.float_data = latent_v;
        return;
    }

    payload.latent_v.kind = xq_layer_payload::storage_kind::quantized;
    payload.latent_v.float_data.clear();
    payload.latent_v.data.resize(cfg.latent_v_bytes_per_token);
    payload.latent_v.qparams.resize(cfg.latent_v_block_sizes.size());

    size_t data_offset = 0;
    size_t dim_offset  = 0;
    for (size_t ib = 0; ib < cfg.latent_v_block_sizes.size(); ++ib) {
        llama::xquant::block_qparams qp{};
        const size_t block = cfg.latent_v_block_sizes[ib];
        llama::xquant::quantize_block(
                latent_v.data() + dim_offset,
                block,
                cfg.spec,
                payload.latent_v.data.data() + data_offset,
                qp);
        payload.latent_v.qparams[ib] = qp;
        data_offset += cfg.latent_v_block_nbytes[ib];
        dim_offset  += block;
    }
}

void llama_memory_xquant::append_latent_k(llama_seq_id seq_id, int32_t il, xq_layer_payload & payload, const std::vector<float> & latent_k) {
    if (latent_k.empty() || seq_id < 0 || seq_id >= LLAMA_MAX_SEQ) {
        payload.latent_k.kind = xq_layer_payload::storage_kind::none;
        payload.latent_k.block.reset();
        payload.latent_k.float_data.clear();
        return;
    }

    auto & seq = seq_states[seq_id];
    const size_t layer_index = static_cast<size_t>(il);
    if (seq.pending_k.size() < layer_cfgs.size()) {
        seq.pending_k.resize(layer_cfgs.size());
    }
    auto & pending = seq.pending_k[layer_index];
    const size_t rank = latent_k.size();
    if (pending.buffer.size() != rank * XQ_LATENT_BLOCK_TOKENS) {
        pending.buffer.assign(rank * XQ_LATENT_BLOCK_TOKENS, 0.0f);
    }
    if (pending.payloads.size() >= XQ_LATENT_BLOCK_TOKENS) {
        finalize_latent_k_block(pending, layer_cfgs[layer_index]);
    }
    const size_t idx = pending.count;
    for (size_t r = 0; r < rank; ++r) {
        pending.buffer[r*XQ_LATENT_BLOCK_TOKENS + idx] = latent_k[r];
    }
    pending.payloads.push_back(&payload);
    payload.latent_k.kind = xq_layer_payload::storage_kind::floating;
    payload.latent_k.block.reset();
    payload.latent_k.offset = static_cast<uint16_t>(idx);
    payload.latent_k.float_data = latent_k;
    pending.count++;
    if (pending.count == XQ_LATENT_BLOCK_TOKENS) {
        finalize_latent_k_block(pending, layer_cfgs[layer_index]);
    }
}

void llama_memory_xquant::finalize_latent_k_block(xq_sequence_state::pending_latent_k & pending, const xq_layer_config & cfg) {
    if (cfg.rank_k == 0 || pending.count != XQ_LATENT_BLOCK_TOKENS) {
        return;
    }

    const size_t rank = cfg.rank_k;
    const size_t bytes_per_channel = llama::xquant::block_data_bytes(cfg.spec.bits, XQ_LATENT_BLOCK_TOKENS);

    auto block = std::make_shared<xq_k_block>();
    block->block_size = XQ_LATENT_BLOCK_TOKENS;
    block->rank       = static_cast<uint32_t>(rank);
    block->data.resize(rank * bytes_per_channel);
    block->qparams.resize(rank);

    for (size_t r = 0; r < rank; ++r) {
        llama::xquant::block_qparams params{};
        llama::xquant::quantize_block(
                pending.buffer.data() + r*XQ_LATENT_BLOCK_TOKENS,
                XQ_LATENT_BLOCK_TOKENS,
                cfg.spec,
                block->data.data() + r*bytes_per_channel,
                params);
        block->qparams[r] = params;
    }

    for (size_t idx = 0; idx < pending.payloads.size(); ++idx) {
        auto * payload = pending.payloads[idx];
        payload->latent_k.kind = xq_layer_payload::storage_kind::block_ref;
        payload->latent_k.block = block;
        payload->latent_k.offset = static_cast<uint16_t>(idx);
        payload->latent_k.float_data.clear();
    }

    pending.payloads.clear();
    pending.count = 0;
}

void llama_memory_xquant::dequantize_payload(const xq_layer_config & cfg, const xq_layer_payload & payload, std::vector<float> & out) const {
    out.resize(cfg.dim);
    switch (payload.kind) {
        case xq_layer_payload::storage_kind::none:
            std::fill(out.begin(), out.end(), 0.0f);
            break;
        case xq_layer_payload::storage_kind::floating:
            out = payload.float_data;
            out.resize(cfg.dim, 0.0f);
            break;
        case xq_layer_payload::storage_kind::quantized:
        case xq_layer_payload::storage_kind::block_ref:
            {
                size_t data_offset = 0;
                size_t dim_offset  = 0;
                for (size_t ib = 0; ib < cfg.block_sizes.size(); ++ib) {
                    const size_t block = cfg.block_sizes[ib];
                    const auto & qp = payload.qparams[ib];
                    llama::xquant::dequantize_block(
                            payload.data.data() + data_offset,
                            block,
                            cfg.spec,
                            qp,
                            out.data() + dim_offset);
                    data_offset += cfg.block_nbytes[ib];
                    dim_offset  += block;
                }
            } break;
    }
}

void llama_memory_xquant::reconstruct_token(const xq_layer_config & cfg, const xq_token_state & token_state, const xq_layer_payload & payload, int32_t il, std::vector<float> & out) const {
    if (cparams.xquant_cl && il < (int) token_state.hat_layers.size()) {
        const auto & hat = token_state.hat_layers[il];
        if (hat.size() == cfg.dim) {
            out = hat;
            return;
        }
    }
    dequantize_payload(cfg, payload, out);
}

void llama_memory_xquant::update_seq_bounds(llama_seq_id seq_id) {
    if (seq_id < 0 || seq_id >= LLAMA_MAX_SEQ) {
        return;
    }
    auto & seq = seq_states[seq_id];
    auto & bounds = seq_bounds[seq_id];
    if (seq.tokens.empty()) {
        bounds.pos_min = -1;
        bounds.pos_max = -1;
        return;
    }
    bounds.pos_min = seq.tokens.begin()->first;
    bounds.pos_max = seq.tokens.rbegin()->first;
}

void llama_memory_xquant::store_layer_tokens(
        int32_t il,
        const float * data,
        size_t n_embd,
        size_t n_tokens,
        const llama_ubatch & ubatch,
        std::array<bool, LLAMA_MAX_SEQ> & touched) {
    if (il < 0 || static_cast<size_t>(il) >= layer_cfgs.size()) {
        return;
    }
    const auto & cfg = layer_cfgs[il];
    if (cfg.bytes_per_token == 0 || cfg.block_sizes.empty()) {
        return;
    }
    if (n_embd < cfg.dim) {
        LLAMA_LOG_WARN("%s: layer %d expected %zu dims but captured %zu\n",
                __func__, il, cfg.dim, n_embd);
        return;
    }

    std::vector<uint8_t> qdata(cfg.bytes_per_token);
    std::vector<llama::xquant::block_qparams> qparams(cfg.block_sizes.size());
    std::vector<float> delta_buffer(cfg.store_delta ? cfg.dim : 0);
    std::vector<float> latent_k(cfg.rank_k);
    std::vector<float> latent_v(cfg.rank_v);
    std::vector<float> latent_kv(cfg.rank_kv);

    auto quantize_into = [&](const float * src, std::vector<uint8_t> & dst, std::vector<llama::xquant::block_qparams> & params) {
        dst.resize(cfg.bytes_per_token);
        params.resize(cfg.block_sizes.size());
        size_t data_offset = 0;
        size_t dim_offset  = 0;
        for (size_t ib = 0; ib < cfg.block_sizes.size(); ++ib) {
            llama::xquant::block_qparams qp{};
            const size_t block_size = cfg.block_sizes[ib];
            llama::xquant::quantize_block(
                    src + dim_offset,
                    block_size,
                    cfg.spec,
                    dst.data() + data_offset,
                    qp);
            params[ib] = qp;
            data_offset += cfg.block_nbytes[ib];
            dim_offset  += block_size;
        }
    };

    for (size_t ti = 0; ti < n_tokens; ++ti) {
        const float * token = data + ti * n_embd;

        if (cfg.rank_k > 0 || cfg.rank_v > 0 || cfg.rank_kv > 0) {
            compute_latents(cfg, token, latent_k, latent_v, latent_kv);
        } else {
            latent_k.clear();
            latent_v.clear();
            latent_kv.clear();
        }
        GGML_UNUSED(latent_kv);

        bool base_ready = false;
        auto ensure_base = [&]() {
            if (base_ready || cfg.store_delta) {
                return;
            }
            quantize_into(token, qdata, qparams);
            base_ready = true;
        };

        if (ti >= ubatch.n_tokens) {
            continue;
        }

        const int32_t n_seq = ubatch.n_seq_id[ti];
        if (n_seq <= 0 || ubatch.seq_id[ti] == nullptr) {
            continue;
        }

        const llama_pos pos = ubatch.pos[ti];
        for (int32_t s = 0; s < n_seq; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id[ti][s];
            if (seq_id < 0 || seq_id >= LLAMA_MAX_SEQ) {
                continue;
            }

            auto & token_state = ensure_token_state(seq_id, pos);
            auto & payload     = token_state.layers[il];

            if (cfg.store_delta) {
                const auto & hat_prev = get_hat_prev(token_state, il);
                for (size_t j = 0; j < cfg.dim; ++j) {
                    const float prev = (hat_prev.size() == dim_model) ? hat_prev[j] : zero_buffer[j];
                    delta_buffer[j] = token[j] - prev;
                }
                quantize_into(delta_buffer.data(), payload.data, payload.qparams);
                payload.kind = xq_layer_payload::storage_kind::quantized;
                payload.float_data.clear();

                auto & hat_cur = get_hat_slot(token_state, il);
                for (size_t j = 0; j < cfg.dim; ++j) {
                    const float prev = (hat_prev.size() == dim_model) ? hat_prev[j] : zero_buffer[j];
                    hat_cur[j] = prev + delta_buffer[j];
                }
            } else {
                ensure_base();
                payload.kind = xq_layer_payload::storage_kind::quantized;
                payload.data = qdata;
                payload.qparams = qparams;
                payload.float_data.clear();
            }

            if (!latent_k.empty()) {
                append_latent_k(seq_id, il, payload, latent_k);
            } else {
                payload.latent_k.kind = xq_layer_payload::storage_kind::none;
                payload.latent_k.block.reset();
                payload.latent_k.float_data.clear();
            }

            if (!latent_v.empty()) {
                append_latent_v(payload, cfg, latent_v);
            } else {
                payload.latent_v.kind = xq_layer_payload::storage_kind::none;
                payload.latent_v.data.clear();
                payload.latent_v.qparams.clear();
                payload.latent_v.float_data.clear();
            }

            touched[seq_id] = true;
        }
    }

void llama_memory_xquant::ingest_post_ln(const llama_ubatch & ubatch, const std::vector<llm_graph_result::xquant_tap> & taps) {
    if (ubatch.n_tokens == 0 || taps.empty()) {
        return;
    }

    std::array<bool, LLAMA_MAX_SEQ> touched{};
    std::vector<float> host_buffer;

    for (const auto & tap : taps) {
        if (!tap.tensor) {
            continue;
        }
        const int32_t il = tap.layer;
        if (il < 0 || static_cast<size_t>(il) >= layer_cfgs.size()) {
            continue;
        }

        const int64_t n_embd   = tap.tensor->ne[0];
        const int64_t n_tokens = tap.tensor->ne[1];

        if (n_tokens != static_cast<int64_t>(ubatch.n_tokens)) {
            LLAMA_LOG_WARN("%s: tap layer %d token mismatch (%lld vs %u)\n",
                    __func__, il, (long long) n_tokens, ubatch.n_tokens);
            continue;
        }

        if (static_cast<size_t>(n_embd) != model.hparams.n_embd) {
            LLAMA_LOG_WARN("%s: tap layer %d embedding mismatch (%lld vs %d)\n",
                    __func__, il, (long long) n_embd, model.hparams.n_embd);
        }

        tensor_to_host(tap.tensor, host_buffer);
        store_layer_tokens(il, host_buffer.data(), static_cast<size_t>(n_embd), static_cast<size_t>(n_tokens), ubatch, touched);
    }

    for (int seq_id = 0; seq_id < LLAMA_MAX_SEQ; ++seq_id) {
        if (touched[seq_id]) {
            update_seq_bounds(seq_id);
        }
    }
}

llama_memory_context_ptr llama_memory_xquant::init_batch(
        llama_batch_allocr & balloc,
        uint32_t n_ubatch,
        bool embd_all) {
    GGML_UNUSED(embd_all);

    balloc.split_reset();

    std::vector<llama_ubatch> ubatches;
    while (true) {
        auto ubatch = balloc.split_simple(n_ubatch);
        if (ubatch.n_tokens == 0) {
            break;
        }
        ubatches.push_back(std::move(ubatch));
    }

    if (balloc.get_n_used() < balloc.get_n_tokens() || ubatches.empty()) {
        LLAMA_LOG_ERROR("%s: failed to prepare ubatch for XQuant memory\n", __func__);
        return std::make_unique<llama_memory_context_xquant>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    return std::make_unique<llama_memory_context_xquant>(*this, std::move(ubatches));
}

llama_memory_context_ptr llama_memory_xquant::init_full() {
    return std::make_unique<llama_memory_context_xquant>(LLAMA_MEMORY_STATUS_SUCCESS);
}

llama_memory_context_ptr llama_memory_xquant::init_update(llama_context *, bool) {
    return std::make_unique<llama_memory_context_xquant>(LLAMA_MEMORY_STATUS_NO_UPDATE);
}

bool llama_memory_xquant::get_can_shift() const {
    return true;
}

void llama_memory_xquant::clear(bool) {
    for (auto & seq : seq_states) {
        seq.tokens.clear();
        seq.pending_k.clear();
    }
    for (auto & bounds : seq_bounds) {
        bounds.pos_min = -1;
        bounds.pos_max = -1;
    }
}

static inline llama_pos clamp_start(llama_pos p0) {
    return p0 < 0 ? 0 : p0;
}

static inline llama_pos clamp_end(llama_pos p1) {
    return p1 < 0 ? std::numeric_limits<llama_pos>::max() : p1;
}

bool llama_memory_xquant::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    const llama_pos start = clamp_start(p0);
    const llama_pos end   = clamp_end(p1);

    auto remove_range = [&](llama_seq_id sid) {
        if (sid < 0 || sid >= LLAMA_MAX_SEQ) {
            return;
        }
        auto & seq = seq_states[sid];
        if (seq.tokens.empty() || start >= end) {
            seq.pending_k.clear();
            return;
        }
        auto it = seq.tokens.lower_bound(start);
        bool changed = false;
        while (it != seq.tokens.end() && it->first < end) {
            it = seq.tokens.erase(it);
            changed = true;
        }
        if (changed) {
            update_seq_bounds(sid);
        }
        seq.pending_k.clear();
    };

    if (seq_id == -1) {
        for (llama_seq_id sid = 0; sid < LLAMA_MAX_SEQ; ++sid) {
            remove_range(sid);
        }
    } else {
        remove_range(seq_id);
    }

    return true;
}

void llama_memory_xquant::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    if (seq_id_src < 0 || seq_id_src >= LLAMA_MAX_SEQ || seq_id_dst < 0 || seq_id_dst >= LLAMA_MAX_SEQ) {
        return;
    }
    const llama_pos start = clamp_start(p0);
    const llama_pos end   = clamp_end(p1);
    if (start >= end) {
        return;
    }

    const auto & src = seq_states[seq_id_src];
    auto & dst = seq_states[seq_id_dst];

    auto it = src.tokens.lower_bound(start);
    while (it != src.tokens.end() && it->first < end) {
        dst.tokens[it->first] = it->second;
        ++it;
    }
    update_seq_bounds(seq_id_dst);
}

void llama_memory_xquant::seq_keep(llama_seq_id) {
    // No-op: sequences are stored independently.
}

void llama_memory_xquant::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    if (seq_id < 0 || seq_id >= LLAMA_MAX_SEQ || shift == 0) {
        return;
    }
    auto & seq = seq_states[seq_id];
    if (seq.tokens.empty()) {
        return;
    }

    const llama_pos start = clamp_start(p0);
    const llama_pos end   = clamp_end(p1);
    if (start == end) {
        return;
    }

    std::vector<std::pair<llama_pos, xq_token_state>> moved;
    auto it = seq.tokens.lower_bound(start);
    while (it != seq.tokens.end() && it->first < end) {
        auto node = std::move(it->second);
        const llama_pos new_pos = it->first + shift;
        it = seq.tokens.erase(it);
        node.pos = new_pos;
        moved.emplace_back(new_pos, std::move(node));
    }
    for (auto & entry : moved) {
        seq.tokens[entry.first] = std::move(entry.second);
    }
    update_seq_bounds(seq_id);
}

void llama_memory_xquant::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    if (seq_id < 0 || seq_id >= LLAMA_MAX_SEQ || d <= 1) {
        return;
    }
    auto & seq = seq_states[seq_id];
    if (seq.tokens.empty()) {
        return;
    }

    const llama_pos start = clamp_start(p0);
    const llama_pos end   = clamp_end(p1);
    if (start == end) {
        return;
    }

    std::vector<std::pair<llama_pos, xq_token_state>> moved;
    auto it = seq.tokens.lower_bound(start);
    while (it != seq.tokens.end() && it->first < end) {
        auto node = std::move(it->second);
        const llama_pos new_pos = node.pos / d;
        it = seq.tokens.erase(it);
        node.pos = new_pos;
        moved.emplace_back(new_pos, std::move(node));
    }
    for (auto & entry : moved) {
        seq.tokens[entry.first] = std::move(entry.second);
    }
    update_seq_bounds(seq_id);
}

llama_pos llama_memory_xquant::seq_pos_min(llama_seq_id seq_id) const {
    if (seq_id < 0 || seq_id >= LLAMA_MAX_SEQ) {
        return -1;
    }
    return seq_bounds[seq_id].pos_min;
}

llama_pos llama_memory_xquant::seq_pos_max(llama_seq_id seq_id) const {
    if (seq_id < 0 || seq_id >= LLAMA_MAX_SEQ) {
        return -1;
    }
    return seq_bounds[seq_id].pos_max;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_xquant::memory_breakdown() const {
    size_t total = 0;
    std::unordered_set<const xq_k_block *> counted_blocks;
    for (const auto & seq : seq_states) {
        for (const auto & kv : seq.tokens) {
            const auto & token = kv.second;
            for (const auto & payload : token.layers) {
                switch (payload.kind) {
                    case xq_layer_payload::storage_kind::none:
                        break;
                    case xq_layer_payload::storage_kind::quantized:
                    case xq_layer_payload::storage_kind::block_ref:
                        total += payload.data.size();
                        total += payload.qparams.size() * sizeof(llama::xquant::block_qparams);
                        break;
                    case xq_layer_payload::storage_kind::floating:
                        total += payload.float_data.size() * sizeof(float);
                        break;
                }

                if (payload.latent_k.kind == xq_layer_payload::storage_kind::block_ref && payload.latent_k.block) {
                    if (counted_blocks.insert(payload.latent_k.block.get()).second) {
                        total += payload.latent_k.block->data.size();
                        total += payload.latent_k.block->qparams.size() * sizeof(llama::xquant::block_qparams);
                    }
                }

                switch (payload.latent_v.kind) {
                    case xq_layer_payload::storage_kind::none:
                        break;
                    case xq_layer_payload::storage_kind::quantized:
                        total += payload.latent_v.data.size();
                        total += payload.latent_v.qparams.size() * sizeof(llama::xquant::block_qparams);
                        break;
                    case xq_layer_payload::storage_kind::floating:
                        total += payload.latent_v.float_data.size() * sizeof(float);
                        break;
                    case xq_layer_payload::storage_kind::block_ref:
                        // not used for V
                        break;
                }

                if (payload.latent_k.kind == xq_layer_payload::storage_kind::floating) {
                    total += payload.latent_k.float_data.size() * sizeof(float);
                }
            }
        }
    }

    if (total == 0) {
        return {};
    }

    std::map<ggml_backend_buffer_type_t, size_t> result;
    result[ggml_backend_cpu_buffer_type()] = total;
    return result;
}

void llama_memory_xquant::state_write(llama_io_write_i &, llama_seq_id, llama_state_seq_flags) const {
    log_unimplemented(__func__);
}

void llama_memory_xquant::state_read(llama_io_read_i &, llama_seq_id, llama_state_seq_flags) {
    log_unimplemented(__func__);
}
