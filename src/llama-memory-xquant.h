#pragma once

#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-memory.h"
#include "llama-xq-quant.h"

#include <array>
#include <map>
#include <memory>
#include <vector>

struct ggml_tensor;
struct llama_model;
struct xq_svd_blob;

// forward declaration for context pointer
class llama_memory_context_xquant;

class llama_memory_xquant : public llama_memory_i {
public:
    llama_memory_xquant(const llama_model & model, const llama_cparams & cparams);
    ~llama_memory_xquant() override;

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id) override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

protected:
    const llama_model & model;
    const llama_cparams & cparams;
    std::unique_ptr<xq_svd_blob> svd_blob;

    void ingest_post_ln(const llama_ubatch & ubatch, const std::vector<llm_graph_result::xquant_tap> & taps);

    struct xq_layer_config {
        llama::xquant::block_spec spec;
        size_t dim = 0;
        bool   store_delta = false;
        std::vector<size_t> block_sizes;
        std::vector<size_t> block_offsets;
        std::vector<size_t> block_nbytes;
        size_t bytes_per_token = 0;

        size_t rank_k  = 0;
        size_t rank_v  = 0;
        size_t rank_kv = 0;
        const float * uk  = nullptr;
        const float * uv  = nullptr;
        const float * ukv = nullptr;
        const float * skbt = nullptr;
        const float * svbt = nullptr;
        size_t dim_k = 0;
        size_t dim_v = 0;
        std::vector<size_t> latent_v_block_sizes;
        std::vector<size_t> latent_v_block_offsets;
        std::vector<size_t> latent_v_block_nbytes;
        size_t latent_v_bytes_per_token = 0;
    };

    struct xq_k_block;

    struct xq_layer_payload {
        enum class storage_kind {
            none,
            quantized,
            floating,
            block_ref,
        };

        storage_kind kind = storage_kind::none;
        std::vector<uint8_t> data;
        std::vector<llama::xquant::block_qparams> qparams;
        std::vector<float> float_data;

        struct latent_k_payload {
            storage_kind kind = storage_kind::none;
            std::shared_ptr<xq_k_block> block;
            uint16_t offset = 0;
            std::vector<float> float_data;
        } latent_k;

        struct latent_v_payload {
            storage_kind kind = storage_kind::none;
            std::vector<uint8_t> data;
            std::vector<llama::xquant::block_qparams> qparams;
            std::vector<float> float_data;
        } latent_v;
    };

    struct xq_token_state {
        llama_pos pos = -1;
        std::vector<xq_layer_payload> layers;
        std::vector<std::vector<float>> hat_layers;
    };

    struct xq_sequence_state {
        struct pending_latent_k {
            std::vector<float> buffer;
            std::vector<xq_layer_payload *> payloads;
            size_t count = 0;
        };

        std::map<llama_pos, xq_token_state> tokens;
        std::vector<pending_latent_k> pending_k;
    };

    struct xq_seq_bounds {
        llama_pos pos_min = -1;
        llama_pos pos_max = -1;
    };

    std::vector<xq_layer_config> layer_cfgs;
    std::array<xq_sequence_state, LLAMA_MAX_SEQ> seq_states;
    std::array<xq_seq_bounds, LLAMA_MAX_SEQ> seq_bounds;
    size_t dim_model = 0;
    std::vector<float> zero_buffer;

    void init_layer_configs();
    void init_svd_layer(xq_layer_config & cfg, size_t il);
    xq_token_state & ensure_token_state(llama_seq_id seq_id, llama_pos pos);
    const std::vector<float> & get_hat_prev(const xq_token_state & token, int32_t il) const;
    std::vector<float> & get_hat_slot(xq_token_state & token, int32_t il);
    void update_seq_bounds(llama_seq_id seq_id);
    void store_layer_tokens(
            int32_t il,
            const float * data,
            size_t n_embd,
            size_t n_tokens,
            const llama_ubatch & ubatch,
            std::array<bool, LLAMA_MAX_SEQ> & touched);
    void compute_latents(const xq_layer_config & cfg,
            const float * token,
            std::vector<float> & out_k,
            std::vector<float> & out_v,
            std::vector<float> & out_kv) const;
    void dequantize_payload(const xq_layer_config & cfg, const xq_layer_payload & payload, std::vector<float> & out) const;
    void reconstruct_token(const xq_layer_config & cfg, const xq_token_state & token_state, const xq_layer_payload & payload, int32_t il, std::vector<float> & out) const;
    void append_latent_k(llama_seq_id seq_id, int32_t il, xq_layer_payload & payload, const std::vector<float> & latent_k);
    void finalize_latent_k_block(xq_sequence_state::pending_latent_k & pending, const xq_layer_config & cfg);
    void append_latent_v(xq_layer_payload & payload, const xq_layer_config & cfg, const std::vector<float> & latent_v);

    uint32_t layer_bits(int32_t il) const;

    void tensor_to_host(ggml_tensor * tensor, std::vector<float> & out) const;

    friend class llama_memory_context_xquant;
};

class llama_memory_xquant_cl : public llama_memory_xquant {
public:
    llama_memory_xquant_cl(const llama_model & model, const llama_cparams & cparams);
    ~llama_memory_xquant_cl() override = default;
};
