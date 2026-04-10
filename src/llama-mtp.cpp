#include "llama-mtp.h"

#include "llama-arch.h"
#include "llama-model.h"

#include <algorithm>
#include <limits>

static ggml_context_ptr llama_mtp_init_ctx(size_t n_tensors) {
    ggml_init_params params = {
        /*.mem_size   =*/ std::max<size_t>(1, n_tensors) * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    return ggml_context_ptr { ggml_init(params) };
}

static bool llm_arch_supports_native_mtp(const llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_QWEN35:
            return true;
        default:
            return false;
    }
}

bool llama_mtp_backend_seed_state::ready() const {
    return backend != nullptr &&
        buft != nullptr &&
        n_embd > 0 &&
        generation > 0 &&
        seed_cache_dev != nullptr &&
        seed_batch_dev != nullptr &&
        seed_cache_rows.size() == LLAMA_MAX_SEQ &&
        seed_batch_rows.size() == LLAMA_MAX_SEQ;
}

bool llama_mtp_backend_seed_state::matches(ggml_backend_t backend, ggml_backend_buffer_type_t buft, uint32_t n_embd) const {
    return this->backend == backend && this->buft == buft && this->n_embd == n_embd && ready();
}

void llama_mtp_backend_seed_state::clear_capture_views() {
    capture_ctxs.clear();
}

void llama_mtp_backend_seed_state::clear() {
    clear_capture_views();
    seed_cache_rows.clear();
    seed_batch_rows.clear();
    seed_cache_dev = nullptr;
    seed_batch_dev = nullptr;
    buf.reset();
    ctx_views.reset();
    ctx_roots.reset();
    backend = nullptr;
    buft = nullptr;
    n_embd = 0;
    generation = 0;
}

void llama_mtp_state::clear() {
    accepted.clear();
    draft.clear();
    seed_epoch = 1;
    seed_mode = LLAMA_MTP_SEED_MODE_NONE;
    std::fill(seed_epoch_by_seq.begin(), seed_epoch_by_seq.end(), 0);
    clear_backend_capture_views();
}

void llama_mtp_state::reserve(uint32_t n_embd, uint32_t n_pos_per_embd) {
    clear();

    this->n_embd = n_embd;
    this->n_pos_per_embd = n_pos_per_embd;

    if (!enabled()) {
        seed_embd.clear();
        clear_backend_seed_storage();
        return;
    }

    seed_embd.resize((size_t) n_embd * desc.n_draft);
    seed_by_seq.resize((size_t) LLAMA_MAX_SEQ * n_embd);
    seed_epoch_by_seq.assign(LLAMA_MAX_SEQ, 0);
    accepted.reserve(desc.n_draft);
    draft.reserve(desc.n_draft);

    ubatch_token.resize(LLAMA_MAX_SEQ);
    ubatch_pos.resize((size_t) LLAMA_MAX_SEQ * n_pos_per_embd);
    ubatch_n_seq_id.resize(LLAMA_MAX_SEQ);
    ubatch_seq_id.resize(LLAMA_MAX_SEQ);
    ubatch_seq_id_unq.resize(LLAMA_MAX_SEQ);
    ubatch_seq_idx.resize(LLAMA_MAX_SEQ, -1);
    ubatch_output.resize(LLAMA_MAX_SEQ);
    temp_seq_ids.resize(LLAMA_MAX_SEQ);
    temp_seq_used.resize(LLAMA_MAX_SEQ);
}

void llama_mtp_state::next_seed_epoch() {
    if (++seed_epoch == 0) {
        seed_epoch = 1;
        std::fill(seed_epoch_by_seq.begin(), seed_epoch_by_seq.end(), 0);
    }
    seed_mode = LLAMA_MTP_SEED_MODE_NONE;
}

void llama_mtp_state::set_seed_mode(llama_mtp_seed_mode mode) {
    seed_mode = mode;
}

bool llama_mtp_state::ensure_backend_seed_storage(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    if (!enabled() || backend == nullptr || buft == nullptr || n_embd == 0) {
        return false;
    }

    if (seed_backend.matches(backend, buft, n_embd)) {
        return true;
    }

    seed_backend.clear();

    seed_backend.backend = backend;
    seed_backend.buft = buft;
    seed_backend.n_embd = n_embd;
    seed_backend.generation = backend_seed_generation_next++;
    if (backend_seed_generation_next == 0) {
        backend_seed_generation_next = 1;
    }

    seed_backend.ctx_roots = llama_mtp_init_ctx(2);
    seed_backend.ctx_views = llama_mtp_init_ctx(2*LLAMA_MAX_SEQ);
    if (!seed_backend.ctx_roots || !seed_backend.ctx_views) {
        seed_backend.clear();
        return false;
    }

    seed_backend.seed_cache_dev = ggml_new_tensor_2d(seed_backend.ctx_roots.get(), GGML_TYPE_F32, n_embd, LLAMA_MAX_SEQ);
    seed_backend.seed_batch_dev = ggml_new_tensor_2d(seed_backend.ctx_roots.get(), GGML_TYPE_F32, n_embd, LLAMA_MAX_SEQ);
    if (!seed_backend.seed_cache_dev || !seed_backend.seed_batch_dev) {
        seed_backend.clear();
        return false;
    }

    ggml_set_name(seed_backend.seed_cache_dev, "mtp_seed_cache_dev");
    ggml_set_name(seed_backend.seed_batch_dev, "mtp_seed_batch_dev");

    seed_backend.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(seed_backend.ctx_roots.get(), buft));
    if (!seed_backend.buf) {
        seed_backend.clear();
        return false;
    }

    seed_backend.seed_cache_rows.resize(LLAMA_MAX_SEQ);
    seed_backend.seed_batch_rows.resize(LLAMA_MAX_SEQ);

    for (int32_t i = 0; i < LLAMA_MAX_SEQ; ++i) {
        const size_t offset_cache = (size_t) i * seed_backend.seed_cache_dev->nb[1];
        const size_t offset_batch = (size_t) i * seed_backend.seed_batch_dev->nb[1];

        auto * cache_row = ggml_view_1d(seed_backend.ctx_views.get(), seed_backend.seed_cache_dev, n_embd, offset_cache);
        auto * batch_row = ggml_view_1d(seed_backend.ctx_views.get(), seed_backend.seed_batch_dev, n_embd, offset_batch);
        if (!cache_row || !batch_row) {
            seed_backend.clear();
            return false;
        }

        ggml_set_name(cache_row, "mtp_seed_cache_row");
        ggml_set_name(batch_row, "mtp_seed_batch_row");

        if (ggml_backend_view_init(cache_row) != GGML_STATUS_SUCCESS ||
            ggml_backend_view_init(batch_row) != GGML_STATUS_SUCCESS) {
            seed_backend.clear();
            return false;
        }

        seed_backend.seed_cache_rows[i] = cache_row;
        seed_backend.seed_batch_rows[i] = batch_row;
    }

    return true;
}

void llama_mtp_state::clear_backend_seed_storage() {
    seed_backend.clear();
}

void llama_mtp_state::clear_backend_capture_views() {
    seed_backend.clear_capture_views();
}

float * llama_mtp_state::seed_row(llama_seq_id seq_id) {
    GGML_ASSERT(0 <= seq_id && seq_id < LLAMA_MAX_SEQ);
    return seed_by_seq.data() + (size_t) seq_id * n_embd;
}

const float * llama_mtp_state::seed_row(llama_seq_id seq_id) const {
    GGML_ASSERT(0 <= seq_id && seq_id < LLAMA_MAX_SEQ);
    return seed_by_seq.data() + (size_t) seq_id * n_embd;
}

bool llama_mtp_state::has_seed(llama_seq_id seq_id) const {
    return 0 <= seq_id && seq_id < LLAMA_MAX_SEQ && !seed_epoch_by_seq.empty() && seed_epoch_by_seq[seq_id] == seed_epoch;
}

void llama_mtp_state::mark_seed(llama_seq_id seq_id) {
    GGML_ASSERT(0 <= seq_id && seq_id < LLAMA_MAX_SEQ);
    seed_epoch_by_seq[seq_id] = seed_epoch;
}

llama_ubatch llama_mtp_state::ubatch_reserve(uint32_t n_seq) {
    GGML_ASSERT(n_seq <= LLAMA_MAX_SEQ);
    GGML_ASSERT(n_pos_per_embd > 0);

    std::fill(ubatch_seq_idx.begin(), ubatch_seq_idx.end(), -1);

    return llama_ubatch {
        /*.b_equal_seqs =*/ true,
        /*.n_tokens     =*/ n_seq,
        /*.n_seq_tokens =*/ 1,
        /*.n_seqs       =*/ n_seq,
        /*.n_seqs_unq   =*/ n_seq,
        /*.n_pos        =*/ n_pos_per_embd,
        /*.token        =*/ ubatch_token.data(),
        /*.embd         =*/ nullptr,
        /*.pos          =*/ ubatch_pos.data(),
        /*.n_seq_id     =*/ ubatch_n_seq_id.data(),
        /*.seq_id       =*/ ubatch_seq_id.data(),
        /*.seq_id_unq   =*/ ubatch_seq_id_unq.data(),
        /*.seq_idx      =*/ ubatch_seq_idx.data(),
        /*.output       =*/ ubatch_output.data(),
        /*.data         =*/ nullptr,
    };
}


llama_mtp_desc llama_mtp_init_desc(const llama_model & model) {
    llama_mtp_desc res = {
        /*.supported            =*/ false,
        /*.n_predict            =*/ model.hparams.nextn_predict_layers,
        // The current native runtime uses the verifier hidden state to draft one
        // continuation token per step. Multi-layer MTP heads may expose more
        // predictor layers in metadata, but recursive multi-token drafting is
        // not implemented here yet.
        /*.n_draft              =*/ model.hparams.nextn_predict_layers > 0 ? 1u : 0u,
        /*.dedicated_embeddings =*/ false,
    };

    if (!llm_arch_supports_native_mtp(model.arch) || res.n_predict == 0 || model.layers.empty()) {
        return res;
    }

    const uint32_t il_mtp = model.hparams.n_layer - res.n_predict;
    if (il_mtp >= model.layers.size()) {
        return res;
    }

    const auto & layer = model.layers[il_mtp];

    const bool has_nextn_inputs =
        layer.nextn.eh_proj != nullptr &&
        layer.nextn.enorm   != nullptr &&
        layer.nextn.hnorm   != nullptr;

    res.dedicated_embeddings = layer.nextn.embed_tokens != nullptr;

    const bool has_nextn_head =
        res.dedicated_embeddings ||
        layer.nextn.shared_head_head != nullptr ||
        model.output != nullptr;

    res.supported = has_nextn_inputs && has_nextn_head;

    return res;
}
