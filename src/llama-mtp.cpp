#include "llama-mtp.h"

#include "llama-arch.h"
#include "llama-model.h"

#include <algorithm>

static bool llm_arch_supports_native_mtp(const llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_QWEN35:
        case LLM_ARCH_QWEN35MOE:
            return true;
        default:
            return false;
    }
}

void llama_mtp_state::clear() {
    accepted.clear();
    draft.clear();
    seed_epoch = 1;
    std::fill(seed_epoch_by_seq.begin(), seed_epoch_by_seq.end(), 0);
}

void llama_mtp_state::reserve(uint32_t n_embd, uint32_t n_pos_per_embd) {
    clear();

    this->n_embd = n_embd;
    this->n_pos_per_embd = n_pos_per_embd;

    if (!enabled()) {
        seed_embd.clear();
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
        /*.n_draft              =*/ model.hparams.nextn_predict_layers,
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
