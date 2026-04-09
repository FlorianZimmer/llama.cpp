#include "llama-mtp.h"

#include "llama-arch.h"
#include "llama-model.h"

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
    seed_by_seq.clear();
}

void llama_mtp_state::reserve(uint32_t n_embd) {
    clear();

    if (!enabled()) {
        seed_embd.clear();
        return;
    }

    seed_embd.resize((size_t) n_embd * desc.n_draft);
    accepted.reserve(desc.n_draft);
    draft.reserve(desc.n_draft);
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
