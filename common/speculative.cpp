#include "speculative.h"

#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <map>
#include <memory>
#include <unordered_map>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

const std::vector<enum common_speculative_type> common_speculative_types = {
    COMMON_SPECULATIVE_TYPE_NONE,
    COMMON_SPECULATIVE_TYPE_DRAFT,
    COMMON_SPECULATIVE_TYPE_EAGLE3,
    COMMON_SPECULATIVE_TYPE_MTP,
    COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V,
    COMMON_SPECULATIVE_TYPE_NGRAM_MOD,
    COMMON_SPECULATIVE_TYPE_NGRAM_CACHE
};

const std::map<std::string, enum common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft",         COMMON_SPECULATIVE_TYPE_DRAFT},
    {"eagle3",        COMMON_SPECULATIVE_TYPE_EAGLE3},
    {"mtp",           COMMON_SPECULATIVE_TYPE_MTP},
    {"ngram_simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram_map_k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram_map_k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram_mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram_cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE}
};

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

static bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("%s: vocab_type tgt: %d\n", __func__, vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("%s: vocab_type dft: %d\n", __func__, vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_DBG("%s: draft model vocab type must match target model to use speculation but ", __func__);
        LOG_DBG("vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (
        llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft) ||
        llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft)
    ) {
        LOG_DBG("%s: draft model special tokens must match target model to use speculation\n", __func__);
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_DBG("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_DBG("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_DBG("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_DBG("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_state
struct common_speculative_state {
    const enum common_speculative_type type;

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.
    int64_t t_sync_us   = 0; // total time spent syncing accepted target outputs back into the proposer.
    int64_t t_sync_fetch_us  = 0; // total time spent fetching accepted target states.
    int64_t t_sync_decode_us = 0; // total time spent updating the proposer from accepted target states.

    common_speculative_state(enum common_speculative_type type) : type(type) {}

    virtual ~common_speculative_state() = default;

    virtual void begin(const llama_tokens & prompt) = 0;

    virtual void draft(
            const common_params_speculative & params,
            llama_context * ctx_tgt,
            llama_seq_id    seq_id_tgt,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) = 0;

    virtual void accept(uint16_t n_accepted) = 0;
    virtual void accept_tokens(llama_context * ctx_tgt, const llama_tokens & ids, const std::vector<int> & idxs) {
        GGML_UNUSED(ctx_tgt);
        GGML_UNUSED(ids);
        GGML_UNUSED(idxs);
    }

    virtual std::string extra_stats() const {
        return {};
    }
};

struct common_speculative_mtp_shared_state {
    uint64_t output_epoch = 0;
};

struct common_speculative_state_draft : public common_speculative_state {
    llama_context * ctx_tgt; // only used for retokenizing from ctx_dft
    llama_context * ctx_dft;

    common_sampler * smpl;

    llama_batch  batch;
    llama_tokens prompt_dft;

    bool vocab_cmpt = true; // whether retokenization is needed
    std::unordered_map<std::string, std::string> vocab_map;

    common_speculative_state_draft(
            enum common_speculative_type type,
            llama_context * ctx_tgt,
            llama_context * ctx_dft,
            const std::vector<std::pair<std::string, std::string>> & replacements)
        : common_speculative_state(type)
        , ctx_tgt(ctx_tgt)
        , ctx_dft(ctx_dft)
    {
        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);
        smpl = nullptr;

        // TODO: optimize or pass from outside?
        // {
        //     common_params_sampling params;
        //     params.no_perf = false;
        //
        //     params.top_k = 40;
        //     params.top_p = 0.9;
        //
        //     params.samplers = {
        //         COMMON_SAMPLER_TYPE_TOP_K,
        //         COMMON_SAMPLER_TYPE_TOP_P,
        //         COMMON_SAMPLER_TYPE_INFILL,
        //     };
        //
        //     result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        // }
        {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers = {
                COMMON_SAMPLER_TYPE_TOP_K,
            };

            smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        }

        vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        LOG_DBG("vocab_cmpt = %d\n", vocab_cmpt);

        if (!vocab_cmpt) {
            LOG_WRN("the target and draft vocabs are not compatible - tokens will be translated between the two\n");

            for (const auto & pair : replacements) {
                vocab_map[pair.first] = pair.second;
            }
        }
    }

    ~common_speculative_state_draft() override {
        llama_perf_context_print(ctx_dft);

        llama_free(ctx_dft);

        common_sampler_free(smpl);

        llama_batch_free(batch);
    }

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            llama_context * ctx_tgt,
            llama_seq_id    seq_id_tgt,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        GGML_UNUSED(ctx_tgt);
        GGML_UNUSED(seq_id_tgt);
        auto * spec = this;

        auto & batch       = spec->batch;
        auto & ctx_tgt_ref = spec->ctx_tgt;
        auto & ctx_dft     = spec->ctx_dft;
        auto & smpl        = spec->smpl;
        auto & prompt_dft  = spec->prompt_dft;

        auto * mem_dft = llama_get_memory(ctx_dft);

        int reuse_i = 0;
        int reuse_n = 0;

        const int n_ctx = llama_n_ctx(ctx_dft) - params.n_max;

        llama_tokens prompt_cnv;
        if (!spec->vocab_cmpt) {
            std::string text;

            text = common_detokenize(ctx_tgt_ref, prompt_tgt, true);
            text = replace_to_dft(text);

            LOG_DBG("%s: main->draft detokenized string: '%s'\n", __func__, text.c_str());

            prompt_cnv = common_tokenize(ctx_dft, text, false, true);

            // convert id_last to draft vocab. llama_detokenize is called directly to avoid an allocation
            const auto * model_tgt = llama_get_model(ctx_tgt_ref);
            const auto * vocab_tgt = llama_model_get_vocab(model_tgt);

            int32_t n_chars = llama_detokenize(vocab_tgt, &id_last, 1, nullptr, 0, false, false);
            GGML_ASSERT(n_chars < 0 && "failed to detokenize id_last");

            text.resize(-n_chars);
            llama_detokenize(vocab_tgt, &id_last, 1, text.data(), text.size(), false, false);
            text = replace_to_dft(text);

            LOG_DBG("main->draft detokenized id_last(%d): '%s'\n", id_last, text.c_str());
            id_last = common_tokenize(ctx_dft, text, false, true)[0];
        }

        const llama_tokens & prompt_cur = spec->vocab_cmpt ? prompt_tgt : prompt_cnv;

        const int i_start = std::max<int>(0, (int) prompt_cur.size() - n_ctx);

        // reuse as much as possible from the old draft context
        // ideally, the draft context should be as big as the target context and we will always reuse the entire prompt
        for (int i = 0; i < (int) prompt_dft.size(); ++i) {
            int cur = 0;
            while (i_start + cur < (int) prompt_cur.size() &&
                    i       + cur < (int) prompt_dft.size() &&
                    prompt_cur[i_start + cur] == prompt_dft[i + cur]) {
                cur++;
            }

            if ((cur >= 256 || n_ctx >= (int) prompt_cur.size()) && cur > reuse_n) {
                reuse_i = i;
                reuse_n = cur;
            }
        }

        LOG_DBG("%s: reuse_i = %d, reuse_n = %d, prompt = %d\n", __func__, reuse_i, reuse_n, (int) prompt_dft.size());

        result.clear();
        result.reserve(params.n_max);

        if (reuse_n == 0) {
            llama_memory_clear(mem_dft, false);
            prompt_dft.clear();
        } else {
            // this happens when a previous draft has been discarded (for example, due to being too small), but the
            // target model agreed with it. in this case, we simply pass back the previous results to save compute
            if (reuse_i + reuse_n < (int) prompt_dft.size() && prompt_dft[reuse_i + reuse_n] == id_last) {
                for (int i = reuse_i + reuse_n + 1; i < (int) prompt_dft.size(); ++i) {
                    result.push_back(prompt_dft[i]);

                    if (params.n_max <= (int) result.size()) {
                        break;
                    }
                }

                return;
            }

            if (reuse_i > 0) {
                llama_memory_seq_rm (mem_dft, 0, 0, reuse_i);
                llama_memory_seq_add(mem_dft, 0, reuse_i, -1, -reuse_i);

                prompt_dft.erase(prompt_dft.begin(), prompt_dft.begin() + reuse_i);
            }

            if (reuse_n < (int) prompt_dft.size()) {
                llama_memory_seq_rm (mem_dft, 0, reuse_n, -1);
                prompt_dft.erase(prompt_dft.begin() + reuse_n, prompt_dft.end());
            }
        }

        // prepare a batch to evaluate any new tokens in the prompt
        common_batch_clear(batch);

        for (size_t i = i_start + reuse_n; i < prompt_cur.size(); ++i) {
            //LOG_DBG("i = %d, i_start = %d, reuse_n = %d, i - i_start = %d, id = %6d\n", i, i_start, reuse_n, i - i_start, prompt_cur[i]);
            common_batch_add(batch, prompt_cur[i], i - i_start, { 0 }, false);

            prompt_dft.push_back(prompt_cur[i]);
        }

        // we should rarely end-up here during normal decoding
        if (batch.n_tokens > 0) {
            //LOG_DBG("%s: draft prompt batch: %s\n", __func__, string_from(ctx, batch).c_str());

            llama_decode(ctx_dft, batch);
        }

        const llama_pos n_past = prompt_dft.size();

        LOG_DBG("%s: n_past = %d\n", __func__, n_past);

        common_batch_clear(batch);
        common_batch_add  (batch, id_last, n_past, { 0 }, true);

        prompt_dft.push_back(id_last);

        LOG_DBG("%s: draft prompt: %s\n", __func__, string_from(ctx_dft, prompt_dft).c_str());

        llama_decode(ctx_dft, batch);

        common_sampler_reset(smpl);

        // sample n_draft tokens from the draft model
        for (int i = 0; i < params.n_max; ++i) {
            common_batch_clear(batch);

            common_sampler_sample(smpl, ctx_dft, 0, true);

            const auto * cur_p = common_sampler_get_candidates(smpl, true);

            for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                LOG_DBG(" - draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                        k, i, cur_p->data[k].id, cur_p->data[k].p, common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
            }

            // add drafted token for each sequence
            const llama_token id = cur_p->data[0].id;

            common_sampler_accept(smpl, id, true);

            result.push_back(id);

            if (params.n_max <= (int) result.size()) {
                break;
            }

            // only collect very high-confidence draft tokens
            if (cur_p->data[0].p < params.p_min) {
                break;
            }

            common_batch_add(batch, id, n_past + i + 1, { 0 }, true);

            // evaluate the drafted tokens on the draft model
            llama_decode(ctx_dft, batch);

            prompt_dft.push_back(id);
        }

        if (!spec->vocab_cmpt) {
            std::string detokenized = common_detokenize(ctx_dft, result, true);
            detokenized = replace_to_tgt(detokenized);
            LOG_DBG("draft->main detokenized string: '%s'\n", detokenized.c_str());
            result = common_tokenize(ctx_tgt, detokenized, false, true);
            if (result.size() > (size_t)params.n_max) {
                result.resize(params.n_max);
            }
        }
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }

    std::string replace_to_dft(const std::string & input) const {
        std::string result = input;

        for (const auto & pair : this->vocab_map) {
            size_t pos = result.find(pair.first);
            while (pos != std::string::npos) {
                result.replace(pos, pair.first.length(), pair.second);
                pos = result.find(pair.first, pos + pair.second.length());
            }
        }

        return result;
    }

    std::string replace_to_tgt(const std::string & input) const {
        std::string result = input;

        for (const auto & pair : this->vocab_map) {
            size_t pos = result.find(pair.second);
            while (pos != std::string::npos) {
                result.replace(pos, pair.second.length(), pair.first);
                pos = result.find(pair.second, pos + pair.first.length());
            }
        }

        return result;
    }
};

struct common_speculative_state_eagle3 : public common_speculative_state {
    common_speculative_state_eagle3(enum common_speculative_type type) : common_speculative_state(type) {}

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            llama_context * ctx_tgt,
            llama_seq_id    seq_id_tgt,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & draft_tokens) override {
        // TODO: implement
        GGML_UNUSED(params);
        GGML_UNUSED(ctx_tgt);
        GGML_UNUSED(seq_id_tgt);
        GGML_UNUSED(prompt_tgt);
        GGML_UNUSED(id_last);
        GGML_UNUSED(draft_tokens);
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }
};

struct common_speculative_state_mtp : public common_speculative_state {
    llama_context * ctx_dft;
    llama_seq_id    seq_id;
    bool            owns_ctx_dft;
    bool            mtp_only_ctx;
    std::shared_ptr<common_speculative_mtp_shared_state> shared_state;
    common_sampler * smpl;
    llama_sampler  * smpl_backend;
    llama_batch batch;
    llama_tokens prompt_dft;
    std::vector<float> hidden_batch;
    std::vector<uint8_t> seq_state;
    size_t prompt_dft_sync_size = 0;
    bool is_synced = false;
    bool have_ready_logits = false;
    int32_t ready_output_idx = -1;
    uint64_t ready_epoch = 0;
    size_t n_ready_fast = 0;
    size_t n_ready_miss_no_logits = 0;
    size_t n_ready_miss_stale = 0;
    size_t n_ready_miss_size = 0;
    size_t n_ready_miss_last = 0;
    size_t n_ready_miss_prefix = 0;
    llama_perf_context_data perf_begin = {};

    common_speculative_state_mtp(
            enum common_speculative_type type,
            llama_context * ctx_dft,
            llama_seq_id    seq_id,
            bool            owns_ctx_dft,
            bool            mtp_only_ctx,
            std::shared_ptr<common_speculative_mtp_shared_state> shared_state)
        : common_speculative_state(type)
        , ctx_dft(ctx_dft)
        , seq_id(seq_id)
        , owns_ctx_dft(owns_ctx_dft)
        , mtp_only_ctx(mtp_only_ctx) {
        this->shared_state = std::move(shared_state);
        common_params_sampling params_cpu;
        params_cpu.no_perf = false;
        params_cpu.top_k = 1;
        params_cpu.samplers = {
            COMMON_SAMPLER_TYPE_TOP_K,
        };

        smpl = common_sampler_init(llama_get_model(ctx_dft), params_cpu);

        auto params_backend = llama_sampler_chain_default_params();
        params_backend.no_perf = false;

        smpl_backend = llama_sampler_chain_init(params_backend);
        llama_sampler_chain_add(smpl_backend, llama_sampler_init_greedy());
        llama_set_sampler(ctx_dft, seq_id, smpl_backend);

        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);
    }

    ~common_speculative_state_mtp() override {
        llama_set_sampler(ctx_dft, seq_id, nullptr);
        if (owns_ctx_dft) {
            llama_perf_context_print(ctx_dft);
            llama_free(ctx_dft);
        }
        common_sampler_free(smpl);
        llama_sampler_free(smpl_backend);
        llama_batch_free(batch);
    }

    void begin(const llama_tokens & prompt) override {
        auto * mem_dft = llama_get_memory(ctx_dft);
        if (owns_ctx_dft && seq_id == 0) {
            llama_memory_clear(mem_dft, false);
        } else {
            llama_memory_seq_rm(mem_dft, seq_id, -1, -1);
        }

        prompt_dft = mtp_only_ctx ? prompt : llama_tokens{};
        prompt_dft_sync_size = 0;
        is_synced = false;
        have_ready_logits = false;
        ready_output_idx = -1;
        ready_epoch = 0;
        perf_begin = llama_perf_context(ctx_dft);
    }

    void draft(
            const common_params_speculative & params,
            llama_context * ctx_tgt,
            llama_seq_id    seq_id_tgt,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        result.clear();
        result.reserve(1);

        bool is_ready = have_ready_logits;
        if (!have_ready_logits) {
            n_ready_miss_no_logits++;
        } else if (ready_epoch != shared_state->output_epoch) {
            n_ready_miss_stale++;
            is_ready = false;
        } else if (prompt_dft.size() != prompt_tgt.size() + 1) {
            n_ready_miss_size++;
            is_ready = false;
        } else if (prompt_dft.back() != id_last) {
            n_ready_miss_last++;
            is_ready = false;
        } else if (!std::equal(prompt_tgt.begin(), prompt_tgt.end(), prompt_dft.begin())) {
            n_ready_miss_prefix++;
            is_ready = false;
        } else {
            n_ready_fast++;
        }

        if (!is_ready) {
            if (mtp_only_ctx) {
                // Reduced MTP sidecars only carry appended NextN state, so they cannot
                // replay the full base stack to bootstrap proposals. Wait until the
                // verifier feeds us accepted hidden states through accept_tokens().
                return;
            }

            auto * mem_dft = llama_get_memory(ctx_dft);
            bool bootstrapped = false;

            if (!mtp_only_ctx && ctx_tgt != nullptr && prompt_dft.empty() && !prompt_tgt.empty()) {
                const size_t seq_state_size = llama_state_seq_get_size_ext(ctx_tgt, seq_id_tgt, 0);
                if (seq_state_size > 0) {
                    seq_state.resize(seq_state_size);

                    const size_t n_state = llama_state_seq_get_data_ext(ctx_tgt, seq_state.data(), seq_state.size(), seq_id_tgt, 0);
                    if (n_state == seq_state.size()) {
                        if (owns_ctx_dft && seq_id == 0) {
                            llama_memory_clear(mem_dft, false);
                        } else {
                            llama_memory_seq_rm(mem_dft, seq_id, -1, -1);
                        }

                        const size_t n_restore = llama_state_seq_set_data_ext(ctx_dft, seq_state.data(), seq_state.size(), seq_id, 0);
                        if (n_restore == seq_state.size()) {
                            prompt_dft = prompt_tgt;
                            bootstrapped = true;
                        }
                    }
                }
            }

            if (!bootstrapped) {
                int reuse_i = 0;
                int reuse_n = 0;

                const int n_ctx = llama_n_ctx(ctx_dft) - params.n_max;
                const int i_start = std::max<int>(0, (int) prompt_tgt.size() - n_ctx);

                for (int i = 0; i < (int) prompt_dft.size(); ++i) {
                    int cur = 0;
                    while (i_start + cur < (int) prompt_tgt.size() &&
                            i       + cur < (int) prompt_dft.size() &&
                            prompt_tgt[i_start + cur] == prompt_dft[i + cur]) {
                        cur++;
                    }

                    if ((cur >= 256 || n_ctx >= (int) prompt_tgt.size()) && cur > reuse_n) {
                        reuse_i = i;
                        reuse_n = cur;
                    }
                }

                is_synced = false;
                have_ready_logits = false;

                if (reuse_n == 0) {
                    if (owns_ctx_dft && seq_id == 0) {
                        llama_memory_clear(mem_dft, false);
                    } else {
                        llama_memory_seq_rm(mem_dft, seq_id, -1, -1);
                    }
                    prompt_dft.clear();
                } else {
                    if (reuse_i > 0) {
                        llama_memory_seq_rm (mem_dft, seq_id, 0, reuse_i);
                        llama_memory_seq_add(mem_dft, seq_id, reuse_i, -1, -reuse_i);
                        prompt_dft.erase(prompt_dft.begin(), prompt_dft.begin() + reuse_i);
                    }

                    if (reuse_n < (int) prompt_dft.size()) {
                        llama_memory_seq_rm(mem_dft, seq_id, reuse_n, -1);
                        prompt_dft.erase(prompt_dft.begin() + reuse_n, prompt_dft.end());
                    }
                }

                common_batch_clear(batch);

                for (size_t i = i_start + reuse_n; i < prompt_tgt.size(); ++i) {
                    common_batch_add(batch, prompt_tgt[i], i - i_start, { seq_id }, false);
                    prompt_dft.push_back(prompt_tgt[i]);
                }

                if (batch.n_tokens > 0 && llama_decode(ctx_dft, batch) != 0) {
                    return;
                }
            }

            const llama_pos pos_next = llama_memory_seq_pos_max(mem_dft, seq_id) + 1;

            common_batch_clear(batch);
            common_batch_add(batch, id_last, pos_next, { seq_id }, true);
            prompt_dft.push_back(id_last);

            if (llama_decode_mtp(ctx_dft, batch) != 0) {
                prompt_dft.pop_back();
                return;
            }

            prompt_dft_sync_size = prompt_dft.size();
            is_synced = true;
            have_ready_logits = true;
            ready_output_idx = batch.n_tokens - 1;
            ready_epoch = ++shared_state->output_epoch;
        }

        const llama_token backend_token = llama_get_sampled_token_ith(ctx_dft, ready_output_idx);
        if (backend_token != LLAMA_TOKEN_NULL) {
            if (params.p_min <= 1.0f) {
                result.push_back(backend_token);
            }
            return;
        }

        common_sampler_reset(smpl);

        const llama_token token = common_sampler_sample(smpl, ctx_dft, ready_output_idx, true);
        const auto * cur_p = common_sampler_get_candidates(smpl, true);

        if (cur_p->size > 0 && cur_p->data[0].p >= params.p_min) {
            result.push_back(token);
        }
    }

    void accept(uint16_t n_accepted) override {
        GGML_UNUSED(n_accepted);
    }

    void accept_tokens(llama_context * ctx_tgt, const llama_tokens & ids, const std::vector<int> & idxs) override {
        GGML_ASSERT(ids.size() == idxs.size());

        if (!is_synced && !mtp_only_ctx) {
            return;
        }

        if (is_synced && prompt_dft.size() > prompt_dft_sync_size) {
            auto * mem_dft = llama_get_memory(ctx_dft);
            llama_memory_seq_rm(mem_dft, seq_id, prompt_dft_sync_size, -1);
            prompt_dft.resize(prompt_dft_sync_size);
        }

        const size_t n_embd = llama_model_n_embd_out(llama_get_model(ctx_dft));
        hidden_batch.resize(ids.size()*n_embd);
        {
            common_time_meas tm(t_sync_fetch_us, !gen_perf);
            if (!llama_get_mtp_hiddens(ctx_tgt, idxs.data(), idxs.size(), hidden_batch.data())) {
                is_synced = false;
                have_ready_logits = false;
                ready_output_idx = -1;
                ready_epoch = 0;
                hidden_batch.clear();
                return;
            }
        }

        llama_set_mtp_input_hiddens(ctx_dft, hidden_batch.data(), ids.size(), n_embd);

        auto * mem_dft = llama_get_memory(ctx_dft);
        llama_pos pos_next = mtp_only_ctx ? (llama_pos) prompt_dft.size() : (llama_memory_seq_pos_max(mem_dft, seq_id) + 1);
        common_batch_clear(batch);
        for (size_t i = 0; i < ids.size(); ++i) {
            const bool output = (i + 1 == ids.size());
            common_batch_add(batch, ids[i], pos_next + i, { seq_id }, output);
        }

        {
            common_time_meas tm(t_sync_decode_us, !gen_perf);
            if (llama_decode_mtp(ctx_dft, batch) != 0) {
                is_synced = false;
                have_ready_logits = false;
                ready_output_idx = -1;
                ready_epoch = 0;
                hidden_batch.clear();
                return;
            }
        }

        prompt_dft.insert(prompt_dft.end(), ids.begin(), ids.end());
        prompt_dft_sync_size = prompt_dft.size();
        hidden_batch.clear();
        is_synced = true;
        have_ready_logits = true;
        ready_output_idx = batch.n_tokens - 1;
        ready_epoch = ++shared_state->output_epoch;
    }

    std::string extra_stats() const override {
        const auto perf_end = llama_perf_context(ctx_dft);

        return string_format(", ready(fast,no_logits,stale,size,last,prefix) = %zu %zu %zu %zu %zu %zu"
                             ", ctx(eval,res,prep,out,gb,ga,in,cmp,reuse,builds) = %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %d %d",
                n_ready_fast,
                n_ready_miss_no_logits,
                n_ready_miss_stale,
                n_ready_miss_size,
                n_ready_miss_last,
                n_ready_miss_prefix,
                perf_end.t_mtp_eval_ms        - perf_begin.t_mtp_eval_ms,
                perf_end.t_mtp_reserve_ms     - perf_begin.t_mtp_reserve_ms,
                perf_end.t_mtp_prepare_ms     - perf_begin.t_mtp_prepare_ms,
                perf_end.t_mtp_output_ms      - perf_begin.t_mtp_output_ms,
                perf_end.t_mtp_graph_build_ms - perf_begin.t_mtp_graph_build_ms,
                perf_end.t_mtp_graph_alloc_ms - perf_begin.t_mtp_graph_alloc_ms,
                perf_end.t_mtp_set_inputs_ms  - perf_begin.t_mtp_set_inputs_ms,
                perf_end.t_mtp_compute_ms     - perf_begin.t_mtp_compute_ms,
                perf_end.n_mtp_reused         - perf_begin.n_mtp_reused,
                perf_end.n_mtp_graph_builds   - perf_begin.n_mtp_graph_builds);
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_state_ngram_simple : public common_speculative_state {
    common_ngram_simple_config config;

    common_speculative_state_ngram_simple(
            enum common_speculative_type type,
            common_ngram_simple_config config)
        : common_speculative_state(type), config(config) {}

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            llama_context * ctx_tgt,
            llama_seq_id    seq_id_tgt,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        GGML_UNUSED(ctx_tgt);
        GGML_UNUSED(seq_id_tgt);
        result = common_ngram_simple_draft(config, prompt_tgt, id_last);
        GGML_UNUSED(params);
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }
};

struct common_speculative_state_ngram_map_k : public common_speculative_state {
    // draft ngram map for speculative decoding without draft model
    common_ngram_map map;

    common_speculative_state_ngram_map_k(
            enum common_speculative_type type,
            common_ngram_map map)
        : common_speculative_state(type), map(std::move(map)) {}

    void begin(const llama_tokens & prompt) override {
        common_ngram_map_begin(map, prompt);
    }

    void draft(
            const common_params_speculative & params,
            llama_context * ctx_tgt,
            llama_seq_id    seq_id_tgt,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        GGML_UNUSED(ctx_tgt);
        GGML_UNUSED(seq_id_tgt);
        common_ngram_map_draft(map, prompt_tgt, id_last, result);
        GGML_UNUSED(params);
    }

    void accept(uint16_t n_accepted) override {
        common_ngram_map_accept(map, n_accepted);
    }
};

struct common_speculative_state_ngram_mod : public common_speculative_state {
    common_ngram_mod & mod;

    // the last position in the prompt that was added to the ngram container
    size_t i_last = 0;

    // length of the last drafted n‑gram (number of tokens returned by draft)
    size_t n_draft_last = 0;

    // consecutive accept rounds with low acceptance fraction (< 0.5)
    int n_low = 0;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    common_speculative_state_ngram_mod(enum common_speculative_type type, common_ngram_mod & mod)
        : common_speculative_state(type), mod(mod), verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));
    }

    void begin(const llama_tokens & prompt) override {
        i_last = 0;

        n_draft_last = 0;

        const size_t n = mod.get_n();

        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        LOG_INF("%s: ngram_mod occupancy = %zu/%zu (%.2f)\n", __func__, mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            LOG_WRN("%s: ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", __func__, f, f_thold);

            mod.reset();
        }
    }

    void draft(
            const common_params_speculative & params,
            llama_context * ctx_tgt,
            llama_seq_id    seq_id_tgt,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        GGML_UNUSED(params);
        GGML_UNUSED(ctx_tgt);
        GGML_UNUSED(seq_id_tgt);

        n_draft_last = 0;

        const size_t cur_len = prompt_tgt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (i_last + 32 < cur_len) {
            for (size_t i = i_last; i < cur_len - n; ++i) {
                mod.add(prompt_tgt.data() + i);
            }

            i_last = cur_len - n;
        }

        result.resize(n + params.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt_tgt[cur_len - n + 1 + i];
        }
        result[n - 1] = id_last;

        for (int i = 0; i < params.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < params.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n‑gram for later acceptance analysis
        n_draft_last = result.size();
    }

    void accept(uint16_t n_accepted) override {
        if (verbose) {
            LOG_INF("%s: accepted %d tokens from %zu drafted tokens\n", __func__, n_accepted, n_draft_last);
        }

        // compute acceptance fraction if we have a recorded draft length
        if (n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)n_draft_last;
            if (f_acc < 0.5) {
                n_low++;
                if (n_low >= 3) {
                    LOG_WRN("%s: low acceptance streak (%d) – resetting ngram_mod\n", __func__, n_low);

                    mod.reset();
                    n_low = 0;
                }
            } else {
                n_low = 0;
            }
        }
    }
};

struct common_speculative_state_ngram_cache : public common_speculative_state {
    uint16_t n_draft;
    bool save_dynamic;
    bool save_static;

    common_ngram_cache ngram_cache_context;
    common_ngram_cache ngram_cache_dynamic;
    common_ngram_cache ngram_cache_static;

    size_t cache_size = 0; // number of tokens in n-gram cache

    common_speculative_state_ngram_cache(
            const enum common_speculative_type type,
            const std::string & path_static,
            const std::string & path_dynamic,
            uint16_t            n_draft,
            bool                save_dynamic,
            bool                save_static)
        : common_speculative_state(type)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        if (!path_static.empty()) {
            try {
                ngram_cache_static = common_ngram_cache_load(path_static);
            } catch (...) {
                LOG_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);
            } catch (...) {
                LOG_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            llama_context * ctx_tgt,
            llama_seq_id    seq_id_tgt,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        GGML_UNUSED(params);
        GGML_UNUSED(ctx_tgt);
        GGML_UNUSED(seq_id_tgt);

        if (cache_size < prompt_tgt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt_tgt.size() + 1 - cache_size);
            for (size_t j = cache_size; j < prompt_tgt.size(); ++j) {
                tokens_new.push_back(prompt_tgt[j]);
            }
            tokens_new.push_back(id_last); // add the last token

            // Update context ngram cache with new prompt_tgt:
            common_ngram_cache_update(ngram_cache_context, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            cache_size = prompt_tgt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt_tgt.size() + 1);
        for (size_t j = 0; j < prompt_tgt.size(); ++j) {
            inp.push_back(prompt_tgt[j]);
        }
        inp.push_back(id_last);

        result.push_back(id_last);

        common_ngram_cache_draft(inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                ngram_cache_context,
                ngram_cache_dynamic,
                ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    void accept(uint16_t n_accepted) override {
        // TODO: noop
        GGML_UNUSED(n_accepted);
    }
};

struct common_speculative {
    std::vector<std::unique_ptr<common_speculative_state>> impls; // list of implementations to use and their states
    common_speculative_state * curr_impl = nullptr; // current implementation in use (for stats)
};

static llama_context * common_speculative_create_mtp_context(
        const common_params_speculative & params,
        llama_context                   * ctx_tgt,
        uint32_t                          n_seq_max) {
    const llama_model * model_tgt = llama_get_model(ctx_tgt);
    if (!llama_model_supports_mtp(model_tgt)) {
        LOG_WRN("%s: native MTP requested, but the target checkpoint does not expose MTP tensors\n", __func__);
        return nullptr;
    }

    llama_set_mtp_output(ctx_tgt, true);

    auto cparams_mtp = params.cparams_dft;
    const uint32_t mtp_depth = std::max<uint32_t>(1, (uint32_t) llama_model_mtp_depth_max(model_tgt));
    const uint32_t n_batch_mtp = std::max<uint32_t>(2, std::min<uint32_t>(std::max<int32_t>(1, params.n_max), mtp_depth) + 1);
    const uint32_t n_batch_total = n_batch_mtp*std::max<uint32_t>(1, n_seq_max);

    cparams_mtp.n_ctx           = params.n_ctx > 0 ? params.n_ctx : llama_n_ctx(ctx_tgt);
    cparams_mtp.n_batch         = n_batch_total;
    cparams_mtp.n_ubatch        = n_batch_total;
    cparams_mtp.n_seq_max       = n_seq_max;
    cparams_mtp.n_threads       = params.cpuparams.n_threads > 0 ? params.cpuparams.n_threads : llama_n_threads(ctx_tgt);
    cparams_mtp.n_threads_batch = params.cpuparams_batch.n_threads > 0
        ? params.cpuparams_batch.n_threads
        : llama_n_threads_batch(ctx_tgt);
    cparams_mtp.type_k          = params.cache_type_k;
    cparams_mtp.type_v          = params.cache_type_v;
    cparams_mtp.no_perf         = params.cparams_dft.no_perf;
    cparams_mtp.mtp_only        = true;

    llama_context * ctx_mtp = llama_init_from_model(const_cast<llama_model *>(model_tgt), cparams_mtp);
    if (ctx_mtp == nullptr) {
        LOG_ERR("%s", "failed to create MTP draft context\n");
        return nullptr;
    }

    // The current native MTP runtime only consumes logits from the sidecar context.
    // Exporting hidden states here adds an extra device-to-host copy on GPU backends
    // without enabling additional draft depth.
    llama_set_mtp_output(ctx_mtp, false);

    return ctx_mtp;
}

static common_speculative_state_mtp * common_speculative_get_mtp_state(common_speculative * spec) {
    if (spec == nullptr) {
        return nullptr;
    }

    for (auto & impl : spec->impls) {
        if (impl->type == COMMON_SPECULATIVE_TYPE_MTP) {
            return static_cast<common_speculative_state_mtp *>(impl.get());
        }
    }

    return nullptr;
}

static std::shared_ptr<common_speculative_mtp_shared_state> common_speculative_get_mtp_shared_state(llama_context * ctx_mtp) {
    static std::unordered_map<llama_context *, std::weak_ptr<common_speculative_mtp_shared_state>> states;

    auto & weak = states[ctx_mtp];
    auto shared = weak.lock();
    if (!shared) {
        shared = std::make_shared<common_speculative_mtp_shared_state>();
        weak = shared;
    }

    return shared;
}

static common_ngram_map get_common_ngram_map(const common_speculative_config & config) {
    uint16_t size_key   = config.params.ngram_size_n;
    uint16_t size_value = config.params.ngram_size_m;
    bool     key_only   = (config.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
    uint16_t min_hits   = config.params.ngram_min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_state_ngram_cache create_state_ngram_cache(
        const std::string & path_static, const std::string & path_dynamic,
        const common_speculative_config & config) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_state_ngram_cache state(config.type, path_static, path_dynamic, n_draft, save_static, save_dynamic);

    return state;
}

std::string common_speculative_type_name_str() {
    std::string result;
    for (size_t i = 0; i < common_speculative_types.size(); i++) {
        if (i > 0) {
            result += ", ";
        }
        result += common_speculative_type_to_str(common_speculative_types[i]);
    }
    return result;
}

std::string common_speculative_type_to_str(enum common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT:         return "draft";
        case COMMON_SPECULATIVE_TYPE_EAGLE3:        return "eagle3";
        case COMMON_SPECULATIVE_TYPE_MTP:           return "mtp";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram_simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram_map_k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram_map_k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram_mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram_cache";
        default:                                    return "unknown";
    }
}

enum common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const auto it = common_speculative_type_from_name_map.find(name);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

bool common_speculative_is_compat(llama_context * ctx_tgt) {
    auto * mem = llama_get_memory(ctx_tgt);
    if (mem == nullptr) {
        return false;
    }

    bool res = true;

    llama_memory_clear(mem, true);

    // eval 2 tokens to check if the context is compatible
    std::vector<llama_token> tmp;
    tmp.push_back(0);
    tmp.push_back(0);

    int ret = llama_decode(ctx_tgt, llama_batch_get_one(tmp.data(), tmp.size()));
    if (ret != 0) {
        LOG_ERR("%s: llama_decode() failed: %d\n", __func__, ret);
        res = false;
        goto done;
    }

    if (!llama_memory_can_seq_rm_partial(mem)) {
        const size_t seq_state_size = llama_state_seq_get_size(ctx_tgt, 0);
        std::vector<uint8_t> seq_state(seq_state_size);

        if (seq_state_size > 0) {
            const size_t n_state = llama_state_seq_get_data(ctx_tgt, seq_state.data(), seq_state.size(), 0);
            if (n_state != seq_state.size()) {
                LOG_WRN("%s: failed to snapshot target sequence state (%zu != %zu)\n", __func__, n_state, seq_state.size());
                res = false;
                goto done;
            }

            llama_memory_clear(mem, true);

            const size_t n_restore = llama_state_seq_set_data(ctx_tgt, seq_state.data(), seq_state.size(), 0);
            if (n_restore != seq_state.size()) {
                LOG_WRN("%s: failed to restore target sequence state (%zu != %zu)\n", __func__, n_restore, seq_state.size());
                res = false;
                goto done;
            }
        }
    }

done:
    llama_memory_clear(mem, true);
    llama_synchronize(ctx_tgt);

    return res;
}

common_speculative_verifier::common_speculative_verifier(llama_context * ctx_tgt, llama_seq_id seq_id)
    : ctx_tgt(ctx_tgt)
    , mem_tgt(llama_get_memory(ctx_tgt))
    , seq_id(seq_id)
    , can_seq_rm_partial(mem_tgt && llama_memory_can_seq_rm_partial(mem_tgt))
    , use_full_state(false)
    , batch_replay(llama_batch_init(std::max<int32_t>(1, (int32_t) llama_n_batch(ctx_tgt)), 0, 1)) {
}

common_speculative_verifier::~common_speculative_verifier() {
    llama_batch_free(batch_replay);
}

void common_speculative_verifier::set_seq_id(llama_seq_id seq_id) {
    this->seq_id = seq_id;
}

bool common_speculative_verifier::uses_full_state() const {
    return use_full_state;
}

bool common_speculative_verifier::restore_snapshot() {
    if (!use_full_state) {
        return false;
    }

    llama_memory_clear(mem_tgt, true);

    if (full_state_size == 0) {
        return true;
    }

    const size_t n_state = llama_state_set_data(ctx_tgt, full_state.data(), full_state_size);
    if (n_state != full_state_size) {
        LOG_ERR("%s: failed to restore full target state (%zu != %zu)\n", __func__, n_state, full_state_size);
        return false;
    }

    return true;
}

int32_t common_speculative_verifier::n_past_after(const llama_tokens & ids) const {
    GGML_ASSERT(!ids.empty());
    return n_past_base + ids.size() - 1;
}

void common_speculative_verifier::append_replay(const llama_tokens & ids, llama_batch & batch) const {
    GGML_ASSERT(!ids.empty());

    common_batch_add(batch, id_prev, n_past_base - 1, { seq_id }, false);

    for (size_t i = 0; i + 1 < ids.size(); ++i) {
        common_batch_add(batch, ids[i], n_past_base + i, { seq_id }, false);
    }
}

bool common_speculative_verifier::begin(int32_t n_past, llama_token id_prev) {
    n_past_base = n_past;
    this->id_prev = id_prev;

    if (can_seq_rm_partial) {
        return true;
    }

    if (use_full_state) {
        full_state_size = llama_state_get_size(ctx_tgt);
        if (full_state.size() < full_state_size) {
            full_state.resize(full_state_size);
        }

        if (full_state_size == 0) {
            return true;
        }

        const size_t n_state = llama_state_get_data(ctx_tgt, full_state.data(), full_state_size);
        if (n_state != full_state_size) {
            LOG_ERR("%s: failed to snapshot full target state (%zu != %zu)\n", __func__, n_state, full_state_size);
            return false;
        }

        return true;
    }

    seq_state_size = llama_state_seq_get_size(ctx_tgt, seq_id);
    if (seq_state.size() < seq_state_size) {
        seq_state.resize(seq_state_size);
    }

    if (seq_state_size == 0) {
        return true;
    }

    const size_t n_state = llama_state_seq_get_data(ctx_tgt, seq_state.data(), seq_state_size, seq_id);
    if (n_state != seq_state_size) {
        LOG_ERR("%s: failed to snapshot target sequence state (%zu != %zu)\n", __func__, n_state, seq_state_size);
        return false;
    }

    return true;
}

bool common_speculative_verifier::finish(size_t n_draft, const llama_tokens & ids, int32_t & n_past) {
    if (ids.empty()) {
        LOG_ERR("%s: no accepted tokens to commit\n", __func__);
        return false;
    }

    n_past = n_past_base + ids.size() - 1;

    if (ids.size() == n_draft + 1) {
        return true;
    }

    if (can_seq_rm_partial) {
        if (!llama_memory_seq_rm(mem_tgt, seq_id, n_past, -1)) {
            LOG_ERR("%s: failed to remove speculative suffix for seq %d at pos %d\n", __func__, seq_id, n_past);
            return false;
        }

        return true;
    }

    llama_memory_clear(mem_tgt, true);

    if (use_full_state) {
        if (!restore_snapshot()) {
            return false;
        }
    } else if (seq_state_size > 0) {
        const size_t n_state = llama_state_seq_set_data(ctx_tgt, seq_state.data(), seq_state_size, seq_id);
        if (n_state != seq_state_size) {
            LOG_ERR("%s: failed to restore target sequence state (%zu != %zu)\n", __func__, n_state, seq_state_size);
            return false;
        }
    }

    common_batch_clear(batch_replay);
    append_replay(ids, batch_replay);

    if (llama_decode(ctx_tgt, batch_replay) != 0) {
        LOG_ERR("%s: failed to replay accepted speculative prefix\n", __func__);
        return false;
    }

    return true;
}

static common_speculative * common_speculative_init_impl(
        common_params_speculative & params,
        llama_context             * ctx_tgt,
        llama_context             * ctx_mtp_shared,
        llama_seq_id               seq_id_dft,
        bool                       owns_ctx_mtp) {
    llama_context * ctx_dft = nullptr;
    if (params.model_dft) {
        ctx_dft = llama_init_from_model(params.model_dft, params.cparams_dft);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s", "failed to create draft context\n");
            return nullptr;
        }
    }

    llama_context * ctx_mtp = nullptr;
    if (params.type == COMMON_SPECULATIVE_TYPE_MTP) {
        if (ctx_mtp_shared != nullptr) {
            ctx_mtp = ctx_mtp_shared;
        } else {
            ctx_mtp = common_speculative_create_mtp_context(params, ctx_tgt, llama_n_seq_max(ctx_tgt));
            if (ctx_mtp == nullptr) {
                return nullptr;
            }
        }
    }

    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        bool has_draft = !params.mparams_dft.path.empty();
        bool has_draft_eagle3 = false; // TODO PR-18039: if params.speculative.eagle3
        bool has_mtp = params.type == COMMON_SPECULATIVE_TYPE_MTP && ctx_mtp != nullptr;

        bool has_ngram_cache   = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_CACHE);
        bool has_ngram_simple  = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
        bool has_ngram_map_k   = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
        bool has_ngram_map_k4v = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
        bool has_ngram_mod     = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MOD);

        // In a more complex implementation we could use the same implementation but with different parameters.
        // This was initially used in PR-18471 but removed to simplify the code.
        if (has_ngram_simple) {
            // This implementation can guess a lot of tokens without any draft model.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, params));
        }
        if (has_ngram_map_k) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, params));
        }
        if (has_ngram_map_k4v) {
            // This implementation can guess tokens with high acceptance rate but is more expensive.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, params));
        }
        if (has_ngram_mod) {
            // shared instance for all speculative decoding contexts
            if (!params.ngram_mod) {
                params.ngram_mod = std::make_shared<common_ngram_mod>(params.ngram_size_n, 4*1024*1024);

                LOG_INF("%s: initialized ngram_mod with n=%d, size=%zu (%.3f MB)\n", __func__,
                        params.ngram_size_n, params.ngram_mod->size(),
                        (float)(params.ngram_mod->size_bytes())/1024/1024);

                if (params.ngram_size_n < 16) {
                    LOG_WRN("%s: ngram_mod n=%d is too small - poor quality is possible, see: https://github.com/ggml-org/llama.cpp/pull/19164\n", __func__, params.ngram_size_n);
                }
            }

            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, params));
        }
        if (has_ngram_cache) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, params));
        }
        if (has_draft) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT, params));
        }
        if (has_draft_eagle3) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_EAGLE3, params));
        }
        if (has_mtp) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_MTP, params));
        }
    }

    std::vector<std::unique_ptr<common_speculative_state>> impls = {};

    for (const common_speculative_config & config : configs) {
        LOG_DBG("%s: adding implementation %s\n", __func__, common_speculative_type_to_str(config.type).c_str());
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT: {
                impls.push_back(std::make_unique<common_speculative_state_draft>(config.type,
                    /* .ctx_tgt      = */ ctx_tgt,
                    /* .ctx_dft      = */ ctx_dft,
                    /* .replacements = */ params.replacements
                ));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_state_eagle3>(config.type));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_MTP: {
                impls.push_back(std::make_unique<common_speculative_state_mtp>(
                        config.type,
                        ctx_mtp,
                        seq_id_dft,
                        owns_ctx_mtp,
                        true,
                        common_speculative_get_mtp_shared_state(ctx_mtp)));
                if (owns_ctx_mtp) {
                    ctx_mtp = nullptr;
                }
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram      = */ ngram_size_key,
                    /* .size_mgram      = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_state_ngram_simple>(
                    /* .type            = */ config.type,
                    /* .state           = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(std::make_unique<common_speculative_state_ngram_map_k>(
                    (config.type),
                    get_common_ngram_map(config)
                ));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                GGML_ASSERT(config.params.ngram_mod);
                impls.push_back(std::make_unique<common_speculative_state_ngram_mod>(config.type, *config.params.ngram_mod));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(
                        params.lookup_cache_static, params.lookup_cache_dynamic, config);
                impls.push_back(std::make_unique<common_speculative_state_ngram_cache>(state));
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        LOG_WRN("%s", "no implementations specified for speculative decoding\n");
        return nullptr;
    }

    auto * result = new common_speculative {
        /* .impls = */ std::move(impls)
    };

    return result;
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt) {
    return common_speculative_init_impl(params, ctx_tgt, nullptr, 0, true);
}

common_speculative * common_speculative_init_shared_mtp(
        common_params_speculative & params,
        llama_context             * ctx_tgt,
        llama_context             * ctx_mtp,
        llama_seq_id               seq_id_dft) {
    return common_speculative_init_impl(params, ctx_tgt, ctx_mtp, seq_id_dft, false);
}

llama_context * common_speculative_init_mtp_context(
        const common_params_speculative & params,
        llama_context                   * ctx_tgt,
        uint32_t                          n_seq_max) {
    return common_speculative_create_mtp_context(params, ctx_tgt, n_seq_max);
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(prompt);
        impl->n_call_begin++;
    }
}

llama_tokens common_speculative_draft(
        common_speculative * spec,
        const common_params_speculative & params,
        llama_context * ctx_tgt,
        llama_seq_id seq_id_tgt,
        const llama_tokens & prompt_tgt, // specified in target model vocab
        llama_token id_last) {
    llama_tokens result;

    spec->curr_impl = nullptr; // reset current implementation

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(params, ctx_tgt, seq_id_tgt, prompt_tgt, id_last, result);
            impl->n_call_draft++;
        }

        if (!result.empty()) {
            LOG_DBG("%s: called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n", __func__,
                    common_speculative_type_to_str(impl.get()->type).c_str(), prompt_tgt.size(),
                    impl.get()->n_call_draft, result.size());

            spec->curr_impl = impl.get(); // set current implementation for stats
            impl->n_gen_drafts++;
            impl->n_gen_tokens += result.size();

            break; // We have a draft, so break out of the loop and return it.
        }
    }

    return result;
}

void common_speculative_accept(common_speculative * spec, uint16_t n_accepted) {
    if (n_accepted == 0) {
        return;
    }

    common_speculative_state * impl = spec->curr_impl;

    GGML_ASSERT(impl);

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);
        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(n_accepted);
        impl->n_call_accept++;
    }
}

void common_speculative_accept_tokens(
        common_speculative * spec,
             llama_context * ctx_tgt,
       const llama_tokens  & ids,
       const std::vector<int> & idxs) {
    if (spec == nullptr || ids.empty()) {
        return;
    }

    GGML_ASSERT(ids.size() == idxs.size());

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_sync_us, !impl->gen_perf);
        impl->accept_tokens(ctx_tgt, ids, idxs);
    }
}

void common_speculative_accept_tokens_batch(
        const std::vector<common_speculative *> & specs,
                       llama_context            * ctx_tgt,
        const std::vector<llama_tokens>         & ids_batch,
        const std::vector<std::vector<int>>     & idxs_batch) {
    GGML_ASSERT(specs.size() == ids_batch.size());
    GGML_ASSERT(specs.size() == idxs_batch.size());

    if (specs.empty()) {
        return;
    }

    std::vector<common_speculative_state_mtp *> mtp_states;
    mtp_states.reserve(specs.size());

    llama_context * ctx_dft_shared = nullptr;
    size_t total_tokens = 0;

    for (size_t i = 0; i < specs.size(); ++i) {
        auto * spec = specs[i];
        if (spec == nullptr || ids_batch[i].empty()) {
            continue;
        }

        GGML_ASSERT(ids_batch[i].size() == idxs_batch[i].size());

        auto * mtp = common_speculative_get_mtp_state(spec);
        if (mtp == nullptr) {
            common_speculative_accept_tokens(spec, ctx_tgt, ids_batch[i], idxs_batch[i]);
            return;
        }

        if (ctx_dft_shared == nullptr) {
            ctx_dft_shared = mtp->ctx_dft;
        } else if (ctx_dft_shared != mtp->ctx_dft) {
            for (size_t j = 0; j < specs.size(); ++j) {
                common_speculative_accept_tokens(specs[j], ctx_tgt, ids_batch[j], idxs_batch[j]);
            }
            return;
        }

        mtp_states.push_back(mtp);
        total_tokens += ids_batch[i].size();
    }

    if (mtp_states.size() <= 1 || ctx_dft_shared == nullptr) {
        for (size_t i = 0; i < specs.size(); ++i) {
            common_speculative_accept_tokens(specs[i], ctx_tgt, ids_batch[i], idxs_batch[i]);
        }
        return;
    }

    const size_t n_embd = llama_model_n_embd_out(llama_get_model(ctx_dft_shared));
    std::vector<float> hidden_concat(total_tokens*n_embd);
    std::vector<llama_pos> pos_next(specs.size(), 0);

    size_t hidden_offset_tokens = 0;

    for (size_t i = 0; i < specs.size(); ++i) {
        auto * mtp = common_speculative_get_mtp_state(specs[i]);
        if (mtp == nullptr || ids_batch[i].empty()) {
            continue;
        }

        if (!mtp->is_synced) {
            common_speculative_accept_tokens(specs[i], ctx_tgt, ids_batch[i], idxs_batch[i]);
            return;
        }

        if (mtp->prompt_dft.size() > mtp->prompt_dft_sync_size) {
            auto * mem_dft = llama_get_memory(mtp->ctx_dft);
            llama_memory_seq_rm(mem_dft, mtp->seq_id, mtp->prompt_dft_sync_size, -1);
            mtp->prompt_dft.resize(mtp->prompt_dft_sync_size);
        }

        {
            common_time_meas tm(mtp->t_sync_fetch_us, !mtp->gen_perf);
            if (!llama_get_mtp_hiddens(ctx_tgt, idxs_batch[i].data(), idxs_batch[i].size(), hidden_concat.data() + hidden_offset_tokens*n_embd)) {
                mtp->is_synced = false;
                mtp->have_ready_logits = false;
                mtp->ready_output_idx = -1;
                for (size_t j = 0; j < specs.size(); ++j) {
                    if (specs[j] != specs[i]) {
                        common_speculative_accept_tokens(specs[j], ctx_tgt, ids_batch[j], idxs_batch[j]);
                    }
                }
                return;
            }
        }

        auto * mem_dft = llama_get_memory(mtp->ctx_dft);
        pos_next[i] = llama_memory_seq_pos_max(mem_dft, mtp->seq_id) + 1;
        hidden_offset_tokens += ids_batch[i].size();
    }

    llama_set_mtp_input_hiddens(ctx_dft_shared, hidden_concat.data(), total_tokens, n_embd);

    llama_batch batch = llama_batch_init(std::max<int32_t>(1, (int32_t) llama_n_batch(ctx_dft_shared)), 0, 1);
    common_batch_clear(batch);

    for (size_t i = 0; i < specs.size(); ++i) {
        auto * mtp = common_speculative_get_mtp_state(specs[i]);
        if (mtp == nullptr || ids_batch[i].empty()) {
            continue;
        }

        for (size_t j = 0; j < ids_batch[i].size(); ++j) {
            const bool output = (j + 1 == ids_batch[i].size());
            const int32_t batch_idx = batch.n_tokens;
            common_batch_add(batch, ids_batch[i][j], pos_next[i] + j, { mtp->seq_id }, output);
            if (output) {
                mtp->ready_output_idx = batch_idx;
            }
        }
    }

    const int64_t t_decode_start = ggml_time_us();
    const int ret = llama_decode_mtp(ctx_dft_shared, batch);
    const double t_decode_us = ggml_time_us() - t_decode_start;

    llama_batch_free(batch);

    const double t_decode_us_share = mtp_states.empty() ? 0.0 : t_decode_us / mtp_states.size();
    const uint64_t batch_epoch = ret == 0 ? ++mtp_states.front()->shared_state->output_epoch : 0;

    for (size_t i = 0; i < specs.size(); ++i) {
        auto * mtp = common_speculative_get_mtp_state(specs[i]);
        if (mtp == nullptr || ids_batch[i].empty()) {
            continue;
        }

        if (ret != 0) {
            mtp->is_synced = false;
            mtp->have_ready_logits = false;
            mtp->ready_output_idx = -1;
            mtp->ready_epoch = 0;
            continue;
        }

        mtp->prompt_dft.insert(mtp->prompt_dft.end(), ids_batch[i].begin(), ids_batch[i].end());
        mtp->prompt_dft_sync_size = mtp->prompt_dft.size();
        mtp->have_ready_logits = true;
        mtp->ready_epoch = batch_epoch;
        mtp->t_sync_decode_us += t_decode_us_share;
        mtp->t_sync_us += t_decode_us_share;
    }
}

void common_speculative_print_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_sync_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_sync_fetch_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_sync_decode_us / 1000.0;
            str_perf = ", dur(b,g,a,s,sf,sd) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        const std::string str_extra = impl->extra_stats();

        LOG_INF("statistics %s: #calls(b,g,a) = %zu %zu %zu, #gen drafts = %zu, #acc drafts = %zu, #gen tokens = %zu, #acc tokens = %zu%s%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_perf.c_str(),
                str_extra.c_str());
    }
}
