#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;
struct common_speculative_verifier;

// comma separated list of all types
std::string common_speculative_type_name_str();

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// check if the llama_context is compatible for speculative decoding
// note: clears the memory of the context
bool common_speculative_is_compat(llama_context * ctx_tgt);

common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt);

common_speculative * common_speculative_init_shared_mtp(
        common_params_speculative & params,
        llama_context             * ctx_tgt,
        llama_context             * ctx_mtp,
        llama_seq_id               seq_id_dft);

llama_context * common_speculative_init_mtp_context(
        const common_params_speculative & params,
        llama_context                   * ctx_tgt,
        uint32_t                          n_seq_max);

void common_speculative_free(common_speculative * spec);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt);

// sample up to n_draft tokens and add them to the batch using the draft model
llama_tokens common_speculative_draft(
                     common_speculative * spec,
        const common_params_speculative & params,
                     llama_context     * ctx_tgt,
                           llama_seq_id  seq_id_tgt,
                     const llama_tokens & prompt,
                            llama_token   id_last);

// informs the speculative decoder that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, uint16_t n_accepted);

// synchronize speculative state with the accepted target tokens and their sampled output rows.
// idxs must have the same length as ids and refer to output rows from the most recent target decode.
void common_speculative_accept_tokens(
        common_speculative * spec,
             llama_context * ctx_tgt,
       const llama_tokens  & ids,
       const std::vector<int> & idxs);

void common_speculative_accept_tokens_batch(
        const std::vector<common_speculative *> & specs,
                       llama_context            * ctx_tgt,
        const std::vector<llama_tokens>         & ids_batch,
        const std::vector<std::vector<int>>     & idxs_batch);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

struct common_speculative_verifier {
    common_speculative_verifier(llama_context * ctx_tgt, llama_seq_id seq_id);
    ~common_speculative_verifier();

    common_speculative_verifier(const common_speculative_verifier &) = delete;
    common_speculative_verifier & operator=(const common_speculative_verifier &) = delete;

    bool begin(int32_t n_past, llama_token id_prev);
    bool finish(size_t n_draft, const llama_tokens & ids, int32_t & n_past);

    void set_seq_id(llama_seq_id seq_id);
    bool uses_full_state() const;
    bool restore_snapshot();
    int32_t n_past_after(const llama_tokens & ids) const;
    void append_replay(const llama_tokens & ids, llama_batch & batch) const;

private:
    llama_context * ctx_tgt;
    llama_memory_t mem_tgt;
    llama_seq_id seq_id;
    const bool can_seq_rm_partial;
    const bool use_full_state;

    llama_batch batch_replay;

    int32_t n_past_base = 0;
    llama_token id_prev = LLAMA_TOKEN_NULL;

    size_t full_state_size = 0;
    std::vector<uint8_t> full_state;
    size_t seq_state_size = 0;
    std::vector<uint8_t> seq_state;
};
