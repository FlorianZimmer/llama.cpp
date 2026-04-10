# Code Excerpts

These are short excerpts of local changes that do not exist in public upstream master at the stated merge-base. They are meant to orient the browser model before it reads the included source files.

## 1. Public API surface for native MTP and greedy output-token handling

File: `include/llama.h`

```cpp
// model capability / metadata
LLAMA_API bool llama_model_has_native_mtp(const struct llama_model * model);
LLAMA_API uint32_t llama_model_n_native_mtp_predict(const struct llama_model * model);

// native draft entrypoints
LLAMA_API int32_t llama_native_mtp_draft_batch(...);
LLAMA_API int32_t llama_native_mtp_draft(...);

// output-transfer controls used by the server fast path
LLAMA_API void llama_set_output_tokens(struct llama_context * ctx, bool output_tokens);
LLAMA_API void llama_set_output_logits(struct llama_context * ctx, bool output_logits);
LLAMA_API llama_token llama_get_output_token_ith(struct llama_context * ctx, int32_t i);

// hybrid/recurrent memory helpers used for rollback / replay
LLAMA_API bool llama_memory_seq_rm_attn(...);
LLAMA_API bool llama_memory_seq_rm_recr(...);
LLAMA_API bool llama_memory_seq_cp_recr(...);
```

Why it matters:

- the current server fast path depends on direct greedy verifier-token access and optional logits suppression
- rollback / replay for the hybrid-recurrent native-MTP path relies on the split memory helpers

## 2. New runtime state for backend-resident seed capture

File: `src/llama-mtp.h`

```cpp
struct llama_mtp_backend_seed_state {
    ggml_backend_t             backend        = nullptr;
    ggml_backend_buffer_type_t buft           = nullptr;
    uint32_t                   n_embd         = 0;
    uint64_t                   generation     = 0;
    ggml_tensor *              seed_cache_dev = nullptr;
    ggml_tensor *              seed_batch_dev = nullptr;
    ...
};

struct llama_mtp_state {
    llama_mtp_desc desc;
    ...
    llama_mtp_seed_mode         seed_mode = LLAMA_MTP_SEED_MODE_NONE;
    uint64_t                    backend_seed_generation_next = 1;
    llama_mtp_backend_seed_state seed_backend;
    ...
    bool ensure_backend_seed_storage(ggml_backend_t backend, ggml_backend_buffer_type_t buft);
};
```

Why it matters:

- the branch already removed a full backend -> host -> backend seed round-trip
- future work should generally build on this path instead of reopening seed transport from scratch

## 3. Backend capture path in decode

File: `src/llama-context.cpp`

```cpp
static bool mtp_capture_seed_rows_backend(
        ggml_tensor * tensor,
        const std::map<llama_seq_id, uint32_t> & seq_to_row,
        llama_mtp_state & dst,
        ggml_backend_sched_t sched,
        size_t row_size) {
    ...
    if (!dst.ensure_backend_seed_storage(backend, buft)) {
        return false;
    }
    ...
    for (const auto & [seq_id, src_row] : src_rows) {
        ggml_backend_tensor_copy_async(backend, backend, src_row, dst.seed_backend.seed_cache_rows[seq_id]);
        dst.mark_seed(seq_id);
    }
    ...
}
```

Why it matters:

- this is part of the already-landed structural cleanup
- it improved architecture cleanliness but did not by itself make the broad CUDA speedups appear

## 4. Qwen 3.5 dense and MoE builders are wired for native MTP

Files:

- `src/models/qwen35.cpp`
- `src/models/qwen35moe.cpp`

Relevant shape of the implementation:

```cpp
// Qwen3Next uses a single Q projection that outputs query + gate
ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s);
...
ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, ...);
gate = ggml_cont_2d(ctx0, gate, ...);
...
cur = build_attn(inp, ..., Qcur, Kcur, Vcur, ..., kq_scale, il);
...
cur = ggml_mul(ctx0, cur, gate_sigmoid);
```

Why it matters:

- this private branch already has working model-side graph support for both dense `qwen35` and MoE `qwen35moe`
- the main remaining problem is not “support the model family” but “make the runtime economics work”

## 5. Server accept / rollback / replay hot path with new per-step profiling

File: `tools/server/server-context.cpp`

```cpp
bool use_output_tokens = false;
bool disable_output_logits = true;
...
if (!server_native_mtp_can_use_output_fast_path(it_slot->smpl.get(), it_slot->uses_native_mtp()) ||
    std::find(it_slot->i_batch_dft.begin(), it_slot->i_batch_dft.end(), batch_idx) == it_slot->i_batch_dft.end()) {
    disable_output_logits = false;
    continue;
}
use_output_tokens = true;
...
llama_set_output_tokens(ctx, use_output_tokens);
llama_set_output_logits(ctx, !disable_output_logits);

const int ret = llama_decode(ctx, batch_view);
...
if (slot.uses_native_mtp()) {
    result.step = ++slot.native_mtp_step;
    result.t_accept_us = ggml_time_us() - t_accept_start;
    slot.native_mtp_profile.n_accept += 1;
    slot.native_mtp_profile.t_accept_us += result.t_accept_us;
}
...
const bool restore_ok =
    (!has_partial_state || llama_memory_seq_rm_attn(llama_get_memory(ctx), slot.id, spec_result.n_prompt_base, -1)) &&
    (use_recurrent_backup
        ? llama_memory_seq_cp_recr(llama_get_memory(ctx), slot.native_mtp_backup_id, slot.id, -1, -1)
        : slot.restore_native_mtp_state());
...
const bool replay_ok = replay_native_mtp_prefix_batch(native_replay_slots);
...
if (slot.uses_native_mtp_post_replay_guard()) {
    slot.native_mtp_skip_next_draft = std::max(slot.native_mtp_skip_next_draft, 1);
}
...
if (slot.uses_native_mtp_post_replay_guard() && slot.native_mtp_skip_next_draft > 0) {
    slot.native_mtp_skip_next_draft -= 1;
    n_draft_max = 0;
}
...
SLT_INF(slot,
        "native MTP step:"
        " step=%d"
        " drafted=%zu"
        " accepted=%zu"
        " replay=%d"
        " draft=%" PRId64 " us"
        " snapshot=%" PRId64 " us"
        " accept=%" PRId64 " us"
        " restore=%" PRId64 " us"
        " replay_us=%" PRId64
        " total=%" PRId64 " us\n",
        ...);
```

Why it matters:

- this is where the current control-path economics are most visible
- the branch already has a direct greedy accept fast path plus optional logits suppression, but accept and replay still dominate on many cases
- the current branch now also carries a conservative correctness guard for hybrid/recurrent replay: force one plain verifier step immediately after replay

## 6. Graph input support for native-MTP seed transport

File: `src/llama-graph.h`

```cpp
class llm_graph_input_mtp_seed : public llm_graph_input_i {
public:
    llm_graph_input_mtp_seed(
            uint32_t n_embd,
            uint32_t n_mtp,
            llama_mtp_seed_mode mode,
            const float * seed,
            ggml_tensor * seed_backend,
            uint64_t seed_generation);

    void set_input(const llama_ubatch * ubatch) override;
    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * t_seed = nullptr; // F32 [n_embd, n_mtp]
    ...
};
```

Why it matters:

- graph reuse and input reuse for the MTP seed path are already explicit concerns in this branch
- if the next plan proposes graph-level work, it should account for the existing seed-input abstraction rather than treating the graph layer as unchanged upstream code

## 7. Harness support for step-level acceptance and timing summaries

File: `scripts/validate_mtp_cuda.py`

```python
STEP_RE = re.compile(
    r"native MTP step:"
    r" step=(?P<step>\\d+)"
    r" drafted=(?P<drafted>\\d+)"
    r" accepted=(?P<accepted>\\d+)"
    r" replay=(?P<replay>[01])"
    r" draft=(?P<draft_us>\\d+)\\s+us"
    r" snapshot=(?P<snapshot_us>\\d+)\\s+us"
    r" accept=(?P<accept_us>\\d+)\\s+us"
    r" restore=(?P<restore_us>\\d+)\\s+us"
    r" replay_us=(?P<replay_us>\\d+)"
    r" total=(?P<total_us>\\d+)\\s+us"
)
```

Why it matters:

- the benchmark harness can now report both aggregate phase totals and per-step acceptance/runtime summaries
- the next optimization plan should assume this level of measurement exists and use it as the gate

## 8. New MTP quant audit and checked-in override files

Files:

- `scripts/audit_mtp_quantization.py`
- `scripts/mtp_quant_overrides/qwen35moe-a3b-q4_k_m-balanced.tensor-types.txt`
- `scripts/mtp_quant_overrides/qwen35moe-a3b-q4_k_m-strict.tensor-types.txt`

Relevant behavior:

```text
reference: /mnt/models/GGUF/Qwen3.5-35B-A3B-MTP-bf16-fixed.gguf

blk.40.nextn.eh_proj.weight
  q4  Q4_K  rel_rmse ~= 0.0759  cosine ~= 0.9971
  q5  Q5_K  rel_rmse ~= 0.0417  cosine ~= 0.9991
  q8  Q8_0  rel_rmse ~= 0.0086  cosine ~= 0.9999
  balanced -> q5:Q5_K
  strict   -> q8:Q8_0
```

Checked-in balanced override:

```text
^blk\.40\.nextn\.eh_proj\.weight$=Q5_K
```

Why it matters:

- the external review should not waste time blaming the A3B `Q4_K_M` exactness failure on the runtime alone
- there is now a concrete quantization-side explanation and a small checked-in remedy
- but the current balanced remedy is not a full fix:
  - the rebuilt balanced A3B `Q4_K_M` GGUF still fails the narrow `bad np=1` exactness check
  - so the external review should treat this as a partially-reduced quantization problem, not a fully-closed issue
