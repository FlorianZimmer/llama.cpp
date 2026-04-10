# Code Excerpts

These notes point at the local code that differs materially from public upstream and matters for the dense-only native-MTP review.

## 1. Native-MTP capability is now dense-only

File: `src/llama-mtp.cpp`

Current capability gate:

```cpp
static bool llm_arch_supports_native_mtp(const llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_QWEN35:
            return true;
        default:
            return false;
    }
}
```

Why it matters:

- the live branch no longer claims native-MTP support for `qwen35moe`
- any next-step plan should treat this branch as dense-only unless it explicitly proposes reintroducing MoE later

## 2. Dense runtime still uses the same generic native-MTP plumbing

Files:

- `include/llama.h`
- `src/llama-mtp.h`
- `src/llama-context.cpp`
- `src/llama-graph.h`
- `src/llama-graph.cpp`

Important local API / runtime additions:

```cpp
LLAMA_API bool llama_model_has_native_mtp(const struct llama_model * model);
LLAMA_API uint32_t llama_model_n_native_mtp_predict(const struct llama_model * model);

LLAMA_API int32_t llama_native_mtp_draft_batch(...);
LLAMA_API int32_t llama_native_mtp_draft(...);

LLAMA_API void llama_set_output_tokens(struct llama_context * ctx, bool output_tokens);
LLAMA_API void llama_set_output_logits(struct llama_context * ctx, bool output_logits);
LLAMA_API llama_token llama_get_output_token_ith(struct llama_context * ctx, int32_t i);
```

Why it matters:

- the branch already added the generic surface needed for the current fast accept path
- a good next-step plan should avoid reopening solved output-transfer or seed-plumbing work unless there is new evidence

## 3. Backend-resident seed transport already exists

File: `src/llama-mtp.h`

Relevant shape:

```cpp
struct llama_mtp_backend_seed_state {
    ggml_backend_t             backend        = nullptr;
    ggml_backend_buffer_type_t buft           = nullptr;
    ggml_tensor *              seed_cache_dev = nullptr;
    ggml_tensor *              seed_batch_dev = nullptr;
    ...
};
```

Why it matters:

- this branch already removed a backend -> host -> backend seed round-trip
- future work should build on this instead of proposing another standalone seed transport pass

## 4. Dense Qwen 3.5 graph support is already in place

File: `src/models/qwen35.cpp`

What matters:

- the dense builder already has the native-MTP graph path
- the remaining problem is not model-family enablement
- the remaining problem is runtime economics on the current one-token design

## 5. Server hot path already has fast accept plus visibility

File: `tools/server/server-context.cpp`

Important local behavior:

```cpp
const bool pure_fast_path_verifier_batch =
    has_output_rows && use_output_tokens && disable_output_logits;
const bool logits_suppressed_for_accept =
    has_output_rows && disable_output_logits;
...
SLT_INF(slot,
        "native MTP step:"
        " step=%d"
        " drafted=%zu"
        " accepted=%zu"
        " replay=%d"
        " fast=%d"
        " logits_suppressed=%d"
        " forced_plain=%d"
        " guard=%d"
        ...
);
```

Current replay-guard scope:

```cpp
static bool server_native_mtp_needs_post_replay_guard(const llama_model * model) {
    return llama_model_is_recurrent(model);
}
```

Why it matters:

- the broad hybrid guard was removed from dense `qwen35`
- the live branch no longer carries a MoE-specific replay guard path
- step-level visibility already exists, so any next plan can be benchmark-driven at the per-step level

## 6. The validator is already strong enough for small-step optimization work

File: `scripts/validate_mtp_cuda.py`

Important points:

- exact `np=1` comparison to greedy baseline
- repeat support
- JSON output
- `--compare-json`
- aggregate profile parsing
- per-step visibility parsing

Why it matters:

- the next plan does not need a new benchmark harness before optimization work begins
- each step can already be gated against both baseline and the previous native-MTP result

## 7. Historical failures are still relevant even though MoE is out of scope

The live branch removed MoE from code scope, but the review should still account for this history:

- broad replay guard fixed correctness but regressed dense `9B q8_0`
- draft-side caching / scheduler work did not survive repeated end-to-end tok/s testing
- dense visibility showed fast-path coverage was already near-saturated
- `9B q8_0` remains the only checked dense path with a meaningful `np=1` win
- `9B UD-Q4_K_XL` and `27B UD-Q4_K_XL` still do not meet the likely maintenance bar

That history lives mostly in the included docs and context notes, not in the live code paths anymore.
