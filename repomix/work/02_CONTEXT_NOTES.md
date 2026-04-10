CONTEXT_NOTES:

Branch and diff base:

- Current branch: `feat/native-mtp-upstream-prep`
- Public-upstream diff base used for comparison: `d6f3030047f85a98b009189e76f441fe818ea44d`
- Relative to `upstream/master`, this private branch currently changes `32` files with about `3651` insertions and `68` deletions.

What this private mirror has that public upstream master may not:

- HF -> GGUF conversion support for Qwen 3.5 native MTP / NextN tensors
- GGUF metadata and loader plumbing for `nextn_predict_layers`
- public native-MTP APIs in `include/llama.h`
- `llama_mtp` runtime state and draft path in `src/llama-mtp.*` and `src/llama-context.cpp`
- Qwen 3.5 dense and Qwen 3.5 MoE native-MTP graph builders
- server integration for `--spec-type mtp`
- benchmark harness `scripts/validate_mtp_cuda.py`
- docs for model prep, benchmarking, and optimization planning

Key local files to treat as most relevant:

- `include/llama.h`
- `src/llama-mtp.h`
- `src/llama-mtp.cpp`
- `src/llama-context.h`
- `src/llama-context.cpp`
- `src/llama-graph.h`
- `src/llama-graph.cpp`
- `src/models/qwen35.cpp`
- `src/models/qwen35moe.cpp`
- `tools/server/server-context.cpp`
- `scripts/validate_mtp_cuda.py`
- `docs/development/native-mtp-benchmarks.md`
- `docs/development/native-mtp-optimization-plan.md`
- `docs/development/native-mtp-model-prep.md`

Current runtime facts:

- Current native runtime only drafts `1` continuation token per step even if metadata reports more predictor layers.
- `np=1` is the correctness-clean contract.
- `np>1` on the current hybrid/recurrent path is stability-only.
- The backend-resident seed path already exists; reopening seed transport as a standalone project is probably low value unless new evidence says otherwise.

Latest full benchmark sweep summary:

- The current 3-repeat matrix includes the landed hybrid/recurrent post-replay guard fix.
- No checked model or quant is net-positive on `np=1`.
- Only `Qwen3.5-9B q8_0` showed repeatable net-positive speedups, and only on the easier `np=2` cases.
- `Qwen3.5-9B UD-Q4_K_XL` remained slower everywhere despite decent short-case acceptance.
- `Qwen3.5-27B UD-Q4_K_XL` was slower everywhere.
- `Qwen3.5-35B-A3B` was slower on every checked quant.
- Current branch state after the replay fix:
  - `Qwen3.5-35B-A3B Q4_K_M` is `np=1` exact again on the checked `primary`, `good`, and `bad` CUDA cases
  - it is still substantially slower than baseline, so the correctness fix did not make it speed-positive

Quantization-side and replay-side follow-up after the sweep:

- The A3B `Q4_K_M` `bad np=1` exactness failure is not a generic native-MTP runtime failure, but it is also no longer explainable by quantization alone.
- A direct BF16-vs-quant audit was added in `scripts/audit_mtp_quantization.py`.
- On the current A3B GGUFs:
  - all MTP norm tensors already remain `F32`
  - the only quantized MTP tensor is `blk.40.nextn.eh_proj.weight`
  - `Q4_K_M` stores it as `Q4_K`
  - `Q5_K_M` stores it as `Q5_K`
  - `UD-Q4_K_XL` stores it as `Q8_0`
- Measured against the BF16 GGUF:
  - `Q4_K`: `rel_rmse ~= 0.0759`, cosine `~= 0.9971`
  - `Q5_K`: `rel_rmse ~= 0.0417`, cosine `~= 0.9991`
  - `Q8_0`: `rel_rmse ~= 0.0086`, cosine `~= 0.9999`
- Checked-in override files now exist for the A3B `Q4_K_M` recipe:
  - balanced: promote only `blk.40.nextn.eh_proj.weight` to `Q5_K`
  - strict: promote it to `Q8_0`
- The balanced A3B `Q4_K_M` rebuild has now been performed and promoted to the canonical GGUF filename on disk.
- The previous pre-balanced file was then removed to recover `/mnt/models` space.
- Important runtime-facing caveat after rebuilding:
  - the balanced A3B `Q4_K_M` GGUF still failed the narrow `bad np=1` exactness validation
  - so this quantization fix was necessary but not sufficient for that case
- Replay isolation after rebuilding narrowed the remaining issue further:
  - disabling the greedy accept fast path still did not fix the divergence
  - the model is taking the hybrid recurrent-backup restore path, not `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`
  - the first replayed verifier logits after restore were still correct:
    - replayed token `271` at `pos=10`
    - top-1 next token `248068` with probability `~0.91`
    - that matches the greedy baseline continuation at that point
  - the divergence starts on the first speculative step after replay, not on the replayed next-token state itself
  - a one-step debug cooldown after replay restored exactness on both:
    - the short traced repro
    - the full `bad np=1` CUDA validation case
- Current best local reading for A3B `Q4_K_M`:
  - quant quality was part of the problem
  - the remaining correctness gap was in the first speculative verifier batch after replay on the hybrid/MoE path
- Fix now landed locally:
  - hybrid/recurrent native-MTP slots always force one plain verifier step immediately after replay
  - this restored `np=1` exactness on the checked A3B `Q4_K_M` cases
  - 9B and 27B dense references stayed exact on the same spot-check matrix
  - A3B `np=2` stayed stability-clean in a smoke run
- `Qwen3.5-27B-MTP-UD-Q4_K_XL` already stores its `blk.64.nextn.eh_proj.weight` as `Q8_0`, so there is no obvious MTP-head under-quantization issue there.
- `Qwen3.5-9B` no longer had a BF16 GGUF on disk, so only a surrogate audit against the shipped `q8_0` GGUF was possible.
- That surrogate audit still showed the key operational fact:
  - both shipped 9B quants already keep the MTP head at `Q8_0`
  - so there was no balanced MTP-head update to apply on 9B

Representative results:

- 9B `UD-Q4_K_XL`, `primary np=1`: `175.91 -> 159.01 tok/s` (`0.904x`)
- 9B `q8_0`, `primary np=1`: `150.53 -> 150.31 tok/s` (`0.999x`)
- 9B `q8_0`, `primary np=2`: `127.85 -> 133.26 tok/s` (`1.042x`)
- 27B `UD-Q4_K_XL`, `primary np=1`: `72.35 -> 62.27 tok/s` (`0.861x`)
- 35B-A3B `Q4_K_M`, `primary np=1`: `228.26 -> 170.72 tok/s` (`0.748x`)

Representative profile readings from new per-step instrumentation:

- 9B `UD-Q4_K_XL`, `primary np=1`:
  - acceptance `12/15 (0.800)`
  - `draft 21.212 ms`
  - `accept 75.516 ms`
  - `replay 30.801 ms`
  - mean step total `8503.1 us`
- 9B `q8_0`, `primary np=1`:
  - acceptance `12/15 (0.800)`
  - `draft 23.585 ms`
  - `accept 99.580 ms`
  - `replay 8.597 ms`
  - mean step total `8785.1 us`
- 27B `UD-Q4_K_XL`, `bad np=2`:
  - acceptance `140/184 (0.761)`
  - `draft 127.729 ms`
  - `accept 958.105 ms`
  - `replay 395.062 ms`
  - mean step total `7962.4 us`

Interpretation we already believe locally:

- the remaining blocker is runtime economics, not merely model support
- accept cost is still the recurring hot-path tax
- replay remains a major bad-prompt cliff, especially on 27B
- 9B `q8_0` proves the approach can win on easier `np=2` cases, but the margin is narrow and the `np=1` win has disappeared in the 3-repeat median
- MoE support is currently more of a functionality milestone than a speed milestone

Recent local-only additions on top of the earlier private native-MTP work:

- `tools/server/server-context.cpp` now emits per-step lines:
  - `native MTP step: step=... drafted=... accepted=... replay=... draft=... us snapshot=... us accept=... us restore=... us replay_us=... total=... us`
- `scripts/validate_mtp_cuda.py` now parses:
  - aggregate `native MTP profile: ...`
  - per-step `native MTP step: ...`
  - step-level acceptance and timing summaries into JSON
- docs were updated to include the full 2026-04-10 multi-model matrix and the new optimization reading

What to optimize for:

- real end-to-end tok/s, not isolated micro-timing wins
- small benchmark-gated steps
- maintainable changes that could be upstreamed incrementally
- generic dense-path improvements before model-specific tricks

What is probably not worth prioritizing:

- another large seed-transport project
- recursive multi-token native drafting before the single-token path is broadly speed-positive
- large model-specific fused-kernel work before server/runtime control-path costs are better understood
- MoE-specific speed tuning as the main line of work right now

What should not be misattributed:

- do not treat the A3B `Q4_K_M` exactness failure as proof that the runtime rollback/accept path is generically wrong
- do not treat it as “just a broken GGUF” either
- the current evidence is:
  - quant quality needed to be fixed first
  - restore+replay can rebuild the correct immediate next-token state
  - the first speculative step after replay was the remaining correctness fault line
  - the current branch addresses that with a conservative post-replay plain-step guard on hybrid/recurrent slots
