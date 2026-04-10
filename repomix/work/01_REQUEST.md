USER_GOAL:
Review the current private native-MTP state in this llama.cpp mirror and produce a concrete next-step optimization plan for getting real net-positive end-to-end speedups on CUDA while staying upstream-friendly and maintainable.

DELIVERABLE_TYPE: PLAN

USER_REQUEST:
Please review this private-mirror native-MTP implementation and produce a pragmatic optimization plan focused on real end-to-end speedups, not internal timing wins alone.

Important benchmark context from the latest full sweep on 2026-04-10:

- Models tested:
  - Qwen3.5-9B `UD-Q4_K_XL`
  - Qwen3.5-9B `q8_0`
  - Qwen3.5-27B `UD-Q4_K_XL`
  - Qwen3.5-35B-A3B `Q4_K_M`
  - Qwen3.5-35B-A3B `Q5_K_M`
  - Qwen3.5-35B-A3B `UD-Q4_K_XL`
- Cases tested: `primary`, `good`, `bad`
- Parallel counts tested: `np=1`, `np=2`
- Repeats: `3`
- Current result:
  - no checked model or quant is net-positive on `np=1`
  - Qwen3.5-9B `q8_0` is still the only speed-positive path, but only on the easier `np=2` cases
  - Qwen3.5-9B `UD-Q4_K_XL` is slower everywhere
  - Qwen3.5-27B is speed-negative everywhere
  - Qwen3.5-35B-A3B is speed-negative everywhere
  - Qwen3.5-35B-A3B `Q4_K_M` is now `np=1` exact again after the landed post-replay guard fix

Representative exact `np=1` results:

- 9B `UD-Q4_K_XL`: `175.91 -> 159.01 tok/s` (`0.904x`)
- 9B `q8_0`: `150.53 -> 150.31 tok/s` (`0.999x`)
- 27B `UD-Q4_K_XL`: `72.35 -> 62.27 tok/s` (`0.861x`)
- 35B-A3B `Q4_K_M`: `228.26 -> 170.72 tok/s` (`0.748x`)
- 35B-A3B `Q5_K_M`: `221.34 -> 167.90 tok/s` (`0.759x`)
- 35B-A3B `UD-Q4_K_XL`: `202.80 -> 129.34 tok/s` (`0.638x`)

Representative speed-positive `np=2` results:

- 9B `q8_0`, `primary np=2`: `127.85 -> 133.26 tok/s` (`1.042x`)
- 9B `q8_0`, `good np=2`: `132.66 -> 137.12 tok/s` (`1.034x`)

Representative profile summaries from the new per-step instrumentation:

- 9B `UD-Q4_K_XL`, `primary np=1`:
  - acceptance `12/15 (0.800)`
  - `draft ~= 21.2 ms`
  - `accept ~= 75.5 ms`
  - `replay ~= 30.8 ms`
- 9B `q8_0`, `primary np=1`:
  - acceptance `12/15 (0.800)`
  - `draft ~= 23.6 ms`
  - `accept ~= 99.6 ms`
  - `replay ~= 8.6 ms`
- 27B `UD-Q4_K_XL`, `bad np=2`:
  - acceptance `140/184 (0.761)`
  - `draft ~= 127.7 ms`
  - `accept ~= 958.1 ms`
  - `replay ~= 395.1 ms`

Important quantization-side follow-up found after the sweep:

- the `Qwen3.5-35B-A3B Q4_K_M` `bad np=1` exactness failure was not just a generic runtime bug, but it also was not explained by quantization alone
- all A3B MTP GGUFs preserve the same MTP metadata and tensor set
- the only differing MTP tensor across the tested A3B quants is:
  - `blk.40.nextn.eh_proj.weight`
- that tensor is:
  - `Q4_K` in the failing `Q4_K_M` GGUF
  - `Q5_K` in the passing `Q5_K_M` GGUF
  - `Q8_0` in the passing `UD-Q4_K_XL` GGUF
- BF16-vs-quant audit on that tensor showed:
  - `Q4_K`: `rel_rmse ~= 0.0759`, cosine `~= 0.9971`
  - `Q5_K`: `rel_rmse ~= 0.0417`, cosine `~= 0.9991`
  - `Q8_0`: `rel_rmse ~= 0.0086`, cosine `~= 0.9999`
- checked-in balanced recommendation for that tensor is `Q5_K`
- checked-in strict recommendation for that tensor is `Q8_0`
- the balanced `Q4_K_M` rebuild is now the canonical GGUF on disk
- the remaining exactness issue was isolated to the first speculative step after replay on the hybrid/recurrent path
- the current branch now fixes that conservatively by forcing one plain verifier step immediately after replay on hybrid/recurrent native-MTP slots

Please:
1. Identify the most likely remaining bottlenecks in the current implementation.
2. Propose upstream-friendly optimization steps in priority order.
3. Keep the plan benchmark-gated after each step against:
   - greedy baseline
   - the immediately previous native-MTP step
4. Explicitly call out which ideas are likely not worth doing yet.
5. Distinguish:
   - dense-model generic work that could improve 9B and 27B
   - work that may help MoE correctness or stability but is unlikely to rescue speed
   - work that is runtime-side versus work that is really quantization-side
6. Use the provided local file excerpts and docs as the source of truth for this private branch.
7. If you need public upstream context, fetch it surgically from public `ggml-org/llama.cpp` only for exact files or symbols that matter. Do not broaden into a full-repo read.

CONSTRAINTS:
- Prefer small-to-medium scoped steps over large speculative rewrites.
- Preserve the current validated correctness contract:
  - `np=1` exactness matters
  - `np>1` on hybrid/recurrent native-MTP is stability-focused, not strict batch-invariant exactness
- Focus on runtime/server/graph overhead before suggesting deeper kernel or scheduler redesign.
- Stay close to something that could plausibly be upstreamed in pieces.
