# Native MTP Optimization Plan

This plan is for the current private native-MTP branch, with the goal of turning native MTP into a real end-to-end CUDA win on the dense Qwen 3.5 path while staying maintainable and upstream-friendly.

## Constraints

- judge every kept step by end-to-end tok/s, not internal phase timing alone
- compare every kept step against:
  - greedy baseline
  - the immediately previous native-MTP step
- keep `np=1` exact against greedy baseline
- treat `np>1` on hybrid/recurrent native-MTP as stability-only
- prefer generic runtime improvements over model-specific hacks
- keep server/runtime layering clean; if a small API is needed, make it generic

## Current Problem

The 2026-04-10 CUDA matrix in [native-mtp-benchmarks.md](native-mtp-benchmarks.md) changed the optimization target from “make 9B UD-Q4 less bad” to “explain why nothing is `np=1` speed-positive and why only 9B Q8 wins on the easier `np=2` cases”.

Current state:

- `Qwen3.5-9B q8_0` is still the only checked model/quant with repeatable net-positive wins, but only on the easier `np=2` cases:
  - `primary np=2`: `127.85 -> 133.26 tok/s` (`1.042x`)
  - `good np=2`: `132.66 -> 137.12 tok/s` (`1.034x`)
  - `primary np=1` is now effectively break-even: `150.53 -> 150.31 tok/s` (`0.999x`)
- `Qwen3.5-9B UD-Q4_K_XL` stays slightly slower even though short-run acceptance is still strong:
  - `primary np=1`: `175.91 -> 159.01 tok/s` (`0.904x`)
  - profile totals there: `draft ~= 21.2 ms`, `accept ~= 75.5 ms`, `replay ~= 30.8 ms`
- `Qwen3.5-27B UD-Q4_K_XL` is slower everywhere and especially bad on `np=2`:
  - `primary np=1`: `72.35 -> 62.27 tok/s`
  - `bad np=2`: `58.64 -> 53.61 tok/s`
- `Qwen3.5-35B-A3B` is still slower on every checked quant.
- `Qwen3.5-35B-A3B Q4_K_M` is now `np=1` exact again on the checked CUDA cases after a permanent one-step post-replay guard was added for hybrid/recurrent native-MTP slots, but it is still far from speed-positive.

Interpretation:

- the remaining problem is still runtime economics, not just “does the model expose an MTP head”
- the current single-token draft can pay off on a favorable dense quant (`9B q8_0`), but the margin is small
- accept cost is still the first recurring bottleneck on the good path
- replay is the main bad-prompt cliff on the larger dense path
- MoE is currently a functionality target, not a speed target
- the backend-resident seed path already landed and should not be reopened as a standalone project unless later evidence changes

A3B correctness side note:

- the A3B `Q4_K_M` exactness failure is now narrowed beyond quant quality alone:
  - promoting `blk.40.nextn.eh_proj.weight` from `Q4_K` to `Q5_K` was the right balanced GGUF fix, but it did not clear `bad np=1`
  - disabling the greedy accept fast path also did not clear it
  - tracing showed the model is on the hybrid recurrent-backup restore path and that the first replayed verifier logits still match baseline
  - exactness came back when the first speculative step after replay was skipped once
- that diagnosis is now codified as the current conservative fix:
  - hybrid/recurrent native-MTP slots force one plain verifier step immediately after replay
  - this restores the `np=1` lossless contract on the checked A3B `Q4_K_M` cases without special-casing Qwen or MoE by name
  - the deeper follow-up, if we want it, is to explain why the first speculative verifier batch after replay is not baseline-equivalent and then remove or relax the guard

## Priority Order

### 1. Harden the benchmark gate

Files:

- `scripts/validate_mtp_cuda.py`
- `docs/development/native-mtp-benchmarks.md`

Required output from the harness:

- repeated baseline vs native-MTP comparisons
- JSON output
- optional `LLAMA_SERVER_MTP_PROFILE=1` parsing
- per-step `native MTP step:` parsing so acceptance and draft/accept/replay cost can be tracked per speculative step
- comparison against greedy baseline and, optionally, a previous native-MTP JSON result

Fixed CUDA cases:

- `primary`: dense 9B short exact regression case
- `good`: known good/stable case
- `bad`: replay-heavy stability case

This step is mandatory before runtime changes are judged.

Status:

- landed in commit `47ba219dd`

### 2. Reduce accept-path overhead first

Primary target:

- `tools/server/server-context.cpp`

Approach:

- add a guarded fast path for pure greedy verifier handling
- keep the current generic sampler path unchanged as fallback
- only use the fast path when:
  - greedy sampling is active
  - no grammar is active
  - no sampler behavior depends on the full generic post-logits pipeline

Likely supporting files:

- `include/llama.h` only if a small generic API is required
- `src/llama-context.cpp`
- `src/llama-graph.h`
- `src/llama-graph.cpp`

Why this is first:

- accept cost is currently the largest recurring bucket and is paid every speculative step

Status:

- landed in commit `5cfd26302`
- result: small but repeatable `np=1` gain on the dense 9B CUDA gate from direct greedy verifier-token accept, with generic fallback still intact

### 3. Reduce verifier output-transfer cost on the same greedy path

Primary targets:

- `tools/server/server-context.cpp`
- `src/llama-context.cpp`
- `include/llama.h`

Approach:

- keep the greedy verifier-token fast path from step 2
- when a decode chunk consists only of pure-greedy native-MTP verifier outputs, request output tokens but suppress raw-logit downloads for that chunk
- keep raw logits enabled for all mixed or generic sampling batches

Why this is next:

- after step 2, the remaining accept cost is still dominated by always-paid verifier output handling
- this keeps the change generic and local, without model-specific hacks

Status:

- landed locally after step 2
- result: another small but repeatable `np=1` gain on the dense 9B CUDA gate
- current reading after the wider model sweep:
  - enough to make `9B q8_0` speed-positive on the easier `np=2` cases
  - not enough to rescue `9B UD-Q4_K_XL`
  - nowhere near enough for 27B or the current A3B quants
### 4. Check draft-path hot reuse before redesigning it

Primary target:

- `src/llama-context.cpp`

Approach:

- instrument MTP graph reuse hits and misses
- instrument graph allocation frequency in steady state
- only add a dedicated cached MTP graph result or scheduler if the counters prove the current path is cold in steady state

Do not:

- start a second standalone seed-path project
- add a dedicated scheduler unless counters show it is necessary

Status:

- attempted with a dedicated MTP scheduler/result cache
- dropped
- reason: end-to-end gains did not hold up under repeated benchmarking; the draft bucket barely moved, so the apparent win was noise rather than a real structural improvement

### 5. Add a conservative replay-triggered cooldown

Primary target:

- `tools/server/server-context.cpp`

Approach:

- after a native replay on a slot, skip native MTP on that slot for a small fixed window
- keep the rule deterministic, simple, and profile-visible
- do not build an opaque adaptive controller

Purpose:

- reduce bad-prompt thrash without destabilizing the good exact path

Why this is now more urgent:

- 27B `bad np=2` still spent `~395 ms` in replay across 3 repeats, versus only `~128 ms` in draft
- even on 9B Q8 the bad case is clearly negative, so bad-prompt replay still blocks “broadly positive” behavior
- the same cooldown shape is also now a correctness diagnostic on A3B:
  - skipping exactly one post-replay speculative step restored `bad np=1` exactness there
  - that has now been promoted from a debug probe to the current permanent hybrid/recurrent correctness guard

### 6. Keep only small hot-path cleanups that beat noise

Primary targets:

- `tools/server/server-context.cpp`
- `src/llama-mtp.h`
- `src/llama-mtp.cpp`

Examples:

- reuse replay batch storage
- reduce temporary seq-id churn if it is still measurable
- drop container churn only if it moves real tok/s

These are cleanup items, not the main strategy.

## Benchmark Gate

Do not keep any optimization unless all of the following remain true:

- `np=1` exact output still matches greedy baseline on the primary case
- median end-to-end tok/s improves versus the immediately previous native-MTP step on the primary case
- the same change is also checked against greedy baseline
- the change does not cause a meaningful regression on the supporting good case
- `np>1` remains stable: no crash, corruption, or invalid token stream

## Explicitly Deferred

Not worth doing yet:

- another standalone seed-transport project
- recursive multi-token native drafting
- model-specific fused kernels before the control-path costs are addressed
- MoE-specific speed tuning as the main optimization track
- tiny container cleanup before accept/replay/draft-reuse work

Those are follow-up options only after the dense path is broadly speed-positive beyond the current `9B q8_0` `np=2` niche.
