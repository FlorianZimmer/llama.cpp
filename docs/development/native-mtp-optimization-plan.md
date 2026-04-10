# Native MTP Optimization Plan

This plan is for the current private native-MTP branch, with the goal of recovering end-to-end CUDA throughput on the dense 9B `UD-Q4_K_XL` quant while staying maintainable and upstream-friendly.

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

On the dense 9B `UD-Q4_K_XL` CUDA case, native MTP is functionally correct but slower than baseline on the primary short exact run:

- baseline `np=1`: about `175 tok/s`
- native MTP `np=1`: about `161.5 tok/s`
- acceptance: about `5 / 6`
- measured overhead on that run:
  - `draft ~= 12.1 ms`
  - `accept ~= 32.2 ms`
  - `replay ~= 10.5 ms`

Interpretation:

- the path is overhead-limited, not acceptance-limited
- the accept path is the first real bottleneck to attack
- replay is the next real cliff on bad prompts
- the backend-resident seed path already landed and should not be reopened as a standalone optimization project unless later evidence changes

## Priority Order

### 1. Harden the benchmark gate

Files:

- `scripts/validate_mtp_cuda.py`
- `docs/development/native-mtp-benchmarks.md`

Required output from the harness:

- repeated baseline vs native-MTP comparisons
- JSON output
- optional `LLAMA_SERVER_MTP_PROFILE=1` parsing
- comparison against greedy baseline and, optionally, a previous native-MTP JSON result

Fixed CUDA cases:

- `primary`: dense 9B short exact regression case
- `good`: known good/stable case
- `bad`: replay-heavy stability case

This step is mandatory before runtime changes are judged.

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

### 3. Check draft-path hot reuse before redesigning it

Primary target:

- `src/llama-context.cpp`

Approach:

- instrument MTP graph reuse hits and misses
- instrument graph allocation frequency in steady state
- only add a dedicated cached MTP graph result or scheduler if the counters prove the current path is cold in steady state

Do not:

- start a second standalone seed-path project
- add a dedicated scheduler unless counters show it is necessary

### 4. Add a conservative replay-triggered cooldown

Primary target:

- `tools/server/server-context.cpp`

Approach:

- after a native replay on a slot, skip native MTP on that slot for a small fixed window
- keep the rule deterministic, simple, and profile-visible
- do not build an opaque adaptive controller

Purpose:

- reduce bad-prompt thrash without destabilizing the good exact path

### 5. Keep only small hot-path cleanups that beat noise

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

Those are follow-up options only if the single-token dense 9B path becomes speed-positive first.
