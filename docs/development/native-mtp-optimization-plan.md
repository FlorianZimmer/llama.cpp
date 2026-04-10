# Native MTP Optimization Plan

Status note (2026-04-11):

This branch is kept as historical context.
The current fork-level native-MTP line is now paused.
The implementation works, the kept dense `np=1` path remains exact, and the latest dense-only branch recovered a real win on `Qwen3.5-9B q8_0`, but broader dense speedups did not survive across the checked targets and the remaining gap now looks structural rather than local.

This plan is for the current private native-MTP branch, with the goal of turning native MTP into a real end-to-end CUDA win on the dense Qwen 3.5 path while staying maintainable and upstream-friendly.

## Scope Decision

The current local recommendation for the first upstream-oriented series is:

- keep the series dense-only
- make `Qwen3.5-9B q8_0` the only active speed target
- keep `Qwen3.5-27B UD-Q4_K_XL` only as supporting dense correctness / regression coverage
- keep `Qwen3.5-35B-A3B` only as regression coverage for already-landed correctness work
- explicitly defer `qwen35moe` / `Qwen3.5-35B-A3B` from the first upstreamable native-MTP series

Why:

- the checked dense path is the only place where native MTP has shown meaningful end-to-end CUDA wins at all
- the current `9B q8_0` regression is now understood well enough to define one last narrow triage branch
- `Qwen3.5-35B-A3B` required both quant-specific rescue work and replay-guard correctness work, and is still materially speed-negative on every checked quant
- deeper hybrid replay/guard economics now look larger than a first upstream-oriented cleanup branch

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
- `Qwen3.5-9B q8_0` also regressed materially from the earlier pre-guard state under the same newer harness:
  - earlier file: `/tmp/native-mtp-bench-20260410/qwen35-9b-q8_0.json`
  - current file: `/tmp/native-mtp-bench-20260410-post-replay-guard/qwen35-9b-q8_0.json`
  - `primary np=1`: `1.082x -> 0.999x`
  - `good np=1`: `1.030x -> 0.938x`
  - `primary np=2`: `1.082x -> 1.042x`
  - `good np=2`: `1.104x -> 1.034x`

Interpretation:

- the remaining problem is still runtime economics, not just “does the model expose an MTP head”
- the current single-token draft can pay off on a favorable dense quant (`9B q8_0`), but the margin is small
- accept cost is still the first recurring bottleneck on the good path
- replay is the main bad-prompt cliff on the larger dense path
- MoE is currently a functionality target, not a speed target
- the backend-resident seed path already landed and should not be reopened as a standalone project unless later evidence changes
- the 2026-04-10 visibility rerun also ruled out a likely server-local win:
  - on the checked Qwen 3.5 dense gates, speculative accept rows are already almost entirely pure fast-path verifier rows with logits suppressed
  - representative coverage from `/tmp/native-mtp-step-01`:
    - `9B q8_0 primary np=1`: `15/15` pure-fast-path, `15/15` logits-suppressed
    - `9B UD-Q4 good np=2`: `180/186` pure-fast-path, `180/186` logits-suppressed
    - `27B UD-Q4 bad np=2`: `180/186` pure-fast-path, `180/186` logits-suppressed
  - that makes a mixed-chunk “split out pure verifier rows” pass unlikely to unlock a large end-to-end win
- Qwen 3.5 dense is also currently classified as `hybrid` in libllama (`src/llama-arch.cpp`), not just `qwen35moe`
  - the current one-step post-replay guard is therefore already the live replay policy on the checked 9B and 27B targets too
  - a separate “dense cooldown” experiment did not create distinct `cooldown_hits`; it collapsed into the same guard behavior and was dropped
- that means the current remaining question is no longer “is there another small verifier fast path in the server?”
  - it is much closer to “can the replay guard be narrowed safely on dense Qwen 3.5, or is the current one-token native-MTP ceiling structural on the hybrid path?”

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

### 1. Keep the benchmark gate and narrow the active targets

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

- active speed target:
  - `Qwen3.5-9B q8_0`
  - `primary`: dense short exact regression case
  - `good`: known good/stable exact case
  - `bad`: replay-heavy stability case
- supporting dense coverage:
  - `Qwen3.5-9B UD-Q4_K_XL`
  - `Qwen3.5-27B UD-Q4_K_XL`
- regression-only coverage:
  - `Qwen3.5-35B-A3B Q4_K_M`
  - only for `np=1` exactness smoke and `np>1` smoke stability when needed

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

### 5. Replay policy work is now a narrow triage question, not a broad cleanup step

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

Status:

- replay policy still matters
- but the current question is no longer “add a second cooldown knob”
- the only narrow runtime branch that still looks justified is:
  - test whether the current post-replay guard can be narrowed from broad hybrid classification to the concrete failing restore/state mode without breaking `np=1` exactness
- if that does not recover a clear repeated `9B q8_0 np=1` win, stop and treat the remaining ceiling as structural for the current one-token native-MTP design

### 5b. Next Branch: Replay-Guard Narrowing Triage

This is the one remaining narrow runtime branch worth trying before stopping.

Target:

- recover the lost dense `9B q8_0 np=1` win if possible
- without reopening a large hybrid-state redesign
- without weakening the checked `np=1` exactness contract

Hypothesis:

- the current broad replay guard is likely the main reason `9B q8_0` regressed from the earlier pre-guard speed-positive state
- dense `Qwen3.5` may not need the same guard scope as the failing A3B replay path

Allowed change shape:

- narrow the guard based on the concrete replay / restore mode or state flags that reproduced the A3B failure
- do not add a new adaptive controller
- do not broaden public API surface unless there is no cleaner local path

Minimum validation matrix:

- active speed target:
  - `Qwen3.5-9B q8_0`: `primary`, `good`, `bad`; `np=1,2`; `repeat=3`
- supporting dense regression checks:
  - `Qwen3.5-9B UD-Q4_K_XL`: `primary`, `good`, `bad`; `np=1,2`; `repeat=3`
  - `Qwen3.5-27B UD-Q4_K_XL`: `primary`, `good`, `bad`; `np=1,2`; `repeat=3`
- regression-only correctness smoke:
  - `Qwen3.5-35B-A3B Q4_K_M`: `primary`, `good`, `bad`; `np=1`; `repeat=1`

Success condition:

- `9B q8_0` regains a clear repeated `np=1` win on `primary` and at least no meaningful regression on `good`
- all checked `np=1` cases remain exact

Failure / stop condition:

- if narrowing the guard does not recover a clear `9B q8_0 np=1` win
- or if exactness regresses on any checked dense or A3B smoke case
- stop this runtime-cleanup line and document the ceiling

### 5c. Stop Condition After Visibility

If all of the following remain true after the visibility pass:

- pure-fast-path verifier coverage is already near-saturated
- logits are already suppressed on almost all speculative accept rows
- the current replay guard is already the live dense-path policy
- repeated end-to-end tok/s remains negative on 9B UD-Q4 and 27B

then stop treating server-local batching tweaks as the leading bet.

At that point the next cycle should explicitly choose one of two directions:

- a separate deeper hybrid replay/guard branch, with the expectation of larger libllama-side changes
- or a documented “structural ceiling” conclusion for current single-token native MTP on Qwen 3.5

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
- deeper hybrid replay/guard economics in this first upstream-oriented branch
- recursive multi-token native drafting
- model-specific fused kernels before the control-path costs are addressed
- MoE-specific speed tuning as the main optimization track
- `qwen35moe` / `Qwen3.5-35B-A3B` as an active speed target for the first upstreamable series
- tiny container cleanup before accept/replay/draft-reuse work

Those are follow-up options only after the dense path is broadly speed-positive beyond the current `9B q8_0` `np=2` niche.

## V1 Upstream Prep Checklist

Before cutting the first upstream-oriented series from this private mirror:

1. keep the series dense-only in both code claims and benchmark framing
2. keep `Qwen3.5-9B q8_0` as the only active speed target
3. keep `Qwen3.5-9B UD-Q4_K_XL` and `Qwen3.5-27B UD-Q4_K_XL` only as supporting dense regression checks
4. keep `Qwen3.5-35B-A3B Q4_K_M` only as a local `np=1` regression smoke target
5. run the one remaining narrow triage branch:
   - test whether the current replay guard can be narrowed safely on dense `qwen35`
   - keep exact `np=1` matching non-negotiable on all checked dense cases
6. stop the runtime-cleanup line if that branch does not recover a clear repeated `9B q8_0 np=1` win
7. prepare the upstream series around the pieces that are already understandable and defensible:
   - backend seed transport
   - greedy verifier fast accept
   - token-only verifier output path
   - benchmark harness / validation docs
8. keep deeper hybrid replay economics, multi-token drafting, and `qwen35moe` speed work out of the first series

If the replay-guard narrowing branch fails, the upstream-ready outcome for v1 is:

- dense-only native MTP support with conservative replay behavior
- explicit documentation that current one-token native MTP has only narrow speed-positive cases on Qwen 3.5
- MoE support retained locally for regression and future work, but not proposed as part of the first upstreamable speed story
