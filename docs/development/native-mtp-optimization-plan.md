# Native MTP Optimization Plan

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

- the post-replay guard was narrowed from broad hybrid coverage to:
  - recurrent models
  - `qwen35moe`
- `Qwen3.5-9B q8_0` recovered the lost dense `np=1` win under the same newer harness:
  - current file: `/tmp/native-mtp-v1-triage/qwen35-9b-q8_0.json`
  - `primary np=1`: `150.94 -> 163.10 tok/s` (`1.081x`)
  - `good np=1`: `148.90 -> 153.65 tok/s` (`1.032x`)
  - `bad np=1`: `148.89 -> 147.56 tok/s` (`0.991x`)
- `Qwen3.5-9B UD-Q4_K_XL` improved under the same narrowing, but is still not a reliable `np=1` win:
  - current file: `/tmp/native-mtp-v1-triage/qwen35-9b-ud-q4.json`
  - `primary np=1`: `175.39 -> 167.94 tok/s` (`0.957x`)
  - `good np=1`: `196.31 -> 182.85 tok/s` (`0.931x`)
  - `primary np=2`: `156.34 -> 157.28 tok/s` (`1.006x`)
- `Qwen3.5-27B UD-Q4_K_XL` remains slower everywhere and is not a near-term speed target:
  - current file: `/tmp/native-mtp-v1-triage/qwen35-27b-ud-q4.json`
  - `primary np=1`: `72.26 -> 59.71 tok/s` (`0.826x`)
  - `good np=1`: `70.96 -> 68.41 tok/s` (`0.964x`)
  - `bad np=2`: `59.17 -> 32.92 tok/s` (`0.556x`)
- `Qwen3.5-35B-A3B Q4_K_M` stayed `np=1` exact in the smoke rerun, with the retained MoE-only guard still firing on replayed steps:
  - current file: `/tmp/native-mtp-v1-triage/qwen35-35b-a3b-q4_k_m-smoke.json`
  - `primary`: `forced_plain=1`, `guard=1`
  - `good`: `forced_plain=3`, `guard=3`
  - `bad`: `forced_plain=5`, `guard=5`
- `Qwen3.5-9B q8_0` had previously regressed materially from the earlier pre-guard state under the same newer harness:
  - earlier file: `/tmp/native-mtp-bench-20260410/qwen35-9b-q8_0.json`
  - current file: `/tmp/native-mtp-bench-20260410-post-replay-guard/qwen35-9b-q8_0.json`
  - `primary np=1`: `1.082x -> 0.999x`
  - `good np=1`: `1.030x -> 0.938x`
  - `primary np=2`: `1.082x -> 1.042x`
  - `good np=2`: `1.104x -> 1.034x`

Interpretation:

- the remaining problem is still runtime economics, not just “does the model expose an MTP head”
- the current single-token draft can still pay off on a favorable dense quant (`9B q8_0`)
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
  - a broad hybrid-level post-replay guard regressed the dense path because it also hit 9B and 27B
  - a separate “dense cooldown” experiment did not create any new behavior beyond that broad guard and was dropped
- the replay-policy triage question is now answered for v1:
  - yes, narrowing the guard safely on dense `qwen35` recovered the target `9B q8_0` win
  - no, that same narrowing does not rescue 27B or make Q4 broadly speed-positive

A3B correctness side note:

- the A3B `Q4_K_M` exactness failure is now narrowed beyond quant quality alone:
  - promoting `blk.40.nextn.eh_proj.weight` from `Q4_K` to `Q5_K` was the right balanced GGUF fix, but it did not clear `bad np=1`
  - disabling the greedy accept fast path also did not clear it
  - tracing showed the model is on the hybrid recurrent-backup restore path and that the first replayed verifier logits still match baseline
  - exactness came back when the first speculative step after replay was skipped once
- that diagnosis is now codified as the current conservative fix:
  - the retained post-replay plain-step guard now applies to recurrent models and `qwen35moe`
  - this restores the `np=1` lossless contract on the checked A3B `Q4_K_M` cases while letting dense `qwen35` recover its speed-positive path
  - the deeper follow-up, if we want it, is still to explain why the first speculative verifier batch after replay is not baseline-equivalent on the MoE path and then remove or relax the guard

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

Approach that was kept:

- keep the deterministic one-step post-replay plain-step guard on:
  - recurrent models
  - `qwen35moe`
- do not apply that guard to dense `qwen35`
- keep the rule profile-visible
- do not build an adaptive controller

Status:

- replay policy still matters
- the broad hybrid-level guard was too expensive for dense `qwen35`
- narrowing it to `qwen35moe` / recurrent paths was the right dense-only V1 triage move
- that recovered the target `9B q8_0 np=1` win without reopening the checked A3B exactness failure

### 5b. Landed: Replay-Guard Narrowing Triage

This was the one remaining narrow runtime branch worth trying before stopping.

Target:

- recover the lost dense `9B q8_0 np=1` win if possible
- without reopening a large hybrid-state redesign
- without weakening the checked `np=1` exactness contract

Hypothesis:

- the current broad replay guard is likely the main reason `9B q8_0` regressed from the earlier pre-guard speed-positive state
- dense `Qwen3.5` may not need the same guard scope as the failing A3B replay path

Implemented change:

- narrow the post-replay guard based on model family:
  - retained on `qwen35moe`
  - removed from dense `qwen35`
- leave recurrent-backup state handling intact
- do not broaden public API surface

Minimum validation matrix:

- active speed target:
  - `Qwen3.5-9B q8_0`: `primary`, `good`, `bad`; `np=1,2`; `repeat=3`
- supporting dense regression checks:
  - `Qwen3.5-9B UD-Q4_K_XL`: `primary`, `good`, `bad`; `np=1,2`; `repeat=3`
  - `Qwen3.5-27B UD-Q4_K_XL`: `primary`, `good`, `bad`; `np=1,2`; `repeat=3`
- regression-only correctness smoke:
  - `Qwen3.5-35B-A3B Q4_K_M`: `primary`, `good`, `bad`; `np=1`; `repeat=1`

Outcome:

- `9B q8_0` regained a clear repeated `np=1` win:
  - `primary np=1`: `0.999x -> 1.081x`
  - `good np=1`: `0.938x -> 1.032x`
- `9B UD-Q4_K_XL` improved, but stayed mixed:
  - `primary np=1`: `0.904x -> 0.957x`
  - `good np=1`: `0.857x -> 0.931x`
  - `primary np=2`: `0.904x -> 1.006x`
- `27B UD-Q4_K_XL` did not become viable:
  - `primary np=1`: `0.861x -> 0.826x`
  - `good np=1`: `0.867x -> 0.964x`
  - `bad np=2`: `0.914x -> 0.556x`
- the checked A3B `Q4_K_M` smoke stayed exact with guard hits visible on replayed steps

Reading:

- keep this narrowing for the dense-only V1 branch
- do not treat the result as evidence that Q4 or 27B are suddenly good speed targets
- the dense speed story is still basically `9B q8_0`

### 5c. Stop Condition After Visibility

If all of the following remain true after the visibility pass:

- pure-fast-path verifier coverage is already near-saturated
- logits are already suppressed on almost all speculative accept rows
- the current replay guard choice is already benchmarked on the dense path
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
