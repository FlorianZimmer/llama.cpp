# Native MTP Optimization Plan

Date: 2026-04-10

This note tracks the remaining upstream-friendly native-MTP performance work after the backend-resident seed transport landed. The work here is intentionally benchmark-gated and only applies after the current exact CUDA baseline is stable again.

## Goal

Finish the three remaining native-MTP performance areas together while staying:

- upstream-friendly
- generic across future native-MTP models
- maintainable
- benchmark-driven instead of micro-profile-driven

The three remaining areas are:

1. adaptive backoff on replay-heavy workloads
2. cheaper rejection and replay handling
3. small server hot-path cleanup

## Constraints

- Berlin and Moon must remain exactly equal to greedy baseline for `np=1`.
- Berlin, Moon, and Rust at `np > 1` are stability-only cases on this hybrid/recurrent native-MTP CUDA path, not strict exactness targets.
- Success is judged by end-to-end tok/s on the validated exact CUDA cases, not by lower internal `t_*` counters alone.
- Every default-on change must be compared against:
  - greedy baseline
  - the immediately previous native-MTP step

## Benchmark Protocol

Use [scripts/validate_mtp_cuda.py](/home/florian/llama.cpp-upstream-mtp-plan/scripts/validate_mtp_cuda.py) with:

- `--repeat`
- `--json-out`
- `--allow-known-np2-divergence` for Rust stress runs only

Standard CUDA validation set:

- Berlin exact at `np=1`, stability-only at `np=2`: prompt `Write one short sentence about Berlin.`, seed `42`, `n_predict=48`
- Moon exact at `np=1`, stability-only at `np=2`: prompt `Write two short sentences about the Moon.`, seed `31415`, `n_predict=64`
- Rust stress: prompt `List three reasons Rust is used for systems programming.`, seed `777`, `n_predict=64`

Keep the current exact CUDA config unless there is a strong reason to retune it:

- `-ngl all`
- `-fa on`
- `ctx-size=4096`
- `batch-size=128`
- `ubatch-size=128`
- `threads=4`
- `threads-batch=4`
- `draft-max=1`

Practical scoring rules:

- treat sub-`1%` movement as noise unless it repeats clearly
- do not keep a change because it only improves Rust stress behavior
- do not keep a change because it only improves internal phase timings

## Main Implementation Area

Primary file:

- [tools/server/server-context.cpp](/home/florian/llama.cpp-upstream-mtp-plan/tools/server/server-context.cpp)

Secondary files:

- [scripts/validate_mtp_cuda.py](/home/florian/llama.cpp-upstream-mtp-plan/scripts/validate_mtp_cuda.py)
- [docs/development/native-mtp-benchmarks.md](/home/florian/llama.cpp-upstream-mtp-plan/docs/development/native-mtp-benchmarks.md)
- optionally [tools/server/tests/unit/test_speculative.py](/home/florian/llama.cpp-upstream-mtp-plan/tools/server/tests/unit/test_speculative.py) if a surviving runtime-visible toggle needs coverage

Avoid touching the runtime/backend layers again unless server-side work clearly underdelivers:

- `src/llama-context.cpp`
- `src/llama-mtp.cpp`
- `src/llama-mtp.h`
- `src/llama-graph.cpp`

## Phased Plan

### Step 1: Freeze the benchmark protocol

Scope: small

Status:

- landed in `scripts/validate_mtp_cuda.py`
- landed in `docs/development/native-mtp-benchmarks.md`

Purpose:

- make each optimization step benchmarkable against both greedy baseline and the previous native-MTP step
- allow `np > 1` stability runs without claiming strict exactness on hybrid/recurrent native-MTP CUDA

### Step 2: No-behavior hot-path cleanup

Scope: small

Expected payoff:

- low, but this is the safest place for an immediate exact-case gain

Targets:

- gate fine-grained native-MTP timing behind `LLAMA_SERVER_MTP_PROFILE`
- replace `std::unordered_map<int, llama_tokens> native_mtp_drafts` with bounded scratch storage
- preallocate and reuse obvious replay/draft temporary containers
- cache invariant per-slot native-MTP flags in slot setup/reset instead of recomputing them every round

Keep only the subchanges that either:

- simplify the code, or
- show a repeatable exact-case tok/s gain

### Step 3: Conservative adaptive native-MTP backoff

Scope: medium

Expected payoff:

- high on replay-heavy prompts
- modest on exact cases

Design:

- small per-slot `native_mtp_policy_state`
- dynamic draft cap
- cooldown and probe behavior
- generic signals only:
  - drafted tokens
  - accepted drafted tokens
  - replay occurrence
  - configured `native_mtp_max`

Do not:

- add Qwen-specific heuristics
- add public CLI surface unless the data later proves it is necessary

Default-on rule:

- only keep default-on if the `np=1` exact cases show a repeatable end-to-end gain, or if the `np>1` cases show a clear throughput gain without destabilizing output
- otherwise park behind an env or drop

### Step 4: Replay-path cleanup

Scope: medium

Expected payoff:

- low to moderate
- more likely to help rejection-heavy runs than perfect exact runs

Targets:

- stop reconstructing replay indirectly from `n_prompt_base` and `slot.prompt.tokens`
- capture replay spans explicitly during accept/rollback
- reuse replay batch storage instead of init/free each replay call
- keep decode order and token ordering identical to the current stepped replay path

Keep this only if it:

- stays exact, and
- either improves end-to-end tok/s or materially simplifies later work

### Step 5: Narrow packed replay fast path

Scope: medium to high

Expected payoff:

- uncertain on the current exact config
- more future-facing than required for today

Status:

- optional

Requirements:

- exact-equivalent token order
- full fallback to the current stepped replay path
- no scheduler/backend assumptions beyond what the current path already relies on

Drop this step immediately if:

- exactness gets delicate, or
- it only moves internal replay timings without end-to-end gain

### Step 6: Final cleanup

Scope: small

Targets:

- remove temporary tuning branches that did not pay off
- keep helpers small and generic
- reserve surviving per-slot vectors once at slot/task setup if they remain hot
- update benchmark documentation with final numbers and kept/dropped changes

## Validation Checklist

Required after every landed step:

- Berlin exact: `np=1`
- Moon exact: `np=1`
- Berlin and Moon `np=2`: stable non-empty outputs, no corruption, no invalid token stream
- draft activity still reported on the exact `np=1` cases
- Rust stress remains stable:
  - no crash
  - no corruption
  - no invalid token stream

Do not claim success from:

- lower `t_replay_us`, `t_accept_us`, or similar counters alone
- Rust-only wins with flat Berlin/Moon results
- one-off wins that disappear on repeat

## Current Blocker

There is no longer a blocking exactness prerequisite for `np > 1`. Repeated Berlin `np=2` validation showed the same near-tie divergence pattern already documented for Rust on this hybrid/recurrent native-MTP CUDA path. That means `np > 1` should be treated as a best-effort stability/performance mode rather than a strict exactness contract unless future batch-invariant backend work changes that.
