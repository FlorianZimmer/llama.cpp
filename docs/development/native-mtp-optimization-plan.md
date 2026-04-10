# Native MTP Optimization Plan

This branch is now explicitly scoped to dense Qwen 3.5 native MTP only.

## V1 Scope

- keep native-MTP support only for `qwen35`
- active speed target: `Qwen3.5-9B q8_0`
- supporting dense regression coverage:
  - `Qwen3.5-9B UD-Q4_K_XL`
  - `Qwen3.5-27B UD-Q4_K_XL`
- keep `np=1` exactness against greedy baseline
- treat `np>1` as stability-only on the current hybrid path

Removed from this V1 prep branch:

- `qwen35moe` native-MTP support
- `Qwen3.5-35B-A3B` as an active benchmark or optimization target
- MoE-specific quantization overrides and replay-guard policy from the live V1 path

## Current Dense Reading

Current dense rerun artifacts:

- `/tmp/native-mtp-next/step02/qwen35-9b-q8_0.json`
- `/tmp/native-mtp-next/step02/qwen35-9b-ud-q4.json`
- `/tmp/native-mtp-next/step02/qwen35-27b-ud-q4.json`
- `/tmp/native-mtp-next/step02/qwen35-9b-q8_0-np2.json` (`np=2` stability-only spot-check)

Current branch result:

- `Qwen3.5-9B q8_0`
  - `primary np=1`: `150.36 -> 170.02 tok/s` (`1.131x`)
  - `good np=1`: `148.65 -> 155.14 tok/s` (`1.044x`)
  - `bad np=1`: `148.66 -> 149.06 tok/s` (`1.003x`)
- `Qwen3.5-9B UD-Q4_K_XL`
  - `primary np=1`: `176.82 -> 177.07 tok/s` (`1.001x`)
  - `good np=1`: `195.84 -> 185.13 tok/s` (`0.945x`)
  - `bad np=1`: `195.79 -> 184.19 tok/s` (`0.941x`)
- `Qwen3.5-27B UD-Q4_K_XL`
  - `primary np=1`: `72.22 -> 60.41 tok/s` (`0.837x`)
  - `good np=1`: `70.94 -> 68.98 tok/s` (`0.972x`)
  - `bad np=1`: `70.92 -> 66.32 tok/s` (`0.935x`)

Interpretation:

- the kept qwen35-local step clears the dense V1 bar on `9B q8_0`
- `9B UD-Q4_K_XL` improved materially versus the previous MTP branch, but it is still not a broad speed-positive target
- `27B UD-Q4_K_XL` also improved versus the previous MTP branch, but it remains regression-only coverage
- the current one-token native-MTP design still has narrow upside; this step likely exhausts the remaining low-risk dense-only branch

## What Survived In This Step

Kept local runtime change:

- exact qwen35 single-token native-MTP no-cache attention specialization in `src/models/qwen35.cpp`
  - keep the gate path, V projection, and output projection
  - skip `wk`, q/k norm, RoPE, and generic attention when the graph contract is exactly one drafted token per sequence
  - keep the generic path for any wider no-cache batch shape
- stop creating dead position / no-cache-attention graph inputs for the exact one-token MTP graph shape

What we did not keep pursuing after this step survived:

- query-half pruning inside `wq`
- draft-logit output pruning

Reason:

- the local qwen35 step already met the `9B q8_0` `np=1` benchmark bar with a small diff, so widening the branch was not necessary for V1

## What The Branch Already Ruled Out

These items were already tried and should not be reopened as “easy next wins”:

- dedicated MTP scheduler / result cache
  - draft-bucket movement did not survive repeated end-to-end benchmarking
- broad hybrid-level post-replay plain-step guard
  - regressed dense `qwen35`, especially `9B q8_0`
- separate dense cooldown experiment
  - created no useful behavior beyond the broad guard and was dropped
- mixed-chunk verifier split as the next likely win
  - step visibility showed dense accept rows are already almost entirely pure fast-path verifier rows with logits suppressed
- MoE as a speed target
  - required correctness rescue work and still stayed materially speed-negative

## Important Regression / Recovery

Under the same newer harness:

- earlier dense result:
  - `/tmp/native-mtp-bench-20260410/qwen35-9b-q8_0.json`
  - `primary np=1`: `1.082x`
  - `good np=1`: `1.030x`
- broad-guard regression:
  - `/tmp/native-mtp-bench-20260410-post-replay-guard/qwen35-9b-q8_0.json`
  - `primary np=1`: `0.999x`
  - `good np=1`: `0.938x`
- current dense-only recovery:
  - `/tmp/native-mtp-v1-triage/qwen35-9b-q8_0.json`
  - `primary np=1`: `1.081x`
  - `good np=1`: `1.032x`
- current qwen35 single-token specialization:
  - `/tmp/native-mtp-next/step02/qwen35-9b-q8_0.json`
  - `primary np=1`: `1.131x`
  - `good np=1`: `1.044x`

So the branch first recovered the earlier dense `q8_0` win once the broad replay guard was removed from dense `qwen35`, then extended it with a local qwen35 draft-graph specialization.

## What We Already Tried On Hybrid-State Behavior

We already tried the shallow hybrid-state experiments:

- rollback / restore / replay hardening
- replay-logit tracing
- disabling the greedy accept fast path as an isolation test
- a one-step post-replay cooldown / plain-step guard
- broad guard on all hybrid Qwen 3.5 models
- narrowed guard to avoid regressing dense `qwen35`

We have not yet done the deeper runtime-state design used by systems like vLLM or SGLang:

- explicit speculative branch-state storage beyond the current restore/replay path
- deeper draft+verify+rewind integration inside libllama
- multi-token native drafting

That deeper work is outside this V1 prep branch.

## V1 Branch Decision

The dense-only question for this branch was:

- can dense `qwen35` achieve a repeatable `np=1` single-user win of at least about `1.05x` on `9B q8_0` without breaking exactness and without becoming quant-fragile?

Answer:

- yes, but narrowly
- the current one-token design is now clearly worth carrying for `Qwen3.5-9B q8_0`
- the same design is still not broad dense speed-positive support for `UD-Q4_K_XL` or `27B`

Practical stop condition from here:

- do not keep stacking more qwen35-local heuristics on this branch unless a new change can be justified as equally small and equally exact
- if a future local step does not beat `/tmp/native-mtp-next/step02/qwen35-9b-q8_0.json`, treat the remaining ceiling as structural and move deeper work to a dedicated follow-up branch

## Benchmark Gate

Keep using:

- [scripts/validate_mtp_cuda.py](../../scripts/validate_mtp_cuda.py)
- [native-mtp-benchmarks.md](native-mtp-benchmarks.md)

Every kept runtime change must beat:

- greedy baseline on the same dense model and prompt
- the immediately previous native-MTP result

Prefer removal over accumulation if a change does not survive the repeated median gate.
