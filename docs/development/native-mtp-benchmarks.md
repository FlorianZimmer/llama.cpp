# Native MTP Benchmarks

Bench date: 2026-04-10

This note records the current dense-only benchmark state for native MTP on the prepared Qwen 3.5 GGUFs under `/mnt/models`.
This benchmark line is now paused after the current dense-only branch result.

## Benchmark Protocol

- compare native MTP against greedy baseline on the same model and prompt
- require exact `np=1` output match to greedy baseline
- treat `np>1` as stability-only on the current hybrid path
- judge changes by repeated end-to-end tok/s, not internal timing alone
- keep `LLAMA_SERVER_MTP_PROFILE=1` available for per-step timing and acceptance

Authoritative harness:

- [scripts/validate_mtp_cuda.py](../../scripts/validate_mtp_cuda.py)

## Method

Backend:

- CUDA only on this host
- GPU: RTX 5090 32 GiB
- binary: `build-cuda-server/bin/llama-server`

Common settings:

- `ctx-size=4096`
- `batch-size=128`
- `ubatch-size=128`
- `threads=4`
- `threads-batch=4`
- `-ngl all`
- `-fa on`
- `draft-max=1`
- repeats: `3`
- cases: `primary`, `good`, `bad`
- `-np`: `1`, `2`

Cases:

- `primary`: `Write one short sentence about Berlin.`, seed `42`, `n_predict=12`
- `good`: `Write two short sentences about the Moon.`, seed `31415`, `n_predict=64`
- `bad`: `List three reasons Rust is used for systems programming.`, seed `777`, `n_predict=64`

Benchmarked models:

- `/mnt/models/GGUF/Qwen3.5-9B-MTP-UD-Q4_K_XL.gguf`
- `/mnt/models/GGUF/Qwen3.5-9B-MTP-q8_0.gguf`
- `/mnt/models/GGUF/Qwen3.5-27B-MTP-UD-Q4_K_XL.gguf`

## Headline Result

- `Qwen3.5-9B q8_0` is the only checked dense path with a meaningful `np=1` win:
  - `primary np=1`: `150.36 -> 170.02 tok/s` (`1.131x`)
  - `good np=1`: `148.65 -> 155.14 tok/s` (`1.044x`)
  - `bad np=1`: `148.66 -> 149.06 tok/s` (`1.003x`)
- `Qwen3.5-9B UD-Q4_K_XL` improved versus the previous native-MTP branch, but is still not a reliable broad speed-positive target:
  - `primary np=1`: `176.82 -> 177.07 tok/s` (`1.001x`)
  - `good np=1`: `195.84 -> 185.13 tok/s` (`0.945x`)
  - `bad np=1`: `195.79 -> 184.19 tok/s` (`0.941x`)
- `Qwen3.5-27B UD-Q4_K_XL` also improved versus the previous native-MTP branch, but remains regression-only for dense V1:
  - `primary np=1`: `72.22 -> 60.41 tok/s` (`0.837x`)
  - `good np=1`: `70.94 -> 68.98 tok/s` (`0.972x`)
  - `bad np=1`: `70.92 -> 66.32 tok/s` (`0.935x`)

## Dense V1 Branch Result

Artifacts:

- `/tmp/native-mtp-next/step02/qwen35-9b-q8_0.json`
- `/tmp/native-mtp-next/step02/qwen35-9b-ud-q4.json`
- `/tmp/native-mtp-next/step02/qwen35-27b-ud-q4.json`
- `/tmp/native-mtp-next/step02/qwen35-9b-q8_0-np2.json` (`np=2` stability-only)

| Model | `-np` | Primary | Good | Bad | Reading |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen3.5-9B Q8_0 | 1 | `1.131x` | `1.044x` | `1.003x` | active speed target; beats previous MTP result |
| Qwen3.5-9B Q8_0 | 2 | `1.261x` | `0.576x` | `0.778x` | stability-only; known divergence still dominates harder prompts |
| Qwen3.5-9B UD-Q4_K_XL | 1 | `1.001x` | `0.945x` | `0.941x` | supporting regression coverage; improved vs previous MTP |
| Qwen3.5-27B UD-Q4_K_XL | 1 | `0.837x` | `0.972x` | `0.935x` | regression-only; improved vs previous MTP |

## Kept Step

The only new kept runtime change in this bench set is a qwen35-local native-MTP draft-graph specialization in `src/models/qwen35.cpp`:

- exact one-token no-cache attention collapse for the current native-MTP contract
- keep gate + V + output projection
- skip `wk`, q/k norm, RoPE, and generic attention on that path
- keep the generic path for any wider batch shape

This step stayed exact at `np=1`, improved `Qwen3.5-9B q8_0` against both greedy baseline and the immediately previous MTP JSON, and improved both supporting dense regression models relative to the previous MTP JSON.

## Broad-Guard Regression And Recovery

The most important dense regression in this branch was the broad post-replay guard.

Same-harness comparison:

- pre-guard:
  - `/tmp/native-mtp-bench-20260410/qwen35-9b-q8_0.json`
  - `primary np=1`: `1.082x`
  - `good np=1`: `1.030x`
- broad-guard:
  - `/tmp/native-mtp-bench-20260410-post-replay-guard/qwen35-9b-q8_0.json`
  - `primary np=1`: `0.999x`
  - `good np=1`: `0.938x`
- current dense-only branch:
  - `/tmp/native-mtp-v1-triage/qwen35-9b-q8_0.json`
  - `primary np=1`: `1.081x`
  - `good np=1`: `1.032x`
- current qwen35 single-token specialization:
  - `/tmp/native-mtp-next/step02/qwen35-9b-q8_0.json`
  - `primary np=1`: `1.131x`
  - `good np=1`: `1.044x`

Interpretation:

- the branch did suffer a real dense regression
- that regression was caused by applying a conservative replay guard too broadly
- removing that guard from dense `qwen35` recovered the earlier `9B q8_0` win
- the kept qwen35-local specialization then widened that recovered `9B q8_0` win further

## Step Visibility Result

The per-step visibility pass ruled out an easy remaining server-local win.

Representative coverage from `/tmp/native-mtp-step-01`:

- `9B q8_0 primary np=1`: `15/15` pure-fast-path, `15/15` logits-suppressed
- `9B UD-Q4 good np=2`: `180/186` pure-fast-path, `180/186` logits-suppressed
- `27B UD-Q4 bad np=2`: `180/186` pure-fast-path, `180/186` logits-suppressed

So the current dense shortfall is not mainly “we are still missing the verifier fast path”.

The kept qwen35-local step preserved that visibility profile on the new `np=1` runs:

- `9B q8_0 primary np=1`: `6/6` pure-fast-path, `6/6` logits-suppressed
- `9B UD-Q4 primary np=1`: `6/6` pure-fast-path, `6/6` logits-suppressed
- `27B UD-Q4 primary np=1`: `7/7` pure-fast-path, `7/7` logits-suppressed

## Current Conclusions

- dense native MTP is worth carrying today on `Qwen3.5-9B q8_0`
- dense `Q4` improved meaningfully versus the previous MTP branch, but still does not justify a broad feature claim
- `27B` remains too expensive for the current one-token runtime design even after the local qwen35 step
- the current dense-only branch is therefore paused in this state rather than extended with more small local heuristics
- this likely exhausts the remaining low-risk dense-only qwen35 branch:
  - if a future local step cannot beat `/tmp/native-mtp-next/step02/qwen35-9b-q8_0.json`, the remaining ceiling should be treated as structural

## Why The Result Is Limited

The benchmark outcome is now clear:

- the implementation works and stays exact at `np=1` on the checked dense path
- the speedup is narrow rather than broad
- the remaining gap is not explained by an obvious missed fast path or a pending dense quantization fix

The main blockers are structural:

- the current runtime drafts only one continuation token per verifier step
- speculative state is still managed with restore / replay style behavior instead of explicit branch-state storage
- that leaves too little amortization on heavier dense targets such as `27B UD-Q4_K_XL`

This is also why parity with speculative runtimes such as vLLM or SGLang is not a realistic expectation for the current branch shape.

## Historical Note

`qwen35moe` / `Qwen3.5-35B-A3B` were removed from this V1 prep branch after local experiments showed:

- extra quantization rescue work was required just to restore exactness
- a conservative replay guard was needed on the MoE path
- throughput stayed materially below baseline even after those fixes

That history is still useful for review, but it is no longer part of the live dense-only V1 benchmark set.
