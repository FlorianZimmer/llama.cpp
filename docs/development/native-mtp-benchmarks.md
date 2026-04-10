# Native MTP Benchmarks

Bench date: 2026-04-10

This note records the current dense-only benchmark state for native MTP on the prepared Qwen 3.5 GGUFs under `/mnt/models`.

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
  - `primary np=1`: `150.94 -> 163.10 tok/s` (`1.081x`)
  - `good np=1`: `148.90 -> 153.65 tok/s` (`1.032x`)
  - `bad np=1`: `148.89 -> 147.56 tok/s` (`0.991x`)
- `Qwen3.5-9B UD-Q4_K_XL` improved after the dense guard narrowing, but is still not a reliable speed-positive target:
  - `primary np=1`: `175.39 -> 167.94 tok/s` (`0.957x`)
  - `good np=1`: `196.31 -> 182.85 tok/s` (`0.931x`)
  - `primary np=2`: `156.34 -> 157.28 tok/s` (`1.006x`)
- `Qwen3.5-27B UD-Q4_K_XL` remains clearly negative and is regression-only for dense V1:
  - `primary np=1`: `72.26 -> 59.71 tok/s` (`0.826x`)
  - `good np=1`: `70.96 -> 68.41 tok/s` (`0.964x`)
  - `bad np=2`: `59.17 -> 32.92 tok/s` (`0.556x`)

## Dense V1 Branch Result

Artifacts:

- `/tmp/native-mtp-v1-triage/qwen35-9b-q8_0.json`
- `/tmp/native-mtp-v1-triage/qwen35-9b-ud-q4.json`
- `/tmp/native-mtp-v1-triage/qwen35-27b-ud-q4.json`

| Model | `-np` | Primary | Good | Bad | Reading |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen3.5-9B Q8_0 | 1 | `1.081x` | `1.032x` | `0.991x` | only active speed target |
| Qwen3.5-9B Q8_0 | 2 | `1.033x` | `1.104x` | `0.998x` | favorable only on easier prompts |
| Qwen3.5-9B UD-Q4_K_XL | 1 | `0.957x` | `0.931x` | `0.934x` | supporting regression coverage |
| Qwen3.5-9B UD-Q4_K_XL | 2 | `1.006x` | `0.694x` | `0.993x` | inconsistent |
| Qwen3.5-27B UD-Q4_K_XL | 1 | `0.826x` | `0.964x` | `0.929x` | regression-only |
| Qwen3.5-27B UD-Q4_K_XL | 2 | `0.548x` | `0.515x` | `0.556x` | clearly not viable |

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

Interpretation:

- the branch did suffer a real dense regression
- that regression was caused by applying a conservative replay guard too broadly
- removing that guard from dense `qwen35` recovered the earlier `9B q8_0` win

## Step Visibility Result

The per-step visibility pass ruled out an easy remaining server-local win.

Representative coverage from `/tmp/native-mtp-step-01`:

- `9B q8_0 primary np=1`: `15/15` pure-fast-path, `15/15` logits-suppressed
- `9B UD-Q4 good np=2`: `180/186` pure-fast-path, `180/186` logits-suppressed
- `27B UD-Q4 bad np=2`: `180/186` pure-fast-path, `180/186` logits-suppressed

So the current dense shortfall is not mainly “we are still missing the verifier fast path”.

## Current Conclusions

- dense native MTP is only worth carrying today on `Qwen3.5-9B q8_0`
- dense `Q4` still does not show the consistent `np=1` single-user win needed for a broad feature claim
- `27B` remains too expensive for the current one-token runtime design
- the next dense-only branch should be judged against a clear maintenance bar:
  - if it cannot produce a repeatable `>= 5%` `np=1` win on `9B q8_0`, the current design is probably not worth carrying upstream as a speed feature

## Historical Note

`qwen35moe` / `Qwen3.5-35B-A3B` were removed from this V1 prep branch after local experiments showed:

- extra quantization rescue work was required just to restore exactness
- a conservative replay guard was needed on the MoE path
- throughput stayed materially below baseline even after those fixes

That history is still useful for review, but it is no longer part of the live dense-only V1 benchmark set.
