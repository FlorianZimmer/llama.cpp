# Native MTP Benchmarks

Bench date: 2026-04-09

Status note (2026-04-11):

This branch is kept as historical context.
The current fork-level native-MTP line is now paused.
The implementation works, the kept dense `np=1` path remains exact, and the latest dense-only branch recovered a real win on `Qwen3.5-9B q8_0`, but broader dense speedups did not survive across the checked targets and the remaining gap now looks structural rather than local.

Model:

- `/mnt/models/GGUF/Qwen3.5-9B-MTP-q8_0.gguf`

Commit used for the code under test:

- `01de729d4` (`native MTP: document np>1 exactness limits`)

Backend seed transport update on 2026-04-10:

- Branch under test: `feat/native-mtp-backend-seed`
- Change: native-MTP seed transport can now keep verifier seed rows on the backend for CUDA/non-host paths instead of doing backend -> host -> backend each step
- Validation status:
  - exact and measured: CUDA Berlin `np=1/2`, CUDA Moon `np=1/2`
  - known stress case unchanged: CUDA Rust `np=2` still demonstrates the documented hybrid/recurrent `np>1` exactness limitation
- rollout/debug aids kept in-tree:
  - `LLAMA_MTP_BACKEND_SEED_DEBUG=1` mirrors host seed rows and verifies backend cache/batch rows against them after synchronize
  - `LLAMA_MTP_BACKEND_SEED_FORCE_HOST=1` is only for comparison/debugging; on non-host multi-sequence CUDA it reproduces the older host round-trip behavior and is not the validated fast path

Backend availability on this host:

- Benchmarked: CPU, CUDA
- Not benchmarked here: Vulkan, HIP, SYCL, Metal

Those other backends were not built or not available on this Linux machine, so the numbers below only cover the backends that were actually runnable in this environment.

## PR-Ready Summary

Native MTP for Qwen 3.5 is now wired end-to-end in llama.cpp: HF to GGUF conversion, model loading, runtime execution in a single verifier context, server integration, tests, and benchmark coverage. The current implementation is validated on the exact `-np 1` path and on some `-np 2` workloads, with measured speedups on both CPU and CUDA for exact cases.

The main known limitation is still the documented hybrid/recurrent `-np > 1` exactness gap. On near-tie tokens, verifier numerics can change with batch shape across multiple live sequences, so native `mtp` can diverge from baseline greedy decode even when rollback and replay logic restore the correct model state after rejected drafts. That limitation is documented rather than hidden behind an incorrect exactness guarantee.

On this host and model, CPU exact cases landed around `1.20x` to `1.53x`, while CUDA exact cases landed around `1.02x` to `1.23x`. The representative stress case remains CUDA Rust `np=2`, which still demonstrates the known exactness limitation and can also be slightly slower than baseline.

With the backend-resident seed transport enabled on CUDA, the exact Berlin and Moon cases remain exact while avoiding the old host round trip. On the validated Berlin `np=2` case, native MTP measured `160.46 / 160.04 tok/s` versus baseline `129.64 / 133.66 tok/s`. On the validated Moon `np=2` case, native MTP measured `139.45 / 139.75 tok/s` versus baseline `130.41 / 133.00 tok/s`.

## Method

## Benchmark Protocol

Every default-on native-MTP performance change should be measured against:

- greedy baseline on the same binary
- the immediately previous landed native-MTP step
- Berlin exact at `np=1`, stability-only at `np=2`: seed `42`, `n_predict=48`
- Moon exact at `np=1`, stability-only at `np=2`: seed `31415`, `n_predict=64`
- Rust stress: `np=2`, seed `777`, `n_predict=64`

Practical rules:

- keep Berlin and Moon exact for `np=1`
- treat Berlin, Moon, and Rust at `np > 1` as stability-only cases on this hybrid/recurrent native-MTP CUDA path unless batch-invariance work changes the contract
- judge success by end-to-end tok/s, not by lower internal `t_*` counters alone
- treat sub-`1%` movement as noise unless it repeats clearly across reruns
- drop or park a candidate if it only helps the Rust stress case or only moves micro-profile timings

The validation harness for this protocol now supports:

- `--repeat` for repeated scenario runs
- `--json-out` for machine-readable baseline/current comparisons
- `--allow-known-np2-divergence` for any `np > 1` stability-only run on this path, including baseline `np=2`

Common settings:

- `ctx-size=4096`
- `batch-size=128`
- `ubatch-size=128`
- `draft-max=1`
- baseline = normal greedy decode
- mtp = `--spec-type mtp`

Backend-specific settings:

- CPU:
  - binary: `build-server/bin/llama-server`
  - `-ngl 0`
  - `-fa off`
  - chosen config: `threads=32`, `threads-batch=32`
- CUDA:
  - binary: `build-cuda-server/bin/llama-server`
  - `-ngl all`
  - `-fa on`
  - chosen config: `threads=4`, `threads-batch=4`

The chosen per-backend configs came from a small tuning pass on the exact Berlin `np=2` case.

## Tuning Pass

CPU, Berlin, `np=2`, `n_predict=48`:

- `t=8 tb=8`: exact, baseline `6.40 / 6.40 tok/s`, mtp `9.56 / 9.56 tok/s`
- `t=16 tb=16`: not exact, baseline `8.62 / 8.62 tok/s`, mtp `10.85 / 10.51 tok/s`
- `t=32 tb=32`: exact, baseline `7.56 / 7.70 tok/s`, mtp `11.41 / 11.41 tok/s`

CUDA, Berlin, `np=2`, `n_predict=48`:

- `t=4 tb=4`: exact, baseline `127.96 / 132.01 tok/s`, mtp `160.56 / 161.00 tok/s`
- `t=8 tb=8`: exact, baseline `128.09 / 132.20 tok/s`, mtp `130.84 / 136.37 tok/s`
- `t=12 tb=12`: exact, baseline `131.72 / 131.42 tok/s`, mtp `133.20 / 138.94 tok/s`

## Full Matrix

### CPU

Config:

- `build-server/bin/llama-server`
- `-ngl 0`
- `-fa off`
- `threads=32`
- `threads-batch=32`

| Case | `-np` | Exact | Baseline tok/s | MTP tok/s | Speedup |
| --- | ---: | --- | --- | --- | --- |
| Berlin, seed 42, `n_predict=48` | 1 | yes | `8.08` | `12.40` | `1.53x` |
| Berlin, seed 42, `n_predict=48` | 2 | yes | `7.77 / 7.77` | `11.22 / 11.22` | `1.44x` mean |
| Moon, seed 31415, `n_predict=64` | 1 | yes | `7.61` | `10.20` | `1.34x` |
| Moon, seed 31415, `n_predict=64` | 2 | yes | `7.55 / 7.55` | `9.79 / 9.80` | `1.30x` mean |
| Rust, seed 777, `n_predict=64` | 1 | yes | `8.28` | `9.96` | `1.20x` |
| Rust, seed 777, `n_predict=64` | 2 | yes on this run | `7.49 / 7.49` | `9.31 / 9.31` | `1.24x` mean |

Observed CPU range:

- exact cases measured here: `1.20x` to `1.53x`

### CUDA

Config:

- `build-cuda-server/bin/llama-server`
- `-ngl all`
- `-fa on`
- `threads=4`
- `threads-batch=4`

| Case | `-np` | Exact | Baseline tok/s | MTP tok/s | Speedup |
| --- | ---: | --- | --- | --- | --- |
| Berlin, seed 42, `n_predict=48` | 1 | yes | `148.69` | `173.37` | `1.17x` |
| Berlin, seed 42, `n_predict=48` | 2 | yes | `128.29 / 132.26` | `160.44 / 160.88` | `1.23x` mean |
| Moon, seed 31415, `n_predict=64` | 1 | yes | `148.65` | `151.79` | `1.02x` |
| Moon, seed 31415, `n_predict=64` | 2 | yes | `131.10 / 131.32` | `141.07 / 140.74` | `1.07x` mean |
| Rust, seed 777, `n_predict=64` | 1 | yes | `148.45` | `146.60` | `0.99x` |
| Rust, seed 777, `n_predict=64` | 2 | no | `128.61 / 131.33` | `127.63 / 127.84` | `0.98x` mean |

Observed CUDA range:

- exact cases measured here: `1.02x` to `1.23x`
- known stress failure here: Rust, `np=2`

## Takeaways

- CPU showed the largest upside on this host for the selected exact configurations, roughly `+20%` to `+53%`.
- CUDA improved clearly on the short/medium exact cases, but the gain was smaller, roughly `+2%` to `+23%`.
- The known hybrid/recurrent `np > 1` exactness limitation is still real on CUDA. The Rust `np=2` case remained a representative failure and was also slightly slower than baseline.
- On exact workloads, native MTP is already useful on both CPU and CUDA with this model, but the benefit is workload-dependent and should not be treated as a universal speedup.

## Speedup Backlog

The items below are the remaining upstream-friendly performance ideas worth tracking after the current scratch-storage cleanup and backend-resident seed transport. The expected gains are rough ranges from the current CUDA profile on this host, not guarantees.

### Completed: Backend-resident MTP seed path

- landed on `feat/native-mtp-backend-seed`
- result: validated exact CUDA path for Berlin `np=1/2` and Moon `np=1/2` without the backend -> host -> backend seed round trip
- design that landed:
  - persistent backend-owned `seed_cache_dev` and `seed_batch_dev`
  - fixed-offset graph-visible backend view for `build_inp_mtp_seed()`
  - explicit seed mode and generation-based graph reuse invalidation
  - conservative fallback that keeps the host path for host-backed and single-sequence cases, while avoiding the old multi-sequence non-host host-round-trip path by default

### Priority 1: Adaptive native-MTP backoff on replay-heavy prompts

- expected gain: large on bad prompts, little or no change on good prompts
- scope: medium
- reason: once snapshotting was removed, replay became the dominant bad-case overhead on rejection-heavy prompts like the Rust stress case
- risk: any adaptive policy has to stay understandable and upstream-friendly; it should not silently trade correctness or make the server behavior hard to reason about
- current status:
  - tried and dropped
  - an env-gated prototype regressed the validated Berlin `np=1` CUDA case from about `172.5 tok/s` to about `151.8 tok/s`, so it was not kept even as a parked runtime option

### Priority 2: Replay-path reduction

- expected gain: modest on exact/easy prompts, more meaningful on prompts with lower draft acceptance
- scope: medium to high
- reason: replay is now the second largest remaining native-MTP overhead on many CUDA runs
- examples: cheaper replay batching, less redundant verifier work after rejection, or avoiding replay entirely on cases where the state can be proven equivalent
- current status:
  - partial bookkeeping cleanup landed
  - replay spans are now captured explicitly and replay scratch storage is reused
  - no clear exact-case end-to-end CUDA win has been measured yet, but the replay path is simpler and avoids repeated reconstruction from prompt state

### Priority 3: Small server hot-path cleanup

- expected gain: low
- scope: small
- reason: there are still per-step container/setup costs in the server loop, but the current profile says they are not the main bottleneck anymore
- current status:
  - partial cleanup landed
  - micro-timing is now actually gated by `LLAMA_SERVER_MTP_PROFILE`
  - batched native draft handoff no longer uses an `unordered_map`
  - exact-case gains have been flat/noisy so far

## Raw Logs

Representative logs from the full matrix are under:

- `/tmp/mtp-bench-cpu-*`
- `/tmp/mtp-bench-cuda-*`
- summary JSON: `/tmp/mtp-bench-results.json`
