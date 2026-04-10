# Native MTP Benchmarks

## Benchmark Protocol

The native-MTP optimization path should be benchmark-gated by end-to-end tok/s, not by internal phase timing alone.

Required gate for any kept optimization:

- compare against greedy baseline on the same model and case
- compare against the immediately previous native-MTP step
- require exact `np=1` output match to greedy baseline
- treat `np>1` on hybrid/recurrent native-MTP as stability-only, not strict exactness
- ignore sub-1% movement unless it repeats consistently

Current dense-CUDA primary gate:

- model: `/mnt/models/GGUF/Qwen3.5-9B-MTP-UD-Q4_K_XL.gguf`
- case: `primary`
- prompt: `Write one short sentence about Berlin.`
- seed: `42`
- `n_predict=12`
- `-np 1`
- `draft-max=1`
- `threads=4`
- `threads-batch=4`

Supporting CUDA gates:

- `good`: Moon exact/stable case
- `bad`: Rust replay-heavy stability case

The authoritative harness for this protocol is [scripts/validate_mtp_cuda.py](../../scripts/validate_mtp_cuda.py), which now supports repeated runs, structured JSON output, optional profile parsing, and relaxed `np>1` validation for the documented hybrid/recurrent limitation.

## Current Optimization Status

Dense 9B CUDA optimization steps on top of the upstream-prep native-MTP path:

| Step | Change | Primary `np=1` result | Status |
| --- | --- | --- | --- |
| 0 | Benchmark gate only | baseline `175.31`, mtp `162.54` | reference |
| 1 | Greedy verifier-accept fast path | baseline `175.55`, mtp `164.12` | kept |
| 2 | Dedicated MTP scheduler/result cache | no repeatable win | dropped |
| 3 | Skip raw-logit downloads on token-only greedy verifier chunks | baseline `171.32`, mtp `167.04` on the primary gate; `np=2` short primary about break-even/slightly positive | kept |

Notes:

- Step 1 reduced accept-path overhead by bypassing the generic sampler path when direct greedy verifier tokens were already available.
- The dedicated MTP scheduler experiment did not materially move the draft bucket under repeated measurement, so it was not kept.
- Step 3 keeps the change generic: the server can disable raw-logit output only for decode chunks that are fully covered by the token-only native-MTP greedy path.

Bench date: 2026-04-09

Model:

- `/mnt/models/GGUF/Qwen3.5-9B-MTP-q8_0.gguf`

Commit used for the code under test:

- `01de729d4` (`native MTP: document np>1 exactness limits`)

Backend availability on this host:

- Benchmarked: CPU, CUDA
- Not benchmarked here: Vulkan, HIP, SYCL, Metal

Those other backends were not built or not available on this Linux machine, so the numbers below only cover the backends that were actually runnable in this environment.

## PR-Ready Summary

Native MTP for Qwen 3.5 is now wired end-to-end in llama.cpp: HF to GGUF conversion, model loading, runtime execution in a single verifier context, server integration, tests, and benchmark coverage. The current implementation is validated on the exact `-np 1` path and on some `-np 2` workloads, with measured speedups on both CPU and CUDA for exact cases.

The main known limitation is still the documented hybrid/recurrent `-np > 1` exactness gap. On near-tie tokens, verifier numerics can change with batch shape across multiple live sequences, so native `mtp` can diverge from baseline greedy decode even when rollback and replay logic restore the correct model state after rejected drafts. That limitation is documented rather than hidden behind an incorrect exactness guarantee.

On this host and model, CPU exact cases landed around `1.20x` to `1.53x`, while CUDA exact cases landed around `1.02x` to `1.23x`. The representative stress case remains CUDA Rust `np=2`, which still demonstrates the known exactness limitation and can also be slightly slower than baseline.

## Method

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
- The current native-MTP runtime is intentionally single-step even when GGUF metadata can represent more than one predictor layer. Extending it to recursive multi-step drafting should be possible within the current architecture, but it is follow-up runtime/model work rather than something already proven by the current branch.

## Qwen3.5-35B-A3B MoE Check

The native MTP path also works functionally on `Qwen3.5-35B-A3B`, but it did not show a speedup on this host after fixing the GGUF conversion bug in the shared MTP RMSNorm tensors.

Checked models:

- `/mnt/models/GGUF/Qwen3.5-35B-A3B-MTP-bf16-fixed.gguf`
- `/mnt/models/GGUF/Qwen3.5-35B-A3B-MTP-Q4_K_M-fixed.gguf`
- `/mnt/models/GGUF/Qwen3.5-35B-A3B-MTP-Q5_K_M-fixed.gguf`
- `/mnt/models/GGUF/Qwen3.5-35B-A3B-MTP-UD-Q4_K_XL.gguf`

Representative CUDA results:

| Case | `-np` | Baseline tok/s | MTP tok/s | Draft / accepted | Outcome |
| --- | ---: | --- | --- | --- | --- |
| Berlin, Q5_K_M fixed, `n_predict=48` | 1 | `234.11` | `172.16` | `15 / 13` | correct, slower |
| Moon, Q5_K_M fixed, `n_predict=48` | 1 | `234.60` | `179.24` | `13 / 11` | correct, slower |
| Berlin, UD Q4_K_XL, `n_predict=12` | 1 | `200.75` | `124.36` | `7 / 4` | correct, slower |

Interpretation:

- this is no longer a GGUF metadata/tensor preservation issue;
- the zero-acceptance cliff was caused by the converter bug and disappeared after the fix;
- the remaining slowdown appears to be a runtime economics issue: `Qwen3.5-35B-A3B` is already a fast active-parameter MoE model on this GPU, and with only one native predictor layer the saved verifier work is too small to amortize the extra draft + accept path cost;
- so the MoE path is still worth keeping as functionality, but this model should not currently be presented as a speed-positive native-MTP case.

## Speedup Backlog

The items below are the remaining upstream-friendly performance ideas worth tracking after the current scratch-storage cleanup. The expected gains are rough ranges from the current CUDA profile on this host, not guarantees.

### Priority 1: Backend-resident MTP seed path

- expected gain: roughly `+5%` to `+15%` on good CUDA workloads if implemented cleanly
- scope: medium to high
- reason: the native MTP path still copies the verifier hidden row back to host memory and then uploads it again as the MTP seed input
- current status: explored, but not landed; the first view-based device-copy prototype was not stable enough for `np=1/2`, so the current tree keeps the safer host-backed path

### Priority 2: Adaptive native-MTP backoff on replay-heavy prompts

- expected gain: large on bad prompts, little or no change on good prompts
- scope: medium
- reason: once snapshotting was removed, replay became the dominant bad-case overhead on rejection-heavy prompts like the Rust stress case
- risk: any adaptive policy has to stay understandable and upstream-friendly; it should not silently trade correctness or make the server behavior hard to reason about

### Priority 3: Replay-path reduction

- expected gain: modest on exact/easy prompts, more meaningful on prompts with lower draft acceptance
- scope: medium to high
- reason: replay is now the second largest remaining native-MTP overhead on many CUDA runs
- examples: cheaper replay batching, less redundant verifier work after rejection, or avoiding replay entirely on cases where the state can be proven equivalent

### Priority 4: Small server hot-path cleanup

- expected gain: low
- scope: small
- reason: there are still per-step container/setup costs in the server loop, but the current profile says they are not the main bottleneck anymore

## Raw Logs

Representative logs from the full matrix are under:

- `/tmp/mtp-bench-cpu-*`
- `/tmp/mtp-bench-cuda-*`
- summary JSON: `/tmp/mtp-bench-results.json`

## Related Notes

For model selection, GGUF preservation checks, quantization guidance, and a reusable external-AI prep prompt, see:

- [native-mtp-model-prep.md](native-mtp-model-prep.md)
