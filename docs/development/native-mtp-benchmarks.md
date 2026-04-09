# Native MTP Benchmarks

Bench date: 2026-04-09

Model:

- `/mnt/models/GGUF/Qwen3.5-9B-MTP-q8_0.gguf`

Commit used for the code under test:

- `01de729d4` (`native MTP: document np>1 exactness limits`)

Backend availability on this host:

- Benchmarked: CPU, CUDA
- Not benchmarked here: Vulkan, HIP, SYCL, Metal

Those other backends were not built or not available on this Linux machine, so the numbers below only cover the backends that were actually runnable in this environment.

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

## Raw Logs

Representative logs from the full matrix are under:

- `/tmp/mtp-bench-cpu-*`
- `/tmp/mtp-bench-cuda-*`
- summary JSON: `/tmp/mtp-bench-results.json`
