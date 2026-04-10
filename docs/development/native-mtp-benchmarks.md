# Native MTP Benchmarks

Bench date: 2026-04-10

Status note (2026-04-11):

This branch is kept as historical context.
The current fork-level native-MTP line is now paused.
The implementation works, the kept dense `np=1` path remains exact, and the latest dense-only branch recovered a real win on `Qwen3.5-9B q8_0`, but broader dense speedups did not survive across the checked targets and the remaining gap now looks structural rather than local.

This note records the current end-to-end CUDA benchmark state for native MTP on the prepared Qwen 3.5 GGUFs under `/mnt/models`.

## Benchmark Protocol

- judge every change by end-to-end tok/s, not phase timing alone
- compare native MTP against greedy baseline on the same model and prompt
- require exact `np=1` output match to greedy baseline
- treat `np>1` as stability-only on the current hybrid/recurrent path
- ignore small movement unless it repeats across the 3-run median
- keep per-step native-MTP timing visible with `LLAMA_SERVER_MTP_PROFILE=1`

Authoritative harness:

- [scripts/validate_mtp_cuda.py](../../scripts/validate_mtp_cuda.py)

The harness now parses two native-MTP profile signals:

- aggregate phase totals from `native MTP profile: ...`
- per-step timing and acceptance from `native MTP step: ...`

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
- `/mnt/models/GGUF/Qwen3.5-35B-A3B-MTP-Q4_K_M-fixed.gguf`
- `/mnt/models/GGUF/Qwen3.5-35B-A3B-MTP-Q5_K_M-fixed.gguf`
- `/mnt/models/GGUF/Qwen3.5-35B-A3B-MTP-UD-Q4_K_XL-fixed.gguf`

## Headline Result

- This 3-repeat matrix includes the landed hybrid/recurrent post-replay guard fix.
- No checked model or quant is net-positive on `np=1` in the current 3-repeat median.
- `Qwen3.5-9B q8_0` is still the only checked path with repeatable net-positive wins, but only on the easier `np=2` cases:
  - `primary np=2`: `127.85 -> 133.26 tok/s` (`1.042x`)
  - `good np=2`: `132.66 -> 137.12 tok/s` (`1.034x`)
- `Qwen3.5-9B UD-Q4_K_XL` is slower everywhere in the current sweep.
- `Qwen3.5-27B UD-Q4_K_XL` is slower everywhere, though less catastrophically than in the pre-fix sweep.
- `Qwen3.5-35B-A3B` is now `np=1` exact across the checked quants, including the balanced `Q4_K_M` GGUF, but remains substantially slower than baseline.

## Current Scope Decision

Based on the current branch history and the post-guard regression, the recommended scope for the first upstream-oriented series is:

- active speed target:
  - `Qwen3.5-9B q8_0`
- supporting dense correctness / regression coverage:
  - `Qwen3.5-9B UD-Q4_K_XL`
  - `Qwen3.5-27B UD-Q4_K_XL`
- regression-only coverage:
  - `Qwen3.5-35B-A3B`, especially `Q4_K_M` `np=1`
- deferred from the first upstreamable series:
  - `qwen35moe` / `Qwen3.5-35B-A3B` as an active speed target

Why this is now the right scope:

- `Qwen3.5-9B q8_0` is still the only path that has ever shown meaningful CUDA wins in this branch
- `Qwen3.5-9B q8_0` also regressed materially after the permanent replay guard, under the same newer harness:
  - pre-guard file: `/tmp/native-mtp-bench-20260410/qwen35-9b-q8_0.json`
  - post-guard file: `/tmp/native-mtp-bench-20260410-post-replay-guard/qwen35-9b-q8_0.json`
  - `primary np=1`: `1.082x -> 0.999x`
  - `good np=1`: `1.030x -> 0.938x`
- the visibility pass showed that speculative accept rows are already almost fully on the greedy verifier fast path with logits suppressed
- so the remaining problem is no longer “recover an easy server fast path”; it is much closer to replay-guard / hybrid-state economics
- `Qwen3.5-35B-A3B` remains materially speed-negative even after the quant-quality rescue and replay-guard correctness fix

Interpretation:

- the current single-token native-MTP upside is real on the 9B Q8 path
- no checked path is currently an `np=1` end-to-end win
- once verifier economics get worse for the model or quant, draft + accept + replay overhead dominates quickly
- the larger dense and MoE cases are not blocked on “more benchmarking”; they are blocked on control-path economics and, for some quants, acceptance quality / exactness

## Full Matrix

### Primary

| Model | `-np` | Baseline tok/s | MTP tok/s | Speedup | `np=1` exact |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen3.5-9B UD-Q4_K_XL | 1 | `175.91` | `159.01` | `0.904x` | yes |
| Qwen3.5-9B UD-Q4_K_XL | 2 | `165.77` | `149.86` | `0.904x` | stability-only |
| Qwen3.5-9B Q8_0 | 1 | `150.53` | `150.31` | `0.999x` | yes |
| Qwen3.5-9B Q8_0 | 2 | `127.85` | `133.26` | `1.042x` | stability-only |
| Qwen3.5-27B UD-Q4_K_XL | 1 | `72.35` | `62.27` | `0.861x` | yes |
| Qwen3.5-27B UD-Q4_K_XL | 2 | `57.34` | `50.05` | `0.873x` | stability-only |
| Qwen3.5-35B-A3B Q4_K_M | 1 | `228.26` | `170.72` | `0.748x` | yes |
| Qwen3.5-35B-A3B Q4_K_M | 2 | `154.35` | `59.16` | `0.383x` | stability-only |
| Qwen3.5-35B-A3B Q5_K_M | 1 | `221.34` | `167.90` | `0.759x` | yes |
| Qwen3.5-35B-A3B Q5_K_M | 2 | `150.87` | `58.45` | `0.387x` | stability-only |
| Qwen3.5-35B-A3B UD-Q4_K_XL | 1 | `202.80` | `129.34` | `0.638x` | yes |
| Qwen3.5-35B-A3B UD-Q4_K_XL | 2 | `145.57` | `69.83` | `0.480x` | stability-only |

### Good

| Model | `-np` | Baseline tok/s | MTP tok/s | Speedup | `np=1` exact |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen3.5-9B UD-Q4_K_XL | 1 | `195.94` | `167.83` | `0.857x` | yes |
| Qwen3.5-9B UD-Q4_K_XL | 2 | `162.82` | `145.65` | `0.895x` | stability-only |
| Qwen3.5-9B Q8_0 | 1 | `148.62` | `139.40` | `0.938x` | yes |
| Qwen3.5-9B Q8_0 | 2 | `132.66` | `137.12` | `1.034x` | stability-only |
| Qwen3.5-27B UD-Q4_K_XL | 1 | `70.94` | `61.48` | `0.867x` | yes |
| Qwen3.5-27B UD-Q4_K_XL | 2 | `59.26` | `55.09` | `0.930x` | stability-only |
| Qwen3.5-35B-A3B Q4_K_M | 1 | `243.16` | `163.47` | `0.672x` | yes |
| Qwen3.5-35B-A3B Q4_K_M | 2 | `170.04` | `74.49` | `0.438x` | stability-only |
| Qwen3.5-35B-A3B Q5_K_M | 1 | `234.38` | `174.36` | `0.744x` | yes |
| Qwen3.5-35B-A3B Q5_K_M | 2 | `180.66` | `93.19` | `0.516x` | stability-only |
| Qwen3.5-35B-A3B UD-Q4_K_XL | 1 | `220.52` | `151.49` | `0.687x` | yes |
| Qwen3.5-35B-A3B UD-Q4_K_XL | 2 | `160.53` | `94.02` | `0.586x` | stability-only |

### Bad

| Model | `-np` | Baseline tok/s | MTP tok/s | Speedup | `np=1` exact |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen3.5-9B UD-Q4_K_XL | 1 | `195.39` | `162.72` | `0.833x` | yes |
| Qwen3.5-9B UD-Q4_K_XL | 2 | `162.59` | `133.78` | `0.823x` | stability-only |
| Qwen3.5-9B Q8_0 | 1 | `148.82` | `125.10` | `0.841x` | yes |
| Qwen3.5-9B Q8_0 | 2 | `132.99` | `115.83` | `0.871x` | stability-only |
| Qwen3.5-27B UD-Q4_K_XL | 1 | `70.87` | `63.97` | `0.903x` | yes |
| Qwen3.5-27B UD-Q4_K_XL | 2 | `58.64` | `53.61` | `0.914x` | stability-only |
| Qwen3.5-35B-A3B Q4_K_M | 1 | `251.70` | `177.53` | `0.705x` | yes |
| Qwen3.5-35B-A3B Q4_K_M | 2 | `190.39` | `108.20` | `0.568x` | stability-only |
| Qwen3.5-35B-A3B Q5_K_M | 1 | `243.17` | `174.54` | `0.718x` | yes |
| Qwen3.5-35B-A3B Q5_K_M | 2 | `186.98` | `111.72` | `0.597x` | stability-only |
| Qwen3.5-35B-A3B UD-Q4_K_XL | 1 | `224.48` | `174.48` | `0.777x` | yes |
| Qwen3.5-35B-A3B UD-Q4_K_XL | 2 | `167.94` | `128.60` | `0.766x` | stability-only |

## Per-Step Native-MTP Profile

These tables come from the new per-step `native MTP step:` profile parsing. Totals are aggregate MTP-only time across the 3 repeats for the given case and `-np`, while `mean step total us` is the average end-to-end cost of one speculative step after draft/snapshot/accept/restore/replay are combined.

### Primary `np=1`

| Model | Acceptance | Draft ms | Accept ms | Replay ms | Mean step total us |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-9B UD-Q4_K_XL | `12/15 (0.800)` | `21.212` | `75.516` | `30.801` | `8503.1` |
| Qwen3.5-9B Q8_0 | `12/15 (0.800)` | `23.585` | `99.580` | `8.597` | `8785.1` |
| Qwen3.5-27B UD-Q4_K_XL | `9/15 (0.600)` | `24.823` | `129.636` | `48.043` | `13502.1` |
| Qwen3.5-35B-A3B Q4_K_M | `12/15 (0.800)` | `17.023` | `54.279` | `13.234` | `5637.1` |
| Qwen3.5-35B-A3B Q5_K_M | `12/15 (0.800)` | `17.124` | `55.569` | `13.250` | `5730.9` |
| Qwen3.5-35B-A3B UD-Q4_K_XL | `9/15 (0.600)` | `18.619` | `58.111` | `23.066` | `6654.7` |

Primary `np=1` interpretation:

- 9B Q8 still has the smallest replay bucket on the dense path, but even there the `np=1` median is now effectively break-even rather than a clean win.
- 9B UD-Q4 has the same short-case acceptance ratio as Q8, but replay is much larger and the end-to-end result stays negative.
- 27B still shows the same dense scaling problem: lower acceptance and a much more expensive accept+replay path.
- The A3B quants are exact again on `np=1`, and their replay bucket on this short case is modest, but baseline decode is so fast that even modest speculative overhead stays net negative.

### Bad `np=2`

| Model | Acceptance | Draft ms | Accept ms | Replay ms | Mean step total us |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-9B UD-Q4_K_XL | `126/182 (0.692)` | `91.303` | `581.841` | `143.500` | `4463.2` |
| Qwen3.5-9B Q8_0 | `126/184 (0.685)` | `117.090` | `663.657` | `82.937` | `4644.1` |
| Qwen3.5-27B UD-Q4_K_XL | `140/184 (0.761)` | `127.729` | `958.105` | `395.062` | `7962.4` |
| Qwen3.5-35B-A3B Q4_K_M | `141/183 (0.770)` | `316.863` | `394.586` | `185.392` | `4822.4` |
| Qwen3.5-35B-A3B Q5_K_M | `147/183 (0.803)` | `314.705` | `389.498` | `174.579` | `4725.2` |
| Qwen3.5-35B-A3B UD-Q4_K_XL | `158/183 (0.863)` | `316.775` | `387.955` | `122.578` | `4496.7` |

Bad `np=2` interpretation:

- 27B remains the clearest dense regression: even with better acceptance than the 9B bad cases, accept and replay are still too expensive.
- The A3B quants are not primarily limited by replay alone; they are already so fast in baseline mode that even moderate draft + accept overhead is too expensive.
- The balanced `Q4_K_M` GGUF is no longer a correctness outlier after the post-replay guard, but its throughput is still materially below baseline.

## Current Conclusions

- The current native-MTP implementation is now benchmark-clean for:
  - `Qwen3.5-9B`, `Qwen3.5-27B`, and the checked `Qwen3.5-35B-A3B` quants on `np=1` correctness
  - `Qwen3.5-35B-A3B Q4_K_M` after the post-replay guard fix
  - `Qwen3.5-9B q8_0` as the only current net-positive path, but only on the easier `np=2` cases
- The current implementation is not yet benchmark-good for:
  - any checked `np=1` end-to-end win
  - broad dense-model speedups across quants
  - 27B throughput
  - MoE throughput
  - `Qwen3.5-35B-A3B Q4_K_M` throughput even after the exactness fix

Practical reading:

- if the target is “ship a native-MTP case that is actually faster today”, the only clean answer from this matrix is still `Qwen3.5-9B q8_0`, and only on the easier `np=2` cases
- if the target is “make native MTP broadly speed-positive”, the next optimization work has to reduce real control-path cost, especially accept and replay, while preserving the `np=1` exactness contract
- for the first upstream-oriented series, the pragmatic read is narrower:
  - keep `9B q8_0` as the only active speed target
  - keep `27B` as supporting dense regression coverage
  - keep A3B only as correctness / stability regression coverage
  - do not treat `qwen35moe` as a speed target for v1

## Step-01 Visibility Gate

After the main 2026-04-10 sweep, the branch was rerun with expanded per-step visibility in `native MTP step:` so the validator could count:

- pure fast-path verifier steps
- logits-suppressed accept steps
- forced plain post-replay steps
- guard hits

Dense-gate result:

- speculative accept rows are already almost fully on the intended greedy verifier fast path
- representative coverage from `/tmp/native-mtp-step-01`:
  - `9B q8_0 primary np=1`: `15/15` pure-fast-path, `15/15` logits-suppressed
  - `9B q8_0 bad np=2`: `182/186` pure-fast-path, `182/186` logits-suppressed
  - `9B UD-Q4 good np=2`: `180/186` pure-fast-path, `180/186` logits-suppressed
  - `27B UD-Q4 bad np=2`: `180/186` pure-fast-path, `180/186` logits-suppressed
- A3B `Q4_K_M` smoke remained exact on `primary`, `good`, and `bad` `np=1`

Implication:

- there is little evidence left for a profitable “split pure verifier rows out of mixed decode chunks” optimization on the current Qwen 3.5 path
- the remaining drag is mostly verifier/replay economics, not a hidden fast-path coverage leak

Follow-up that was tried and discarded:

- a separate dense one-step post-replay cooldown was tested and then dropped
- reason:
  - current libllama marks `Qwen3.5-9B` and `Qwen3.5-27B` as `hybrid`, so the existing post-replay guard is already the live policy there
  - the cooldown trial produced `0` distinct `cooldown_hits` on the q8 gate and no new behavioral separation from the existing guard

Next local branch implied by this result:

- the only remaining narrow runtime branch worth trying is to see whether the replay guard can be narrowed safely for dense `Qwen3.5`
- if that does not recover a clear repeated `9B q8_0 np=1` win, the current one-token native-MTP design should be treated as having reached its structural ceiling on this hybrid path

## Raw Artifacts

Per-model JSON summaries:

- `/tmp/native-mtp-bench-20260410-post-replay-guard/qwen35-9b-ud-q4.json`
- `/tmp/native-mtp-bench-20260410-post-replay-guard/qwen35-9b-q8_0.json`
- `/tmp/native-mtp-bench-20260410-post-replay-guard/qwen35-27b-ud-q4.json`
- `/tmp/native-mtp-bench-20260410/qwen35-9b-q8_0.json`
- `/tmp/native-mtp-bench-20260410-post-replay-guard/qwen35-35b-a3b-q4_k_m.json`
- `/tmp/native-mtp-bench-20260410-post-replay-guard/qwen35-35b-a3b-q5_k_m.json`
- `/tmp/native-mtp-bench-20260410-post-replay-guard/qwen35-35b-a3b-ud-q4.json`
- `/tmp/native-mtp-step-01/qwen35-9b-q8_0.json`
- `/tmp/native-mtp-step-01/qwen35-9b-ud-q4.json`
- `/tmp/native-mtp-step-01/qwen35-27b-ud-q4.json`
- `/tmp/native-mtp-step-01/qwen35-35b-a3b-q4_k_m-smoke.json`

Each JSON also points at its per-scenario log directory under `/tmp`.

## Related Notes

- [native-mtp-optimization-plan.md](native-mtp-optimization-plan.md)
- [native-mtp-model-prep.md](native-mtp-model-prep.md)
