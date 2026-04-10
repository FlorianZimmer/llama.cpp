CONTEXT_NOTES:

Branch / diff context:

- Current branch: `feat/native-mtp-upstream-prep`
- Public-upstream diff base previously used for local comparison: `d6f3030047f85a98b009189e76f441fe818ea44d`
- This is a private mirror. The local docs and included files are the authoritative source for this branch state.

What this branch already adds relative to public upstream:

- Qwen 3.5 native-MTP GGUF conversion / metadata / loader plumbing
- native-MTP runtime state and draft path in `src/llama-mtp.*` and `src/llama-context.cpp`
- Qwen 3.5 dense and MoE graph builders
- server integration for `--spec-type mtp`
- direct greedy verifier-token access plus raw-logit suppression controls
- replay / rollback support for hybrid-recurrent paths
- CUDA benchmark harness and quant-audit tooling

Files that matter most for the current question:

- `tools/server/server-context.cpp`
- `src/llama-mtp.h`
- `src/llama-mtp.cpp`
- `src/llama-context.cpp`
- `src/llama-graph.h`
- `src/llama-graph.cpp`
- `src/models/qwen35.cpp`
- `src/models/qwen35moe.cpp`
- `src/llama-arch.cpp`
- `scripts/validate_mtp_cuda.py`
- `scripts/audit_mtp_quantization.py`
- `docs/development/native-mtp-benchmarks.md`
- `docs/development/native-mtp-optimization-plan.md`
- `docs/development/native-mtp-model-prep.md`

Current runtime facts:

- Native runtime still drafts only `1` continuation token per step even if metadata reports more predictor layers.
- `np=1` is the correctness-clean contract.
- `np>1` on current hybrid/recurrent native-MTP is stability-only.
- The branch already has:
  - greedy verifier accept fast path
  - optional raw-logit suppression on token-only accept batches
  - backend-resident seed transport
  - per-step runtime/acceptance profiling

Important historical benchmark fact:

- Under the same newer harness, `Qwen3.5-9B q8_0` was clearly faster before the permanent replay guard:
  - file: `/tmp/native-mtp-bench-20260410/qwen35-9b-q8_0.json`
  - `primary np=1`: `150.83 -> 163.23 tok/s` (`1.082x`)
  - `good np=1`: `148.88 -> 153.38 tok/s` (`1.030x`)
  - `primary np=2`: `128.08 -> 138.55 tok/s` (`1.082x`)
  - `good np=2`: `131.40 -> 145.13 tok/s` (`1.104x`)

Current benchmark state after the permanent replay guard:

- file: `/tmp/native-mtp-bench-20260410-post-replay-guard/qwen35-9b-q8_0.json`
- `primary np=1`: `150.53 -> 150.31 tok/s` (`0.999x`)
- `good np=1`: `148.62 -> 139.40 tok/s` (`0.938x`)
- `primary np=2`: `127.85 -> 133.26 tok/s` (`1.042x`)
- `good np=2`: `132.66 -> 137.12 tok/s` (`1.034x`)
- `bad np=1`: `148.82 -> 125.10 tok/s` (`0.841x`)
- `bad np=2`: `132.99 -> 115.83 tok/s` (`0.871x`)

Current full-matrix reading:

- no checked model or quant is net-positive on `np=1`
- only `Qwen3.5-9B q8_0` is still speed-positive at all, and only on the easier `np=2` cases
- `Qwen3.5-9B UD-Q4_K_XL` is slower everywhere
- `Qwen3.5-27B UD-Q4_K_XL` is slower everywhere
- `Qwen3.5-35B-A3B` is slower everywhere

Representative current results:

- 9B `UD-Q4_K_XL`, `primary np=1`: `175.91 -> 159.01 tok/s` (`0.904x`)
- 9B `q8_0`, `primary np=1`: `150.53 -> 150.31 tok/s` (`0.999x`)
- 9B `q8_0`, `primary np=2`: `127.85 -> 133.26 tok/s` (`1.042x`)
- 27B `UD-Q4_K_XL`, `primary np=1`: `72.35 -> 62.27 tok/s` (`0.861x`)
- 35B-A3B `Q4_K_M`, `primary np=1`: `228.26 -> 170.72 tok/s` (`0.748x`)

Representative current profile readings:

- 9B `UD-Q4_K_XL`, `primary np=1`:
  - acceptance `12/15 (0.800)`
  - `draft 21.212 ms`
  - `accept 75.516 ms`
  - `replay 30.801 ms`
- 9B `q8_0`, `primary np=1`:
  - acceptance `12/15 (0.800)`
  - `draft 23.585 ms`
  - `accept 99.580 ms`
  - `replay 8.597 ms`
- 27B `UD-Q4_K_XL`, `bad np=2`:
  - acceptance `140/184 (0.761)`
  - `draft 127.729 ms`
  - `accept 958.105 ms`
  - `replay 395.062 ms`

Visibility pass result from `/tmp/native-mtp-step-01`:

- the branch now logs step-level `fast`, `logits_suppressed`, `forced_plain`, `cooldown`, and `guard` flags
- speculative accept rows are already almost fully on the intended greedy fast path with logits suppressed
- representative coverage:
  - `9B q8_0 primary np=1`: `15/15` pure-fast-path, `15/15` logits-suppressed
  - `9B q8_0 bad np=2`: `182/186`
  - `9B UD-Q4 good np=2`: `180/186`
  - `27B UD-Q4 bad np=2`: `180/186`
- local conclusion from that pass:
  - the earlier “split pure verifier rows from mixed decode chunks” idea is probably not the main remaining win
  - remaining losses are mostly verifier/replay economics, not missed fast-path coverage

Important architectural fact uncovered after that pass:

- `Qwen3.5` dense is also marked `hybrid` in libllama, not just `Qwen3.5-MoE`
- that means the current one-step post-replay guard is already the live replay policy on 9B and 27B too
- a trial “dense cooldown” branch produced no distinct `cooldown_hits` and was discarded

Interpretation of the regression:

- the older `9B q8_0` wins likely regressed because the permanent replay guard applies on dense Qwen 3.5 as well
- so the current question is not “is there still a missed greedy fast path?”
- it is closer to:
  - does dense Qwen 3.5 actually need the same replay guard as A3B?
  - if yes, is the remaining ceiling structural for one-token native MTP on hybrid Qwen 3.5?
  - if no, can the guard be narrowed safely without breaking `np=1` exactness?

MoE / A3B context:

- `Qwen3.5-35B-A3B Q4_K_M` originally had a real quantization-side weakness in `blk.40.nextn.eh_proj.weight`
- balanced quant fix promoted that tensor from `Q4_K` to `Q5_K`
- that was necessary but not sufficient
- the remaining exactness issue was isolated to the first speculative step after replay
- current branch fixed that conservatively by forcing one plain verifier step immediately after replay on hybrid/recurrent native-MTP slots
- A3B is now correctness-clean on the checked `np=1` cases, but still materially slower than baseline on every checked quant

Current local belief on scope:

- dense Qwen 3.5 still looks like the only plausible speed target
- MoE Qwen 3.5 looks more like a functionality/correctness burden than a likely speed-positive target for the current one-token design
- the next meaningful runtime question may be too large for this first upstream-oriented branch because it likely involves deeper hybrid replay/guard behavior rather than another small server-local cleanup
