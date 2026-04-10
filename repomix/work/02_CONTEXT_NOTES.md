CONTEXT_NOTES (optional):

- Repo / branch state:
  - repo: `/home/florian/llama.cpp-upstream-mtp-plan`
  - branch: `feat/native-mtp-backend-seed`
  - current commit in this private mirror: `0c6c01d1f` (`native MTP: keep seed transport on backend`)

- High-level native-MTP status already implemented in this private mirror:
  - HF -> GGUF + loader support for Qwen 3.5 native MTP / NextN
  - runtime native-MTP path for Qwen 3.5 and Qwen 3.5 MoE
  - server integration for `--spec-type mtp`
  - rollback / replay hardening for recurrent / hybrid models
  - backend-resident seed transport for CUDA / non-host verifier paths

- Current correctness contract:
  - validated exact:
    - CUDA Berlin `np=1/2`
    - CUDA Moon `np=1/2`
  - known limitation:
    - CUDA Rust `np=2` still diverges because of the documented hybrid / recurrent `np>1` exactness limitation
    - do not spend this plan trying to “solve” that batch-invariance problem unless there is a very small upstream-friendly mitigation

- Important recent finding:
  - the backend-resident seed transport was worth landing structurally, but it did not materially improve end-to-end CUDA throughput on the validated exact cases
  - actual before/after MTP tok/s on this host were roughly flat:
    - Berlin `np=1`: `173.37 -> 172.71`
    - Berlin `np=2`: `160.44 / 160.88 -> 160.46 / 160.04`
    - Moon `np=1`: `151.79 -> 150.84`
    - Moon `np=2`: `141.07 / 140.74 -> 139.45 / 139.75`
  - implication:
    - the host round trip is no longer the highest-value remaining target
    - the next real wins have to come from draft policy / replay cost / server runtime overhead

- Remaining optimization backlog to plan:
  1. adaptive native-MTP backoff on replay-heavy prompts
  2. replay-path reduction
  3. small server hot-path cleanup

- What is already known about those priorities:
  - replay-heavy workloads are the main bad case now
  - replay dominates the remaining overhead on prompts like the Rust stress case
  - smaller server-loop/container setup costs still exist, but they are clearly secondary
  - future steps must always prove real end-to-end tok/s improvement, not just lower internal timings

- Current validation / benchmarking method already present in this mirror:
  - unit tests:
    - `LLAMA_SERVER_BIN_PATH=/home/florian/llama.cpp-upstream-mtp-plan/build-cuda-server/bin/llama-server .venv-tests/bin/python -m pytest tools/server/tests/unit/test_speculative.py -q`
  - exact CUDA validation:
    - `python3 scripts/validate_mtp_cuda.py --binary build-cuda-server/bin/llama-server --model /mnt/models/GGUF/Qwen3.5-9B-MTP-q8_0.gguf ...`
  - benchmark note:
    - `docs/development/native-mtp-benchmarks.md`

- Key benchmark inputs currently considered representative:
  - Berlin exact case:
    - prompt: `Write one short sentence about Berlin.`
    - seed: `42`
    - `n_predict=48`
    - exact for `np=1/2`
  - Moon exact case:
    - prompt: `Write two short sentences about the Moon.`
    - seed: `31415`
    - `n_predict=64`
    - exact for `np=1/2`
  - Rust stress case:
    - prompt: `List three reasons Rust is used for systems programming.`
    - seed: `777`
    - `n_predict=64`
    - still non-exact for `np=2`, but should not regress into corruption / crashes / invalid token streams

- What the external planner should assume about implementation style:
  - keep it upstream-friendly and simple
  - prefer generic native-MTP runtime improvements over Qwen-only hacks
  - keep changes understandable for future native-MTP model support
  - avoid large scheduler/backend changes unless the payoff is clearly better than server/runtime policy improvements

- Recent private-mirror changes not necessarily obvious from upstream public files:
  - backend-resident native-MTP seed transport is already implemented here
  - it uses persistent `seed_cache_dev` / `seed_batch_dev`, explicit seed mode, and generation-based graph reuse invalidation
  - debug envs currently exist for rollout:
    - `LLAMA_MTP_BACKEND_SEED_DEBUG=1`
    - `LLAMA_MTP_BACKEND_SEED_FORCE_HOST=1`
  - the forced host fallback is only for comparison/debugging and is not the validated fast path on non-host multi-sequence CUDA

- Surgical public llama.cpp context is allowed if needed.
  Suggested upstream URLs to inspect only if necessary:
  - `https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-context.cpp`
    - current server batching / speculative structure
  - `https://github.com/ggml-org/llama.cpp/blob/master/src/llama-context.cpp`
    - current context/runtime patterns
  - `https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml-backend.h`
    - backend copy / sync / event APIs
  - `https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend.cpp`
    - scheduler input-copy behavior / split logic
  Use these surgically. Do not expand into broad repo exploration unless a concrete step truly requires it.
