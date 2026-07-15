# llama.cpp fork contributor guidance

This is a private llama.cpp fork for XQuant development. Keep changes easy to
review and rebase onto upstream while protecting inference correctness,
performance, and the feature-off baseline.

## Project Structure & Module Organization
Core inference code lives in `src/`, tensor kernels in `ggml/`, and public headers in `include/`. CLI tools (`llama-cli`, `llama-server`, benches) land in `build/bin/` after configuration. Utilities stay in `tools/` or `scripts/`, model converters at the root (`convert_*`), reusable weights in `models/`, documentation under `docs/`, tests under `tests/`, and CI automation in `ci/`.

## Build, Test, and Development Commands
Use CMake end-to-end: `cmake -B build` configures a CPU build and `cmake --build build --config Release -j 8` emits optimized binaries. Enable accelerators with flags like `-DGGML_CUDA=ON`, `-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS`, or `-DGGML_SYCL=ON`. Switch to debug with `-DCMAKE_BUILD_TYPE=Debug`. For CI parity, run `bash ci/run.sh ./tmp/results ./tmp/mnt` after exporting the matching `GG_BUILD_*` variables.

## Coding Style & Naming Conventions
Follow CONTRIBUTING.md: four spaces, braces on the same line, minimal STL, and no new third-party dependencies. Keep public APIs on sized integers and prefer plain C where possible. Use `snake_case` for symbols, prefix enum members with their enum name (`LLAMA_VOCAB_TYPE_*`), and follow the `<class>_<action>` template (`llama_sampler_chain_remove`). C/C++ filenames stay lowercase with dashes, Python helpers use lowercase underscores, and `clang-format` (clang-tools v15+) is the fallback formatter.

## Testing Guidelines
Run `ctest --output-on-failure -L main` for CPU coverage when broad CPU behavior
is affected; set `LLAMACPP_TEST_MODELFILE=/path/to/model.gguf` and target the
`model` label when weights are required. `scripts/debug-test.sh test-tokenizer`
helps isolate a failing case under gdb or lldb. Capture `llama-perplexity` or
`llama-bench` evidence when the change can affect, or the claim concerns,
accuracy or throughput; match the benchmark scope to the affected path.

## Commit & Pull Request Guidelines
Squash commits follow `<module> : summary (#issue)` (e.g., `samplers : fix logits clamp (#15321)`). Keep each PR focused, document affected hardware, and include repro steps, metrics, or screenshots. Run the relevant `ctest` labels plus any `ci/run.sh` scenarios you touched before requesting review. Tag CODEOWNERS, allow maintainer write access, and link issues or discussions for traceability.

## XQuant & Model Resources
Configure model and harness locations through environment variables or explicit
arguments; do not add personal absolute paths to shared instructions. The local
matrix currently uses small MHA, MQA, GQA, and MLA GGUFs for semantic coverage
and larger Qwen, GPT-OSS, and Mistral models for final memory/throughput claims.
See `docs/XQUANT_INSTRUCTIONS.md` and `docs/XQUANT_TECH_SPEC.md` for the feature
contract, while verifying their assumptions against the current rebased code.

## Engineering judgment and proportionate validation

- Make the smallest upstream-compatible change that completes the requested
  XQuant or llama.cpp outcome. Preserve the normal KV path when XQuant is off
  and the declared no-KV/fail-fast boundary when it is on.
- Add abstractions only for demonstrated memory-module or sequence semantics;
  do not build compatibility wrappers, CI machinery, or speculative backends
  merely for symmetry with upstream.
- Compile the touched target and run the nearest relevant tests first. Widen to
  CPU `main`, CI scenarios, or accelerator builds only when shared code,
  portability, upstream submission, or unresolved uncertainty warrants it.
- For XQuant semantic changes, begin with the smallest representative model for
  the affected attention path. Exercise all affected MHA/MQA/GQA/MLA paths when
  shared attention, memory, sequence, eviction, or serialization logic changes.
- Run large-model perplexity, memory, and throughput campaigns only when making
  accuracy/performance claims, closing a milestone, or preparing release or
  upstream review. Record hardware, build flags, model identity, and baseline.
- Documentation and other non-executable metadata need focused diff/link checks
  only. Do not run models or full CI when they cannot validate the change.
- Ask when a material design choice or expensive external validation is needed;
  otherwise implement, run focused checks, and stop when the outcome is proven.

## Security & Configuration Tips
Report vulnerabilities through `SECURITY.md`; never include secrets or proprietary weights in issues. GPU backends depend on vendor SDKs (CUDA, SYCL, MUSA), so document the driver/runtime versions you validated. Keep shared commands relative to the repo root so instructions remain shell-agnostic.

If work is intended for upstream, follow the current upstream `AGENTS.md` and
`CONTRIBUTING.md`; private-fork permission does not waive upstream contribution
policy or the contributor's responsibility to understand and maintain the code.
