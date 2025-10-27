# Repository Guidelines

## Project Structure & Module Organization
Core inference code lives in `src/`, tensor kernels in `ggml/`, and public headers in `include/`. CLI tools (`llama-cli`, `llama-server`, benches) land in `build/bin/` after configuration. Utilities stay in `tools/` or `scripts/`, model converters at the root (`convert_*`), reusable weights in `models/`, documentation under `docs/`, tests under `tests/`, and CI automation in `ci/`.

## Build, Test, and Development Commands
Use CMake end-to-end: `cmake -B build` configures a CPU build and `cmake --build build --config Release -j 8` emits optimized binaries. Enable accelerators with flags like `-DGGML_CUDA=ON`, `-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS`, or `-DGGML_SYCL=ON`. Switch to debug with `-DCMAKE_BUILD_TYPE=Debug`. For CI parity, run `bash ci/run.sh ./tmp/results ./tmp/mnt` after exporting the matching `GG_BUILD_*` variables.

## Coding Style & Naming Conventions
Follow CONTRIBUTING.md: four spaces, braces on the same line, minimal STL, and no new third-party dependencies. Keep public APIs on sized integers and prefer plain C where possible. Use `snake_case` for symbols, prefix enum members with their enum name (`LLAMA_VOCAB_TYPE_*`), and follow the `<class>_<action>` template (`llama_sampler_chain_remove`). C/C++ filenames stay lowercase with dashes, Python helpers use lowercase underscores, and `clang-format` (clang-tools v15+) is the fallback formatter.

## Testing Guidelines
Run `ctest --output-on-failure -L main` for CPU coverage; set `LLAMACPP_TEST_MODELFILE=/path/to/model.gguf` and target the `model` label when weights are required. `scripts/debug-test.sh test-tokenizer` helps isolate a failing case under gdb or lldb. When touching inference-critical code, capture `llama-perplexity` or `llama-bench` numbers in the PR to prove accuracy and throughput hold.

## Commit & Pull Request Guidelines
Squash commits follow `<module> : summary (#issue)` (e.g., `samplers : fix logits clamp (#15321)`). Keep each PR focused, document affected hardware, and include repro steps, metrics, or screenshots. Run the relevant `ctest` labels plus any `ci/run.sh` scenarios you touched before requesting review. Tag CODEOWNERS, allow maintainer write access, and link issues or discussions for traceability.

## XQuant & Model Resources
Before iterating on xquant, note that shared models live under `/Users/florian/Local/models/`. Exercise every attention variant with the supplied GGUFs: MHA `phi-2.Q4_K_M.gguf`, MQA `gemma-2-2b-it-Q4_K_M.gguf`, GQA `qwen2.5-0.5b-instruct-q4_k_m.gguf`, and MLA `deepseek-v2-lite-chat.Q2_K.gguf`. Final memory and throughput checks must use the larger `Qwen3-32B-Q4_K_M.gguf`, `gpt-oss-20b-UD-Q4_K_XL.gguf`, and `Mistral-Small-3.2-24B-Instruct-2506-UD-Q5_K_XL.gguf`. Automate repeat runs with `/Users/florian/Local/test-xquant/test.sh`; update the script if new coverage is required.

## Security & Configuration Tips
Report vulnerabilities through `SECURITY.md`; never include secrets or proprietary weights in issues. GPU backends depend on vendor SDKs (CUDA, SYCL, MUSA), so document the driver/runtime versions you validated. Keep shared commands relative to the repo root so instructions remain shell-agnostic.
