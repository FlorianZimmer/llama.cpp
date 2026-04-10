# Upstream Diff Summary

This private branch should be reviewed relative to public `upstream/master`.

If you need omitted surrounding context, you may consult the public upstream repository surgically, but do not assume this private branch matches upstream in the files listed below.

## Current review target

- branch: `feat/native-mtp-qwen35-dense-speedup`
- local upstream base used here: `upstream/master` at `d6f3030047f85a98b009189e76f441fe818ea44d`

## Important branch fact

The live branch was just narrowed to dense Qwen 3.5 native MTP only.

That means:

- `qwen35` native-MTP remains in scope
- `qwen35moe` native-MTP support was removed again from the live branch
- `Qwen3.5-35B-A3B` / MoE is no longer part of the live V1 code or benchmark scope

Some historical notes in the included docs still describe the removed MoE path because that history matters for planning, but the live code path under review is dense-only.

## High-level change groups vs upstream master

### Native-MTP runtime and API additions

- `include/llama.h`
- `src/llama-mtp.h`
- `src/llama-mtp.cpp`
- `src/llama-context.cpp`
- `src/llama-context.h`
- `src/llama-graph.cpp`
- `src/llama-graph.h`
- `common/speculative.cpp`
- `common/sampling.cpp`
- `common/sampling.h`
- `common/common.h`
- `common/arg.cpp`

What changed:

- native-MTP capability and metadata helpers
- native draft entrypoints
- backend-resident seed transport
- graph input support for native-MTP seed rows
- direct output-token / logits controls used by the greedy verifier fast path

### Qwen 3.5 model / conversion support

- `convert_hf_to_gguf.py`
- `gguf-py/gguf/constants.py`
- `src/llama-model.cpp`
- `src/llama-model.h`
- `src/models/qwen35.cpp`
- `src/models/models.h`
- `src/CMakeLists.txt`
- `src/llama-arch.cpp`

What changed:

- Qwen 3.5 MTP metadata / tensor conversion support
- dense Qwen 3.5 native-MTP graph-builder support
- native-MTP runtime registration and wiring

### Server integration and runtime profiling

- `tools/server/server-context.cpp`
- `tools/server/server-task.cpp`
- `tools/server/README.md`
- `tools/server/tests/unit/test_speculative.py`
- `tools/server/tests/utils.py`
- `tools/cli/README.md`

What changed:

- `--spec-type mtp` server-side integration
- greedy verifier accept fast path
- optional raw-logit suppression for token-only verifier batches
- rollback / replay plumbing for native MTP
- per-step timing / visibility logging

### Benchmarking and quant-audit tooling

- `scripts/validate_mtp_cuda.py`
- `scripts/audit_mtp_quantization.py`

What changed:

- strict CUDA baseline-vs-MTP validation
- per-step acceptance / timing parsing
- quant audit support for native-MTP tensors

## Diff stat against upstream master

Current working-branch diff stat, excluding docs and repomix files:

```text
 common/arg.cpp                                     |   4 +-
 common/common.h                                    |   1 +
 common/sampling.cpp                                |  37 +
 common/sampling.h                                  |   4 +
 common/speculative.cpp                             |   7 +
 convert_hf_to_gguf.py                              |  82 +-
 gguf-py/gguf/constants.py                          |  16 +-
 include/llama.h                                    |  65 ++
 scripts/audit_mtp_quantization.py                  | 321 +++++++
 scripts/validate_mtp_cuda.py                       | 767 +++++++++++++++++
 src/CMakeLists.txt                                 |   1 +
 src/llama-arch.cpp                                 |  12 +-
 src/llama-context.cpp                              | 533 +++++++++++-
 src/llama-context.h                                |  21 +-
 src/llama-graph.cpp                                | 114 ++-
 src/llama-graph.h                                  | 115 +++
 src/llama-model.cpp                                |  65 +-
 src/llama-model.h                                  |   5 +
 src/llama-mtp.cpp                                  | 276 ++++++
 src/llama-mtp.h                                    |  96 +++
 src/models/models.h                                |  14 +
 src/models/qwen35.cpp                              | 105 ++-
 tools/cli/README.md                                |   1 +
 tools/server/README.md                             |   2 +-
 tools/server/server-context.cpp                    | 932 ++++++++++++++++++++-
 tools/server/server-task.cpp                       |  13 +
 tools/server/tests/unit/test_speculative.py        |  48 ++
 tools/server/tests/utils.py                        |   3 +
```

## What not to assume from public upstream

Do not assume public upstream already contains:

- native-MTP API surface
- native Qwen 3.5 dense draft path
- output-token / logits suppression controls used here
- current server-side native-MTP replay / profiling logic
- CUDA validator / quant-audit scripts used here

If you need omitted context from public upstream, fetch only the smallest surrounding files or symbols needed to understand these local changes.
