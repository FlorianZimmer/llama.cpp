# Native MTP Runtime Plan

This document tracks the long-term plan for native MTP support, starting with Qwen 3.5 and targeting an implementation that is suitable for upstreaming, backend performance, and future model support.

## Problem Statement

The current native MTP runtime on this branch is based on a second `llama_context` used as an MTP-only sidecar. That design works on CPU, but it is not a clean long-term fit for CUDA because it introduces:

- a second scheduler and graph build path
- a second KV/memory state to keep in sync
- verifier-to-sidecar hidden-state handoff
- shape churn in the sidecar decode path
- special behavior differences between `-np 1` and `-np 2`
- backend-specific performance work leaking into `common/speculative.cpp`

The sidecar design is acceptable as a short-term prototype, but it is not the right architecture for a long-lived native MTP runtime.

## Long-Term Direction

Native MTP should become a first-class decode capability inside a single `llama_context`.

That means:

- one verifier context
- one scheduler
- one memory/KV state
- one graph build pipeline
- no cross-context hidden-state replay protocol
- no special shared-sidecar code for `-np 2`

The speculative orchestration layer should continue to decide when to draft and when to accept, but native MTP proposal generation should move out of the current second-context replay path and into the normal model/runtime path.

## Goals

- Preserve exact greedy output equality between baseline decoding and native MTP.
- Improve CUDA performance without regressing CPU correctness.
- Keep the implementation backend-agnostic in structure.
- Make Qwen 3.5 the first native-MTP model on the new runtime, but keep the runtime reusable for other native-MTP checkpoints.
- Minimize special casing in `common/speculative.cpp`.
- Make `-np 1` and `-np 2` use the same runtime model, not separate speculative implementations.

## Non-Goals

- Adding CUDA-only kernels before the runtime architecture is fixed.
- Growing the current second-context sidecar into a permanent design.
- Accepting performance wins that change greedy output behavior.
- Accepting a change that only helps `-np 1` but breaks or complicates `-np 2`.

## Current Status

Stable baseline on this branch:

- CPU native MTP works and is validated.
- CUDA native MTP is functionally correct, but slower than baseline.
- Exact greedy output equality is already testable through the server path.
- Validation harness exists in [`scripts/validate_mtp_cuda.py`](../../scripts/validate_mtp_cuda.py).

Current key technical finding:

- The current two-context sidecar design causes repeated graph rebuild pressure on CUDA because the MTP replay shape is not naturally stable.
- Experiments showed that shaping the sidecar replay path more aggressively can improve single-slot CUDA, but every attempt that touched the shared `-np 2` path increased complexity and caused correctness or latency regressions.
- The current server branch therefore keeps native-MTP multi-slot validation on the per-slot runtime and intentionally leaves the older shared native-MTP sidecar path disabled until a single-context runtime replaces it.

This confirms that the architecture, not just a missing CUDA micro-optimization, is the main issue.

## Target Architecture

### 1. Native MTP becomes a `llama_context` capability

The verifier context should be able to:

- expose whether the loaded model supports native MTP
- generate native MTP proposals directly from verifier-owned tensors/state
- return proposal tokens and related accounting data without a second context

Possible internal shape:

- `llama_context` owns a native-MTP runtime object or mode flag
- the graph builder produces proposal tensors when native MTP is enabled
- model-specific code provides the MTP head construction

### 2. `common/speculative` stays orchestration-only

`common/speculative.cpp` should continue to decide:

- whether speculative decoding is enabled
- which implementation is active
- how acceptance/rejection accounting works

It should stop owning the low-level runtime mechanics for native MTP hidden replay.

### 3. Model code owns model-specific MTP graph pieces

Model-specific code should be responsible for:

- declaring native MTP support
- exposing max MTP depth
- building the native MTP proposal head/tensors

The runtime should not hardcode Qwen 3.5 details in `common/`.

## Expected File Ownership

### Runtime and API

- [`include/llama.h`](../../include/llama.h)
  - only if a public API surface is needed
  - prefer private/internal APIs first
- [`src/llama-context.h`](../../src/llama-context.h)
  - native MTP runtime state
  - internal proposal methods
- [`src/llama-context.cpp`](../../src/llama-context.cpp)
  - proposal execution path
  - accounting and output plumbing
  - integration with decode flow

### Graph and scheduler

- [`src/llama-graph.h`](../../src/llama-graph.h)
  - graph params/result extensions if native MTP outputs are part of the main graph
- [`src/llama-graph.cpp`](../../src/llama-graph.cpp)
  - proposal tensor wiring
  - optional reuse instrumentation during development

### Model-specific support

- [`src/models/qwen35.cpp`](../../src/models/qwen35.cpp)
  - Qwen 3.5 native MTP graph construction
  - max depth / proposal head handling

### Speculative orchestration

- [`common/speculative.cpp`](../../common/speculative.cpp)
  - reduce native MTP handling to orchestration
  - eventually remove second-context native MTP replay path

### Server and examples

- [`tools/server/server-context.cpp`](../../tools/server/server-context.cpp)
  - integration checks only
  - avoid putting runtime logic here
- [`examples/speculative-simple/speculative-simple.cpp`](../../examples/speculative-simple/speculative-simple.cpp)
  - keep as a simple validation path

### Tests and validation

- [`scripts/validate_mtp_cuda.py`](../../scripts/validate_mtp_cuda.py)
  - regression gate for exact greedy outputs and `tok/s`
- [`tests`](../../tests)
  - add focused runtime tests where feasible

## Validation Rules

These are hard gates for every phase:

- Greedy decoding must produce exactly the same output as baseline.
- `llama-server` must pass both `-np 1` and `-np 2`.
- CPU correctness must not regress while working on CUDA.
- If a change improves `-np 1` but breaks or hangs `-np 2`, it does not land.
- If a change improves CUDA but changes exact greedy output, it does not land.

## Standard Validation Commands

Set a model path first:

```bash
export MODEL=/path/to/Qwen3.5-9B-MTP-q8_0.gguf
```

Build:

```bash
cmake --build build-cpu --target llama-server llama-speculative-simple test-arg-parser -j 8
cmake --build build-cuda --target llama-server llama-speculative-simple test-arg-parser -j 8
```

Argument parser:

```bash
build-cpu/bin/test-arg-parser
```

CPU validation:

```bash
./scripts/validate_mtp_cuda.py \
  --model "$MODEL" \
  --binary build-cpu/bin/llama-server \
  --ngl 0 \
  --port-base 18280
```

CUDA validation:

```bash
./scripts/validate_mtp_cuda.py \
  --model "$MODEL" \
  --binary build-cuda/bin/llama-server \
  --port-base 18300
```

What to check:

- exact greedy output equality
- `np=1` baseline vs native-MTP `tok/s`
- `np=2` per-slot baseline vs native-MTP `tok/s`
- no hangs or timeouts
- no unexpected graph rebuild explosions in logs while iterating

## Phased Implementation Plan

### Phase 0: Freeze the stable baseline

Purpose:

- Keep one known-good reference before further runtime work.

Edit:

- no behavior changes
- keep the current CPU implementation and validation harness intact

Test:

- run the standard CPU and CUDA validation commands

Validate:

- capture reference output equality and reference `tok/s`

### Phase 1: Add safe runtime instrumentation

Purpose:

- Understand where native MTP time is spent without changing behavior.

Edit:

- [`src/llama-context.cpp`](../../src/llama-context.cpp)
  - extend or refine timing around native MTP prepare/build/output/compute
- [`src/llama-graph.h`](../../src/llama-graph.h)
  - optional debug-only graph reuse reason reporting
- [`src/llama-graph.cpp`](../../src/llama-graph.cpp)
  - optional debug-only input reuse logging

Test:

- rebuild CPU and CUDA
- rerun validation
- run targeted debug sessions with log env vars enabled

Validate:

- no behavior change
- no performance regression from debug code when disabled

### Phase 2: Define a private single-context native MTP runtime API

Purpose:

- Create the internal seam that removes the need for a second context.

Edit:

- [`src/llama-context.h`](../../src/llama-context.h)
  - add private runtime state and helper methods
- [`src/llama-context.cpp`](../../src/llama-context.cpp)
  - add internal proposal entry points
- optionally avoid any public API change in [`include/llama.h`](../../include/llama.h) until needed

Design target:

- proposal generation consumes verifier-owned state
- proposal outputs stay in the verifier runtime
- acceptance does not require verifier-to-sidecar hidden copies

Test:

- compile only at first
- add temporary assertions and invariants

Validate:

- no change in public behavior yet

### Phase 3: Implement Qwen 3.5 native MTP in one context for `-np 1`

Purpose:

- Replace the current two-context path for single-sequence decode first.

Edit:

- [`src/models/qwen35.cpp`](../../src/models/qwen35.cpp)
  - expose the model-specific native MTP proposal head in the normal graph path
- [`src/llama-graph.cpp`](../../src/llama-graph.cpp)
  - carry proposal tensors as part of the main graph result
- [`src/llama-context.cpp`](../../src/llama-context.cpp)
  - execute native MTP proposal generation from the verifier context
- [`common/speculative.cpp`](../../common/speculative.cpp)
  - switch native MTP single-slot orchestration to the new internal runtime path

Test:

- `build-cpu/bin/test-arg-parser`
- CPU validation
- CUDA validation
- direct `llama-speculative-simple` smoke test

Validate:

- exact greedy equality
- CPU still works
- CUDA `np=1` no longer depends on sidecar replay
- graph reuse/build behavior is materially better than the old sidecar design

### Phase 4: Extend the same runtime to multi-sequence `-np 2`

Purpose:

- Make the single-context path the shared implementation for concurrent server slots.

Edit:

- [`src/llama-context.cpp`](../../src/llama-context.cpp)
  - multi-sequence proposal bookkeeping
- [`src/llama-graph.cpp`](../../src/llama-graph.cpp)
  - ensure proposal outputs support multi-sequence decode cleanly
- [`tools/server/server-context.cpp`](../../tools/server/server-context.cpp)
  - remove native-MTP-specific shared-sidecar assumptions
- [`common/speculative.cpp`](../../common/speculative.cpp)
  - delete shared native-MTP second-context replay path once the new path is active

Test:

- CPU validation
- CUDA validation
- repeated `-np 2` runs to rule out slot starvation or hangs

Validate:

- exact greedy equality
- `-np 2` is stable
- no shared-sidecar timeout behavior

### Phase 5: Remove prototype-only native MTP sidecar machinery

Purpose:

- Reduce maintenance burden after the single-context path is proven.

Edit:

- [`common/speculative.cpp`](../../common/speculative.cpp)
  - remove native MTP second-context state and replay code
- [`tools/server/server-context.cpp`](../../tools/server/server-context.cpp)
  - remove setup that only exists for the sidecar path
- docs cleanup

Test:

- full validation matrix

Validate:

- behavior unchanged relative to the new runtime
- codepath count is lower
- no orphaned native MTP branches remain

## Step-by-Step Execution Checklist

This is the concrete order to follow while implementing:

1. Re-run the current stable CPU and CUDA validation harness and save the numbers.
2. Add debug-only graph reuse diagnostics and MTP timing instrumentation.
3. Confirm the current sidecar bottlenecks with logs before changing behavior.
4. Add a private `llama_context` native-MTP runtime interface.
5. Wire Qwen 3.5 native MTP proposal tensors into the normal graph path.
6. Implement single-context native MTP proposal generation for `-np 1`.
7. Validate exact greedy output equality on CPU and CUDA.
8. Compare `np=1` baseline vs native-MTP `tok/s` on CPU and CUDA.
9. Extend the same implementation to shared multi-sequence `-np 2`.
10. Validate exact greedy output equality for `-np 2`.
11. Compare `np=2` per-slot baseline vs native-MTP `tok/s`.
12. Remove the old second-context native MTP replay path only after the new path passes all gates.

## Review Checklist Before Each Commit

- Does the change preserve exact greedy output?
- Does `-np 2` still complete without hanging?
- Is the logic moving out of `common/speculative.cpp` rather than deeper into it?
- Is the implementation more backend-agnostic than before?
- Would this design still make sense for a second native-MTP model?

## Recommendation

Do not continue growing the current two-context native MTP prototype for CUDA.

Use the current branch only as:

- a correctness reference
- a CPU reference
- a source of validation tooling
- a source of operational insights about what the permanent architecture should avoid

The production-quality path is a single-context native MTP runtime.
