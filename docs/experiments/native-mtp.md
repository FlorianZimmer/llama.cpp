# Native MTP Experiment Status

- Status: On hold
- Last active: 2026-04-11
- Main branches:
  - `research/native-mtp-runtime-base`
  - `research/native-mtp-qwen35-dense-speedup`

## Summary

This branch family captures an experimental attempt to integrate native multi-token prediction (MTP / NextN) for Qwen 3.5 into `llama.cpp`.

The work was paused after the implementation reached a functional and benchmarked state but did not broaden into the speedup profile that would justify continued short-cycle optimization work.

The key practical result is:

- the native-MTP path works correctly on the kept dense Qwen 3.5 path;
- the strongest checked win was on `Qwen3.5-9B q8_0` at `np=1`;
- broader dense wins did not hold across the checked targets, especially `Qwen3.5-27B UD-Q4_K_XL`.

## What exists

- A native Qwen 3.5 MTP runtime path and branch history covering runtime, replay, visibility, and benchmark work
- CUDA validation harness work in `scripts/validate_mtp_cuda.py`
- Dense Qwen 3.5 benchmark notes and optimization plans in branch-local docs
- Preserved runtime planning notes describing why the current two-context sidecar design is not the desired long-term architecture

## Key takeaways

- Native MTP in `llama.cpp` is functionally viable for dense Qwen 3.5 under a narrow exact `np=1` setup.
- Small local graph/runtime changes were enough to recover and extend a real single-user win on `Qwen3.5-9B q8_0`.
- The current one-token runtime design does not scale into a broad dense speedup story across the checked quants and model sizes.
- The remaining gap looks structural rather than like a missing easy fast path.

## Main limitations

- The current runtime is still effectively one drafted token per verifier step.
- Speculative state is still managed through restore / replay behavior rather than explicit branch-state storage inside the runtime.
- That keeps amortization too low on the heavier dense targets.
- Some hybrid/recurrent paths also remain sensitive to `np > 1` verifier numerics and replay behavior.

## Why it is on hold

The work was paused because the remaining steps no longer looked like small, maintainable optimizations.

Closing the remaining gap would likely require deeper runtime changes such as:

- explicit speculative branch-state storage
- lower-level draft / verify / commit integration
- true multi-token native drafting

Those are larger architectural projects, so the current branch family is kept as a research snapshot rather than being pushed further as an active optimization series.

## Current caveats

- Experimental and branch-scoped
- Benchmark-positive only in a narrow dense-Qwen setup
- Not upstreamed
- Kept as a research snapshot rather than a supported feature
