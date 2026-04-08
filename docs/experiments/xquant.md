# XQuant Experiment Status

- Status: On hold
- Last active: 2025-08-24
- Main branch: `research/xquant-on-hold`
- Archived raw history: `archive/xquant-raw-2025-08`

## Summary

This branch captures an experimental attempt to integrate an XQuant-based KV rematerialization path into `llama.cpp`.

The work was paused before it reached a production-ready or upstreamable state, but the implementation is kept because it documents the design direction, the tradeoffs explored, and the validation work completed so far.

## What exists

- A prototype XQuant memory path and wrapper in `src/llama-memory-xquant.cpp` and `src/llama-memory-xquant-wrap.cpp`
- Integration work in the KV cache and attention path
- Focused tests in `tests/test-xquant.cpp` and `tests/test-xquant-wrap.cpp`
- A Windows evaluation script in `xquant-eval.ps1`

## Current caveats

- Experimental and incomplete
- Not upstreamed
- Not production-ready
- Kept as a research snapshot rather than a supported feature

## Why it is on hold

The work was paused due to competing priorities before stability, validation, and cleanup reached the standard required for a maintained branch.
