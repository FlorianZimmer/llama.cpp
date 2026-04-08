# XQuant Experiment Status

- Status: On hold
- Last active: 2025-08-24
- Main branch: `research/xquant-on-hold`

## Summary

This branch captures an experimental attempt to integrate an XQuant-based KV rematerialization path into `llama.cpp`.

The work was paused before it reached a production-ready or upstreamable state, but the implementation is kept because it documents the design direction, the tradeoffs explored, and the validation work completed so far.

## What exists

- A prototype XQuant memory path and wrapper in `src/llama-memory-xquant.cpp` and `src/llama-memory-xquant-wrap.cpp`
- Integration work in the KV cache and attention path
- Focused tests in `tests/test-xquant.cpp` and `tests/test-xquant-wrap.cpp`
- A Windows evaluation script in `xquant-eval.ps1`

## Key takeaways

- The prototype showed that an XQuant-backed memory path could be integrated behind the KV cache interface instead of requiring a broad rewrite across attention code paths.
- The most viable direction was to keep the feature gated, preserve baseline behavior when disabled, and concentrate the experimental logic in the memory wrapper and KV access path.
- The remaining work was not around basic integration alone, but around stability, broader validation, and deciding whether the extra complexity was justified for a maintainable long-term branch.

## Current caveats

- Experimental and incomplete
- Not upstreamed
- Not production-ready
- Kept as a research snapshot rather than a supported feature

## Why it is on hold

The work was paused due to competing priorities before stability, validation, and cleanup reached the standard required for a maintained branch.
