# XQuant Technical Spec Capsule

## Core Concept

- Cache a quantized copy of the post-LayerNorm tensor `X` for every layer instead of caching `K` and `V`. During attention, reconstruct `K`/`V` on demand (`K = X̂·Wk`, `V = X̂·Wv`). This replaces two cached tensors with one that quantizes more effectively and slashes memory bandwidth.
- **XQuant-CL:** Store layer-wise deltas `ΔX = X − Accum_{i-1}` plus a small accumulator. Deltas have a narrow dynamic range, so ultra-low bit widths (2–4 bit) remain stable.
- **XQuant-GQA+SVD:** Apply an offline SVD to `Wk`/`Wv` → `Wk ≈ Uk Σk Bkᵀ`, `Wv ≈ Uv Σv Bvᵀ`. Cache latent projections `Xk = X·Uk` and `Xv = X·Uv` (quantized), then rematerialize using the fused matrices `(ΣkBkᵀ)` and `(ΣvBvᵀ)` during decode.

## Quantization Rules

- Asymmetric uniform quantization, **group size = 128**.
- **Per-channel scaling** for “key-like” streams (`Xk`).
- **Per-token scaling** for “value-like” streams (`Xv`, CL deltas).
- Force the first ~3 transformer layers to use **4-bit** even if global bits are lower.
- During decode, keep the trailing ≤ group-size tokens **unquantized** to maintain stable scales.

## Prefill vs Decode Behavior

- **Prefill (per layer, per token):**
  - Capture post-LN `X`.
  - **MHA:** quantize/store `X`.
  - **GQA:** compute `Xk = X·Uk` (per-channel quant) and `Xv = X·Uv` (per-token quant), store both.
  - **CL:** compute `ΔX = X − Accum_{i-1}` (or latent deltas via `Ukv`), quantize/store, then update `Accum_i`.

- **Decode:**
  1. Load cached `X`, `ΔX`, or latent tensors.
  2. Reconstruct `X̂` (add accumulator for CL if needed).
  3. Rematerialize:
     - **MHA:** `K = X̂·Wk`, `V = X̂·Wv`
     - **GQA+SVD:** `K = Xk·(ΣkBkᵀ)`, `V = Xv·(ΣvBvᵀ)`
  4. Apply RoPE to `K`, `V`, and `Q` **after** rematerialization.

## Why It Helps

- Transformer decoding is memory-bandwidth bound. XQuant trades additional compute (rematerialization GEMMs) for dramatically reduced memory traffic/cache pressure, improving throughput especially on constrained GPUs/CPUs.

## Implementation Plan (Commits per Bullet)

### Phase 1 – Flags & Factory Scaffolding

- **CLI plumbing + exclusivity guard**  
  Files: `common/arg.cpp`, `common/arg.h`, `src/llama-cparams.h`, `src/llama-context.cpp`.  
  Add `--xquant`, `--xquant-cl`, `--xq-bits`, `--xq-group`, `--xq-base-layers`, `--xq-gqa-svd`, `--xq-svd-rank`, `--xq-svd-path`. Emit errors if any `--kv-*` flag coexists with `--xquant*`. Extend `llama_cparams` and propagate through context init.

- **XQuant memory stubs + CMake wiring**  
  Create `src/llama-memory-xquant.h`, `src/llama-memory-xquant.cpp`, `src/llama-memory-xquant-cl.cpp`, `src/llama-xq-quant.cpp`. Update `src/CMakeLists.txt` and top-level build. Define skeleton classes (`llama_memory_xquant`, `llama_memory_context_xquant`) without functionality.

- **Factory switch + runtime assert**  
  File: `src/llama-model.cpp`. If any XQuant flag is set, instantiate `llama_memory_xquant` and skip `llama_kv_cache` allocation. `LLAMA_ASSERT` that KV pointers remain null when XQuant is enabled.

### Phase 2 – SVD Contract & Loader

- **On-disk schema + loader hooks**  
  Files: `src/llama-memory-xquant.cpp`, `src/llama-model.cpp`. Define header `{magic, version, dims, dtype, layout}`; payload order `Uk`, `Uv`, optional `Ukv`, fused `(ΣkBkᵀ)`, `(ΣvBvᵀ)`. Search path: explicit `--xq-svd-path` → GGUF sibling `.xqsvd` → model cache. Parse into XQuant memory state; fail fast if blobs missing when `--xq-gqa-svd`.

- **Offline tool scaffolding**  
  File: `tools/xqsvd/xqsvd.cpp` + CMake target. Stub reads model matrices, emits the schema.

### Phase 3 – Graph Hooks & Write Path

- **Post-LN tap**  
  Files: architecture builders (`src/llama-graph.cpp`, `src/llm_build_*.cpp`). After post-LayerNorm, before Q/K/V projections, call `xq_ctx->write(layer, token, X)`.

- **MHA write path**  
  File: `src/llama-memory-xquant.cpp`. Quantize post-LN `X` via helper functions; store data plus scale/zp metadata.

- **CL delta caching**  
  File: `src/llama-memory-xquant-cl.cpp`. Maintain per-layer accumulator, compute `ΔX`, quantize/store, update accumulator.

- **GQA+SVD latent caching**  
  File: `src/llama-memory-xquant.cpp`. Multiply `X` by `Uk`/`Uv` to get `Xk`/`Xv`, apply per-channel/per-token quantization, store.

- **Sequence management mirrors**  
  File: `src/llama-memory-xquant.cpp`. Implement `seq_add`, `seq_rm`, `seq_cp`, sliding-window eviction mirroring KV semantics for X/ΔX/latent storage.

### Phase 4 – Read Path & Rematerialization

- **MHA rematerialization**  
  File: `src/llama-memory-xquant.cpp`. Dequantize cached `X` to `X̂`, compute `K = X̂·Wk`, `V = X̂·Wv`, then route through existing RoPE builders.

- **CL reconstruction**  
  File: `src/llama-memory-xquant-cl.cpp`. Rebuild `X̂ = Accum + ΔX` (or latent equivalent) before matmuls.

- **GQA+SVD GEMMs**  
  File: `src/llama-memory-xquant.cpp`. Dequantize `Xk`/`Xv`, multiply with `(ΣkBkᵀ)` / `(ΣvBvᵀ)` to form `K`/`V`.

- **Backend staging reuse**  
  Files: `src/llama-memory-xquant.cpp`, `src/llama-memory-hybrid.cpp` (if needed). Reuse KV staging helpers for CPU/GPU transfers.

- **RoPE & Q path**  
  Ensure rematerialized `K`/`V` receive RoPE post-processing; `Q` pipeline unchanged.

### Phase 5 – Validation & Docs

- **Static grep guard**  
  Add CI/unit test asserting no `llama_kv_cache` references when compiling XQuant builds/flags.

- **Unit tests & microbenches**  
  Test quant pack/unpack, sequence ops, rematerialization correctness, RoPE parity. Integrate `/Users/florian/Local/test-xquant/test.sh` for reproducible benchmarks (tokens/s, RSS, PPL targets).

- **Documentation**  
  Update README/docs with flag descriptions, SVD workflow, memory savings, diagrams.

- **Final verification**  
  Ensure flags-off path matches baseline; run builds/tests; confirm complete KV replacement, SVD tooling, and documentation.
