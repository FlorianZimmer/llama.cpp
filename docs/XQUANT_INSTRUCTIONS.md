# XQuant Implementation Instructions

## 0) Mission & Non-Negotiables

**Mission:** Implement **XQuant** (post-norm-X caching + K/V rematerialization on demand), **XQuant-CL** (cross-layer deltas + accumulator), and **XQuant-GQA+SVD** (latent caching for grouped-query attention) in `llama.cpp`.

**Hard rules:**

1. **No wrappers. No delegation.** When XQuant is enabled, do **not** call or rely on any `llama_kv_cache*` codepaths (read/write/evict).
2. **Mutual exclusivity:** With `--xquant` or `--xquant-cl`, instantiate **only** the XQuant memory module and **do not** allocate/init KV.
3. **Flags off = baseline:** With flags off, behavior/perf **unchanged**.
4. **Hook sites:** Taps go in **arch-specific builders invoked from `llama-model.cpp`** (e.g., `llama-graph.cpp` / `llm_build_*`), **exactly post-LayerNorm and pre-QKV**.

---

## 1) Flags, Parameters, Exclusivity (Phase 1)

**Implement first.** Follow existing argument/env patterns.

**CLI & env (in `common/arg.cpp`, `common/arg.h`):**

- `--xquant` (enable XQuant; disables KV)
- `--xquant-cl` (enable CL; implies `--xquant`)
- `--xq-bits {8,4,3,2}` (default: 4)
- `--xq-group <int>` (default: 128)
- `--xq-base-layers <int>` (default: 3) — early layers pinned to 4-bit
- `--xq-gqa-svd` (enable latent caching for GQA)
- `--xq-svd-rank {auto|int}`
- `--xq-svd-path <file-or-dir>`
- Provide ENV mirrors (`LLAMA_XQUANT`, `LLAMA_XQ_CL`, `LLAMA_XQ_BITS`, etc.) via `common::arg::set_env` pattern.

**Plumb to `llama_cparams`** (`src/llama-cparams.h/.cpp`) and through context init (`src/llama-context.cpp`).

**Exclusivity guard:** In arg parsing, **fail** if any `--kv-*` flag appears with `--xquant*`.

**Runtime assert:** After factory selection, **assert KV pointers are null** when XQuant is enabled.

---

## 2) Module Files & Build Wiring (Phase 1)

- **New source files:**
  - `src/llama-memory-xquant.h`
  - `src/llama-memory-xquant.cpp`      (XQuant MHA)
  - `src/llama-memory-xquant-cl.cpp`   (CL deltas + accumulator)
  - `src/llama-xq-quant.cpp`           (quant helpers: pack/unpack, scaling)
  - `tools/xqsvd/xqsvd.cpp`            (offline SVD tool)

- **CMake:**
  - Add sources to `src/CMakeLists.txt`; include headers as needed.
  - **Add option:** `option(LLAMA_BUILD_XQ_TOOLS "Build XQuant tools" ON)`. When ON, build `tools/xqsvd`.
  - Keep all builds passing with flags OFF.

---

## 3) Factory Switch in `llama-model.cpp` (Phase 1)

- In model construction, if any `xquant*` flag is set:
  - Instantiate **`llama_memory_xquant`** (or CL variant) via the project’s memory-module factory pattern.
  - **Skip** creation/initialization of **any** `llama_kv_cache`.
  - **LLAMA_ASSERT** KV is not present.

---

## 4) SVD Contract & Loader (Phase 2)

- **Fail-fast rule:** If `--xq-gqa-svd` is set and required blobs are missing/unreadable, **error out** with a clear message. Do **not** silently downgrade.

- **On-disk schema (single file or directory):**
  - **Header:** magic `"XQSV1"`, `u32 version`, model dims, per-layer ranks, dtype tags, layout flags.
  - **Payloads per layer:** `Uk`, `Uv`, optional `Ukv` (row-major), and fused small squares `ΣkBkᵀ`, `ΣvBvᵀ` (contiguous arrays).

- **Search order:**
  1. `--xq-svd-path` if given;
  2. alongside the GGUF (same basename + `.xqsvd`);
  3. model cache directory (if project has one).

- **Loader:** Parse the schema at model load and **store factors** in the XQuant memory module state.

- **Tool (`tools/xqsvd/xqsvd.cpp`):**
  - Reads model weights (`Wk`, `Wv`), computes SVD, writes blobs per the schema above.

---

## 5) Graph Hooks & Write Path (Phase 3)

- **Hook site:** In the **arch-specific builder** the model selects (invoked from `llama-model.cpp`), insert a tap **immediately after post-LayerNorm, pre-QKV**:

```cpp
if (xquant_enabled) xq_ctx.write(layer_id, token_index, X_post_ln);
```

- **Write behavior:**
  - **MHA:** Quantize `X` (asymmetric uniform, `group=128`), store data + scales/zp.
  - **CL:** Maintain per-layer **accumulator**; compute `ΔX = X − Accum_{i-1}`, quantize/store `ΔX`, update `Accum_i`.
  - **GQA+SVD:** Compute `Xk = X·Uk` (**per-channel** quant) and `Xv = X·Uv` (**per-token** quant); quantize/store both.

- **Sequence management:** Implement `seq_add`, `seq_rm`, `seq_cp`, **sliding-window eviction** mirroring KV semantics, now applied to X/ΔX/latent streams.

---

## 6) Read Path & Rematerialization (Phase 4)

- Modify attention build so **past memory** comes **only** from XQuant:
  1. Dequantize cached **X / ΔX / latent** and reconstruct `X̂`:
     - **MHA:** `X̂ ← dequant(X)`
     - **CL:** `X̂ ← Accum + dequant(ΔX)` (or `Accum + dequant(ΔX·Ukv)·Ukvᵀ` for CL+SVD)
     - **GQA+SVD:** dequantize `Xk`, `Xv`
  2. **Rematerialize K/V**:
     - **MHA:** `K = X̂·Wk`, `V = X̂·Wv`
     - **GQA+SVD:** `K = Xk·(ΣkBkᵀ)`, `V = Xv·(ΣvBvᵀ)`
  3. **RoPE timing:** Apply RoPE to `K,V` (and to `Q`) **after** rematerialization (caches are pre-RoPE).

- **Backend staging:** Reuse existing CPU/GPU staging patterns used by KV (e.g., hybrid helpers) for X/ΔX/latent and small matrices.

- **No KV usage:** Ensure attention code paths make **no** calls to any KV APIs when XQuant is enabled.

---

## 7) Quantization Policy (applies in Phases 3–4)

- **Asymmetric uniform; group = 128.**
- **Per-channel** scaling for key-like streams (**`Xk`**), **per-token** for value-like (**`Xv`**, many deltas).
- Force **first `xq_base_layers` (~3) at 4-bit**, even if global bits lower.
- During decode, keep **trailing ≤ group-size tokens unquantized** to stabilize per-channel scales.

---

## 8) Validation, Perf, Docs (Phase 5)

- **Static guard:** Add a CI/unit test that greps build artifacts to confirm **no `llama_kv_cache` symbols** are referenced when `--xquant*` is active.

- **Unit tests (new `tests/`):**
  - **Quant helpers:** pack/unpack correctness across bit-widths and group sizes; per-axis scale application.
  - **CL accumulator:** reconstruction parity (`X̂ = Accum + ΔX`), drift bounds over long sequences.
  - **SVD loader:** schema parse, dtype/layout validation, per-layer rank checks.
  - **GQA+SVD remat:** `Xk·(ΣkBkᵀ)` / `Xv·(ΣvBvᵀ)` shape & numeric sanity; compare to direct `X·Wk/Wv` at 8-bit.
  - **RoPE parity:** K/V/Q angular equivalence vs baseline timing (post-remat).
  - **Seq ops & eviction:** `seq_add/seq_rm/seq_cp`, sliding window, prefix reuse on X/ΔX/latent streams.
  - **Exclusivity:** enabling `--xquant*` ensures **no KV allocations**; enabling KV without `--xquant*` preserves baseline.

- **Perf sanity (scripts under `/Users/florian/Local/test-xquant/`):**
  - **Microbench script**: runs `tools/llama-bench` with representative settings, reporting **tokens/s** and **peak RSS/VRAM** for baseline vs XQuant modes (MHA 4-bit; CL 3-bit; GQA+SVD 2–3-bit).
  - **PPL smoke test target**: quick WikiText-2 run to ensure 4-bit MHA ≤ 0.1 PPL delta; CL 3-bit ≈ ≤ 0.02; 2-3-bit GQA+SVD ≤ ~1 PPL (ballpark). The goal is **reproducibility**, not exhaustive eval.

- **Docs:**
  - Flags + ENV, exclusivity with KV, RoPE timing, SVD workflow (tool usage & on-disk schema), diagrams (below), and example invocations.

---

## 9) Before/After Diagrams (include in docs)

**Current (Baseline KV):**

```mermaid
flowchart LR
  Tokens -->|Prefill| Graph
  Graph --> ProjQKV[Q/K/V projections]
  ProjQKV --> KVWrite[Write K/V to KV cache (pre-RoPE)]
  loop Decode
    NewTok --> Graph
    Graph --> ProjQ[Q for new token]
    KVRead[Read K/V for past tokens] --> RoPE
    ProjQ --> RoPE[Apply RoPE to K,V,Q]
    RoPE --> Attn[Attention] --> FFN --> Next
  end
```

**Planned (XQuant Full Replacement):**

```mermaid
flowchart LR
  Tokens -->|Prefill| Graph
  Graph --> PostLN[Post-LayerNorm X (per layer)]
  PostLN --> XQWrite[Quantize & Write: X / ΔX / Xk|Xv (latent)]
  loop Decode
    NewTok --> Graph
    Graph --> PostLN2[Post-LayerNorm X (current layer)]
    XQRead[Read X/ΔX/latent; reconstruct X̂ if CL]
    XQRead --> Remat[Rematerialize: K = X̂·Wk, V = X̂·Wv (or small GEMMs for SVD)]
    Remat --> RoPE[Apply RoPE to K,V,Q]
    PostLN2 --> QProj[Q for new token]
    QProj --> RoPE
    RoPE --> Attn[Attention] --> FFN --> Next
  end
```

---

## 10) Acceptance Criteria

- **Correctness:** Flags OFF identical to baseline; flags ON allocate **no KV** and run solely via XQuant. RoPE applied **after** remat.
- **Features:** MHA, CL, and GQA+SVD paths implemented; SVD blobs required when `--xq-gqa-svd` is set (fail fast if missing).
- **Performance:** Memory reduction as designed; microbench + RSS show expected improvements; throughput neutral or better in memory-bound regimes.
- **Tests:** All unit tests pass; static grep guard passes; PPL smoke tests within targets.
- **Docs:** Clear usage, schema, and diagrams; CMake option `LLAMA_BUILD_XQ_TOOLS` defaults **ON**.

---

## 11) Implementation Order (execute now)

1. **Phase 1:** CLI/env + `llama_cparams` + factory switch in `llama-model.cpp`; add module files & CMake; assert no KV when XQuant on.
2. **Phase 2:** SVD schema/loader + `tools/xqsvd` (fail fast if blobs missing).
3. **Phase 3:** Graph hook (arch-specific builder) post-LN pre-QKV; implement write paths (MHA, CL, GQA+SVD) + seq/eviction.
4. **Phase 4:** Read/remat (MHA, CL, GQA+SVD) + RoPE timing + backend staging.
5. **Phase 5:** Static guard, unit tests, microbench + PPL smoke, docs/diagrams.

**Do not introduce any wrapper or fallback to KV** when `--xquant*` is active. Verify with asserts and CI grep.
