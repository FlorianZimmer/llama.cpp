# XQuant: KV-Free Memory for llama.cpp

XQuant replaces the traditional K/V cache with a compact stream of quantized post–layer-norm activations. K and V are rematerialized on demand, trading a small amount of extra compute for a large reduction in memory traffic.

## Flags and Environment Variables

| CLI Flag | Environment | Description |
|----------|-------------|-------------|
| `--xquant` | `LLAMA_XQUANT` | Enable XQuant (disables KV cache) |
| `--xquant-cl` | `LLAMA_XQ_CL` | Cross‑layer delta mode; implies `--xquant` |
| `--xq-bits <2|3|4|8>` | `LLAMA_XQ_BITS` | Bit width for stored activations (default 4) |
| `--xq-group <int>` | `LLAMA_XQ_GROUP` | Quantization group size (default 128) |
| `--xq-base-layers <int>` | `LLAMA_XQ_BASE_LAYERS` | Early layers pinned to 4‑bit (default 3) |
| `--xq-gqa-svd` | `LLAMA_XQ_GQA_SVD` | Use latent caching for GQA via SVD |
| `--xq-svd-rank <auto|int>` | `LLAMA_XQ_SVD_RANK` | Rank for SVD factors |
| `--xq-svd-path <path>` | `LLAMA_XQ_SVD_PATH` | Location of `.xqsvd` blobs |

XQuant flags are **mutually exclusive** with all `--kv-*` options. The model factory asserts that no KV cache is present whenever XQuant is active.

## RoPE Timing

When using XQuant, cached activations are stored **pre‑RoPE**. RoPE is applied only after K/V rematerialization, keeping the on‑disk representation agnostic to position.

## SVD Workflow

1. Run the `xqsvd` tool to generate factor files:
   ```
   ./xqsvd -m model.gguf -o model.xqsvd
   ```
2. Place the resulting `model.xqsvd` alongside the GGUF or pass `--xq-svd-path` when running inference.
3. At load time, the runtime parses the `XQSV1` header, validates layer ranks and stores the factors for rematerialization.

## Example Invocations

```bash
# Plain XQuant, 4‑bit
./main -m model.gguf --xquant --xq-bits 4

# Cross-layer deltas at 3‑bit
./main -m model.gguf --xquant-cl --xq-bits 3

# GQA + SVD latent caching
./main -m model.gguf --xquant --xq-gqa-svd --xq-svd-path model.xqsvd
```

## Bench & PPL Scripts

Helper scripts live under `tools/bench/`:

- `xq-bench.sh` – runs `llama-bench` for baseline vs XQuant configurations and reports tokens/s and peak memory.
- `xq-ppl.sh` – quick WikiText‑2 smoke test that compares perplexity deltas across modes.

## Diagrams

### Baseline KV Cache

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

### XQuant Full Replacement

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

## Notes

XQuant remains experimental; accuracy and performance characteristics may vary across models and hardware.
