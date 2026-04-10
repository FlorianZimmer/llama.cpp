# Native MTP Model Prep

This note captures the dense-model workflow and sharp edges for native-MTP testing in this private mirror.

## Current V1 Scope

For the first upstream-oriented native-MTP series in this private mirror:

- keep dense `qwen35` in scope
- active speed target: `Qwen3.5-9B q8_0`
- supporting dense regression coverage:
  - `Qwen3.5-9B UD-Q4_K_XL`
  - `Qwen3.5-27B UD-Q4_K_XL`
- keep `np=1` exactness against greedy baseline

Removed from the live V1 prep branch:

- `qwen35moe`
- `Qwen3.5-35B-A3B`
- MoE-specific quantization overrides

## Storage Rules

- store downloads and generated GGUFs under `/mnt/models`
- use the Ubuntu-native ext4 drive only
- do not use `/home` for large model files
- do not use Windows-mounted paths under `/media`

## What To Verify Before Download

For any dense candidate model, confirm that the original HF checkpoint actually contains MTP / NextN weights.

Check:

- `config.json` for MTP-related fields such as `mtp_num_hidden_layers`
- `model.safetensors.index.json` for `mtp.*` tensors
- model card / config naming only as a hint, not as proof

If the HF checkpoint does not contain MTP tensors, it is not a useful native-MTP target for this runtime.

## What To Verify Before Reusing A Community GGUF

Do not assume community GGUFs preserve MTP.

Before downloading a large GGUF, inspect the header / tensor listing and verify that it contains:

- `*.nextn_predict_layers` metadata
- `blk.*.nextn.*` tensors

If those are missing, the GGUF is usable as a plain model but not for the native-MTP path in this repo.

## Preferred Conversion Path

If the HF checkpoint contains MTP tensors and community GGUFs do not preserve them, build a GGUF locally from the official HF weights using this private mirror.

Typical conversion command:

```bash
env PYTHONPATH=gguf-py .venv-tests/bin/python convert_hf_to_gguf.py \
  /mnt/models/hf/<MODEL_DIR> \
  --outfile /mnt/models/GGUF/<MODEL_NAME>-MTP-bf16.gguf \
  --outtype bf16
```

After conversion, verify the GGUF directly before spending time on quantization.

Expected signals:

- architecture is the expected dense native-MTP arch
- `*.nextn_predict_layers` exists
- `blk.*.nextn.eh_proj.weight`
- `blk.*.nextn.enorm.weight`
- `blk.*.nextn.hnorm.weight`
- any shared-head or dedicated-embedding tensors expected by the model

## Dense Quantization Guidance

For dense models, prefer a high-quality dynamic quant pattern rather than a naive all-`Q4_K_M` conversion.

Practical approach:

- follow Unsloth Dynamic 2.0 style mappings when possible
- use a donor quant layout when possible
- build an imatrix first
- quantize with explicit tensor overrides if needed
- treat native-MTP tensors as a separate quality gate, not as “just another small tail tensor”

## MTP Tensor Quality Pass

For the current dense `qwen35` native-MTP models, the MTP-specific tensor set is small:

- `*.nextn.eh_proj.weight`
- `*.nextn.enorm.weight`
- `*.nextn.hnorm.weight`
- `*.nextn.shared_head_norm.weight`

In the dense GGUFs prepared so far:

- the three `*.norm.weight` tensors already stay `F32`
- the only quantized MTP tensor is `*.nextn.eh_proj.weight`

Use the repo-local audit script to compare candidate quants against a BF16 reference and emit a `llama-quantize --tensor-type-file` override when needed:

```bash
./.venv-tests/bin/python scripts/audit_mtp_quantization.py \
  --reference /mnt/models/GGUF/<MODEL>-MTP-bf16.gguf \
  --candidate q4=/mnt/models/GGUF/<MODEL>-MTP-Q4.gguf \
  --candidate q8=/mnt/models/GGUF/<MODEL>-MTP-q8_0.gguf \
  --baseline q4 \
  --write-balanced-type-file /tmp/<MODEL>-mtp-balanced.tensor-types.txt \
  --write-strict-type-file /tmp/<MODEL>-mtp-strict.tensor-types.txt
```

Current default thresholds in the script:

- balanced:
  - `rel_rmse <= 0.05`
  - `cosine >= 0.999`
- strict:
  - `rel_rmse <= 0.02`
  - `cosine >= 0.9999`

Current dense reading:

- `Qwen3.5-9B` shipped quants already store `blk.32.nextn.eh_proj.weight` as `Q8_0`
- `Qwen3.5-27B-MTP-UD-Q4_K_XL` already stores `blk.64.nextn.eh_proj.weight` as `Q8_0`
- so there is no current dense MTP-head under-quantization fix waiting to be applied

## Validation Contract

After quantization, validate both correctness and runtime behavior.

Use:

```bash
python3 scripts/validate_mtp_cuda.py \
  --binary build-cuda-server/bin/llama-server \
  --model /mnt/models/GGUF/<MODEL>.gguf \
  --prompt 'Write one short sentence about Berlin.' \
  --seed 42 \
  --n-predict 12 \
  --ctx-size 4096 \
  --batch-size 128 \
  --ubatch-size 128 \
  --threads 4 \
  --threads-batch 4 \
  --ngl 99 \
  --flash-attn on \
  --startup-timeout 240 \
  --draft-max 1 \
  --port-base 18880
```

Interpret results carefully:

- `np=1` exactness matters
- `np>1` on the current hybrid path is stability-focused, not strict batch-invariant exactness
- enable `--mtp-profile` when you need per-step acceptance and draft/accept/replay timing in the JSON output
- draft activity should still be non-zero
- speedup is workload-dependent and not guaranteed

## Cleanup Rules

After a model has been successfully quantized and validated:

- keep the final GGUFs
- keep the HF source tree until all derived quants are done
- delete redundant BF16 GGUF intermediates
- delete stale experimental quants that are no longer needed

This matters because `/mnt/models` can fill up quickly during BF16 and multi-quant experiments.

## What We Learned So Far

- dense `qwen35` native MTP works functionally on the prepared 9B and 27B GGUFs
- `Qwen3.5-9B q8_0` is the only checked dense path with a repeatable `np=1` win today
- `Qwen3.5-9B UD-Q4_K_XL` improved with the dense replay-policy narrowing but is still not a clean speed-positive target
- `Qwen3.5-27B UD-Q4_K_XL` stayed exact on the checked CUDA cases but is still speed-negative
- the remaining dense problem is runtime economics, not missing MTP metadata or an obvious dense MTP-head quantization bug
