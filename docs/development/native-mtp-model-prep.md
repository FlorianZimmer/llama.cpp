# Native MTP Model Prep

This note captures the workflow and sharp edges we hit while preparing real models for native-MTP testing in this private mirror.

## Storage Rules

- Store downloads and generated GGUFs under `/mnt/models`.
- Use the Ubuntu-native ext4 drive only.
- Do not use `/home` for large model files.
- Do not use Windows-mounted paths under `/media`.

## What To Verify Before Download

For any candidate model, first confirm that the original HF checkpoint actually contains MTP / NextN weights.

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

This already happened with multiple Qwen3.5-35B-A3B community GGUFs:

- `unsloth`
- `lmstudio-community`
- `bartowski`
- `AesSedai`

They loaded as normal models but dropped the MTP metadata / tensors needed by this implementation.

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

- architecture is the expected native-MTP arch
- `*.nextn_predict_layers` exists
- `blk.*.nextn.eh_proj.weight`
- `blk.*.nextn.enorm.weight`
- `blk.*.nextn.hnorm.weight`
- any shared-head or dedicated-embedding tensors expected by the model

## Quantization Guidance

### Dense models

For dense models, prefer a high-quality dynamic quant pattern rather than a naive all-`Q4_K_M` conversion.

Practical approach:

- follow Unsloth Dynamic 2.0 style mappings
- use a donor quant layout when possible
- build an imatrix first
- then quantize with explicit tensor overrides

### MoE models

Be more careful with MoE models.

Plain metadata-only donor cloning is not enough if it ignores 3D expert tensors.

Lessons from `Qwen3.5-35B-A3B`:

- a naive `quant_clone` style path can miss MoE-specific expert tensors
- that can silently produce the wrong quantization plan
- use a MoE-aware tensor-type map that includes all relevant expert tensors

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
- `np>1` on the current hybrid/recurrent native-MTP runtime is stability-focused, not strict batch-invariant exactness
- draft activity should still be non-zero
- speedup is workload-dependent and not guaranteed

## Cleanup Rules

After a model has been successfully quantized and validated:

- keep the final GGUF(s)
- keep the HF source tree until you are done with all derived quants you may need
- delete redundant BF16 GGUF intermediates
- delete stale experimental quants that are no longer needed

This matters because `/mnt/models` can fill up quickly during BF16 and multi-quant experiments.

## What We Learned So Far

- The native runtime currently works on:
  - `qwen35`
  - `qwen35moe`
- `Qwen3.5-9B` custom GGUF + quant path worked and preserved MTP correctly.
- `Qwen3.5-35B-A3B` custom GGUF + quant path also worked functionally.
- For MoE, the main issue was not “MTP is unsupported”; the main issue was preserving MTP tensors correctly and then accepting that runtime speedup may still be poor.
- Community GGUFs cannot be trusted to preserve MTP unless verified explicitly.
- A model can be a valid native-MTP functionality target without being a speed-positive target.

## Paste-Ready Prompt For External AI

```text
Please help me prepare a Qwen 3 27B model for native-MTP testing in my private llama.cpp mirror.

Important constraints:
- All downloads and generated model files must go under /mnt/models.
- Use the Ubuntu-native ext4 drive only.
- Do not use /home for large files.
- Do not use Windows-mounted paths under /media.
- Do not assume community GGUFs preserve MTP.

What I need from you:
1. First identify the exact HF model that “Qwen 3 27B” should refer to.
2. Before downloading any large GGUF, verify whether the official HF checkpoint actually contains MTP / NextN weights:
   - inspect config.json for fields like mtp_num_hidden_layers
   - inspect model.safetensors.index.json for mtp.* tensors
3. Check whether a popular community GGUF already preserves MTP correctly by verifying that its GGUF contains:
   - *.nextn_predict_layers
   - blk.*.nextn.* tensors
4. Only reuse a community GGUF if those MTP fields are truly present.
5. If community GGUFs strip MTP, then build our own GGUF from the official HF checkpoint using this private mirror’s convert_hf_to_gguf.py.
6. After conversion, verify the output GGUF contains the expected MTP metadata and tensors before quantizing it.
7. Produce a high-quality quant, not a naive one-size-fits-all quant:
   - for dense models, prefer an Unsloth Dynamic 2.0 style dynamic quant layout
   - build an imatrix first
   - use explicit tensor overrides where needed
8. Validate the final GGUF with the local CUDA validator.

Known repo-specific context:
- This private mirror already has native-MTP conversion/runtime support for qwen35 and qwen35moe.
- The runtime currently validates np=1 exactness; np>1 on hybrid/recurrent native-MTP is stability-focused, not strict batch-invariant exactness.
- We previously found that several community Qwen3.5-35B-A3B GGUFs from unsloth, lmstudio-community, bartowski, and AesSedai dropped the MTP metadata/tensors entirely.
- We successfully built our own MTP-preserving GGUFs for Qwen3.5-9B and Qwen3.5-35B-A3B.
- For MoE models, naive quant-clone style tensor mapping was not enough because it missed 3D expert tensors.

Please return:
- the exact candidate model choice
- whether it truly contains MTP
- whether an existing GGUF can be reused
- if not, the exact download / convert / verify / imatrix / quantize / validate plan
- expected disk usage and cleanup recommendations

If you need extra context from public llama.cpp or model repos, use it surgically rather than reading the whole project.
```
