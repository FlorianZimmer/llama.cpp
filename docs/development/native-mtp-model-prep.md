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
- treat native-MTP tensors as a separate quality gate, not as “just another small tail tensor”

### MoE models

Be more careful with MoE models.

Plain metadata-only donor cloning is not enough if it ignores 3D expert tensors.

Lessons from `Qwen3.5-35B-A3B`:

- a naive `quant_clone` style path can miss MoE-specific expert tensors
- that can silently produce the wrong quantization plan
- use a MoE-aware tensor-type map that includes all relevant expert tensors

## MTP Tensor Quality Pass

For the current `qwen35` and `qwen35moe` native-MTP models, the MTP-specific tensor set is small:

- `*.nextn.eh_proj.weight`
- `*.nextn.enorm.weight`
- `*.nextn.hnorm.weight`
- `*.nextn.shared_head_norm.weight`

In the GGUFs prepared so far:

- the three `*.norm.weight` tensors already stay `F32`
- the only quantized MTP tensor is `*.nextn.eh_proj.weight`

That makes MTP quality tuning much simpler than a full-model dynamic quant pass: the main question is how aggressively `*.nextn.eh_proj.weight` can be quantized before native-MTP exactness or acceptance degrades.

Use the repo-local audit script to compare candidate quants against a BF16 reference and emit a `llama-quantize --tensor-type-file` override:

```bash
./.venv-tests/bin/python scripts/audit_mtp_quantization.py \
  --reference /mnt/models/GGUF/<MODEL>-MTP-bf16.gguf \
  --candidate q4=/mnt/models/GGUF/<MODEL>-MTP-Q4_K_M.gguf \
  --candidate q5=/mnt/models/GGUF/<MODEL>-MTP-Q5_K_M.gguf \
  --candidate q8=/mnt/models/GGUF/<MODEL>-MTP-UD-Q4_K_XL.gguf \
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

Interpretation:

- balanced is the recommended default for “best quality / size tradeoff”
- strict is for exactness-first experiments where a few extra MiB on the MTP head are acceptable

If the audit recommends a promotion, quantize from BF16 with the generated override file:

```bash
build/bin/llama-quantize \
  --imatrix /mnt/models/imatrix/<MODEL>.gguf \
  --tensor-type-file /tmp/<MODEL>-mtp-balanced.tensor-types.txt \
  /mnt/models/GGUF/<MODEL>-MTP-bf16.gguf \
  /mnt/models/GGUF/<MODEL>-MTP-Q4_K_M-mtp-balanced.gguf \
  Q4_K_M
```

Checked-in example override files:

- [scripts/mtp_quant_overrides/qwen35moe-a3b-q4_k_m-balanced.tensor-types.txt](../../scripts/mtp_quant_overrides/qwen35moe-a3b-q4_k_m-balanced.tensor-types.txt)
- [scripts/mtp_quant_overrides/qwen35moe-a3b-q4_k_m-strict.tensor-types.txt](../../scripts/mtp_quant_overrides/qwen35moe-a3b-q4_k_m-strict.tensor-types.txt)

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
- enable `--mtp-profile` when you need per-step acceptance and draft/accept/replay timing in the JSON output
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
- `Qwen3.5-27B` custom GGUF + quant path also worked and stayed `np=1` exact on the checked CUDA cases, but it was still speed-negative.
- `Qwen3.5-35B-A3B` custom GGUF + quant path also worked functionally.
- `Qwen3.5-35B-A3B Q4_K_M` showed that “MTP tensors preserved” is not enough by itself:
  - it was still slow
  - and it diverged from greedy baseline on the checked `bad np=1` CUDA case
- A direct BF16-vs-quant MTP audit showed why `Qwen3.5-35B-A3B Q4_K_M` is special:
  - all three `*.nextn.*norm.weight` tensors were already `F32`
  - the only differing MTP tensor across the A3B quants was `blk.40.nextn.eh_proj.weight`
  - `Q4_K_M` stored that tensor as `Q4_K`
  - `Q5_K_M` stored it as `Q5_K`
  - `UD-Q4_K_XL` stored it as `Q8_0`
  - relative to BF16, the measured error on that tensor was:
    - `Q4_K`: `rel_rmse ~= 0.0759`, cosine `~= 0.9971`
    - `Q5_K`: `rel_rmse ~= 0.0417`, cosine `~= 0.9991`
    - `Q8_0`: `rel_rmse ~= 0.0086`, cosine `~= 0.99996`
  - the balanced recommendation is therefore to promote that tensor to at least `Q5_K` in the A3B `Q4_K_M` recipe
- `Qwen3.5-27B-MTP-UD-Q4_K_XL` already stores `blk.64.nextn.eh_proj.weight` as `Q8_0`, so there is no obvious MTP-head under-quantization issue there.
- `Qwen3.5-9B` no longer had a BF16 GGUF under `/mnt/models/GGUF` during this pass, so only a surrogate audit against the existing `q8_0` file was possible.
- That surrogate audit still answered the important deployment question:
  - both shipped 9B quants already store `blk.32.nextn.eh_proj.weight` as `Q8_0`
  - so there is no pending balanced MTP-head promotion to apply on 9B
- We rebuilt `Qwen3.5-35B-A3B-MTP-Q4_K_M-fixed.gguf` with the balanced override and promoted:
  - `blk.40.nextn.eh_proj.weight: Q4_K -> Q5_K`
- That balanced rebuild is now the canonical `Q4_K_M` GGUF on disk.
- The older pre-balanced backup was removed after the swap to recover disk space under `/mnt/models`.
- Important caveat:
  - the balanced rebuild improved the MTP tensor quality as intended
  - but a narrow `bad np=1` validation still failed exactness
  - so “balanced” is the right size/quality policy for the GGUF set, but it is not enough by itself to make A3B `Q4_K_M` a correctness-clean native-MTP target
- Isolation result:
  - disabling the greedy native-MTP accept fast path with a temporary debug gate did not fix the `bad np=1` divergence
  - tracing showed `Qwen3.5-35B-A3B` is using the hybrid recurrent-backup restore path, not `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`
  - the first replayed verifier logits after restore were still correct:
    - replayed token `271` at `pos=10`
    - replay top-1 next token `248068` with `p ~= 0.91`
    - that matches the greedy baseline continuation at that point
  - the divergence appeared on the first speculative step after replay:
    - without any extra guard, native MTP continued `[271, 248068, 271, ...]`
    - greedy baseline continued `[271, 248068, 198, ...]`
  - a one-step temporary cooldown after replay restored exactness on both:
    - the short traced repro
    - the full `bad np=1` validation case
  - that narrows the remaining A3B `Q4_K_M` issue further:
    - restore+replay can rebuild a correct immediate next-token state
    - the first speculative verifier batch after replay is the piece that breaks exactness on this model/quant
- Fix now landed in the server:
  - hybrid/recurrent native-MTP slots now always force one plain verifier step immediately after a replay
  - that restored `np=1` exactness for the canonical A3B `Q4_K_M` GGUF on the checked `primary`, `good`, and `bad` CUDA cases
  - the same guard stayed exact on the checked 9B and 27B dense references, and A3B `np=2` remained stability-clean in a smoke run
  - the tradeoff is conservative: it protects the lossless contract first, and deeper equivalence cleanup can still happen later if we want to recover some of that post-replay throughput
- For MoE, the main issue was not “MTP is unsupported”; the main issue was preserving MTP tensors correctly and then accepting that runtime speedup may still be poor.
- Community GGUFs cannot be trusted to preserve MTP unless verified explicitly.
- A model or quant can be a valid native-MTP functionality target without being a speed-positive target.
- Every new quant still needs a real `np=1` exactness check; do not assume exactness transfers automatically across quants of the same model family.

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
