# Native MTP Execution Plan

## 1) Understanding

- Start from clean upstream `master`, keep only `scripts/validate_mtp_cuda.py`, and design native MTP fresh for upstream maintainability rather than preserving the private fork's structure.
- Target Qwen 3.5 first, but make the runtime/backend architecture generic enough to support other native-MTP-capable models later.
- Prefer a first-class single-context runtime in `llama_context` unless a stronger correctness/performance case appears for something else.
- Success means: exact greedy output equality vs baseline, `llama-server` works with `-np 1` and `-np 2`, normal greedy decode is unchanged when MTP is off, and baseline vs MTP tok/s is measured and compared.

Assumptions:

- Initial bring-up should target the text / LM-only path first.
- `llm_build_qwen35moe` may be in `src/models/qwen35.cpp` or an adjacent upstream file; use the file that actually owns that symbol.

Success criteria:

- CPU and GPU exact greedy equality vs baseline.
- `llama-server` exactness with `-np 1` and `-np 2`.
- Baseline greedy behavior unchanged when MTP is disabled.
- MTP tok/s recorded for baseline vs MTP, with a clear promotion gate before upstream PR.

## 2) Repo Map (relevant only)

- `common/speculative.cpp` / `common/speculative.h`: current speculative orchestration, including the sidecar draft-context pattern to avoid repeating.
- `common/common.h`: `common_speculative_type` and `common_params_speculative`; public config surface that needs an `mtp` mode.
- `common/arg.cpp`: `--spec-type` parsing/help text.
- `src/llama-context.h` / `src/llama-context.cpp`: core seam for a single-context native-MTP design (`decode`, `process_ubatch`, `graph_reserve`, `graph_params`, `graph_max_nodes`, `output_reorder`, `output_resolve_row`).
- `src/llama-graph.h` / `src/llama-graph.cpp`: graph inputs/results, reuse checks, and static-topology patterns worth copying for native MTP.
- `src/models/qwen35.cpp`: best place for Qwen 3.5-specific native-MTP graph/runtime logic.
- `src/models.h`: declarations for any new Qwen 3.5 MTP builder/helper.
- `src/llama-model.cpp` / `src/llama-model.h`: model/hparam/tensor plumbing for loading Qwen 3.5 native-MTP metadata/tensors.
- `src/llama-arch.cpp` / `src/llama-arch.h`: verify QWEN35/QWEN35MOE tensor-name coverage for `NEXTN_*`.
- `gguf-py/gguf/constants.py`, `gguf-py/gguf/tensor_mapping.py`, `convert_hf_to_gguf.py`: converter and GGUF mapping path for Qwen 3.5 MTP tensors.
- `tools/server/server-context.cpp`: real stress surface for `-np 1` / `-np 2`, batching, rollback, and slot state.
- `tools/server/tests/unit/test_speculative.py`: regression scaffolding for speculative/server behavior.
- `scripts/validate_mtp_cuda.py`: preserved harness for exact-output and tok/s validation.

## 3) Key Findings

- The current speculative abstractions in the packed snapshot are pre-decode, external-draft oriented. That makes them a poor fit for native MTP, which should be driven by verifier state already inside the main context, not by a second runtime.
- Upstream already has generic NextN/MTP scaffolding: `llama_hparams` includes `nextn_predict_layers`, so the clean path is to finish Qwen 3.5 conversion/loading/runtime on top of existing generic plumbing instead of inventing a private tensor/runtime format. Official Qwen 3.5 27B and 122B-A10B configs both expose native MTP with `mtp_num_hidden_layers: 1` and `mtp_use_dedicated_embeddings: false`, which is a strong fit for a simple continuation-draft first implementation.
- `llama_context` graph reuse and output ordering are sensitive. The packed snapshot already points at `llm_graph_result::can_reuse`, `build_inp_out_ids`, `build_sampling`, and `output_ids`/`output_reorder` as sharp edges. Native MTP must respect those exact seams or it will reintroduce the same correctness/performance failures seen in the sidecar approach.
- Extra public-upstream context checked locally from `master`: `src/llama-context.cpp` around `process_ubatch` / `decode` / `graph_reserve` / `graph_params`, `src/llama-graph.cpp` around `llm_graph_result::can_reuse` / `build_inp_out_ids` / `build_sampling`, `tools/server/server-context.cpp` around speculative batching/accept paths, and the current converter / GGUF files for Qwen 3.5 and generic NextN plumbing.

## 4) Approach

- Recommended architecture: single-context native MTP as a first-class capability inside `llama_context`, with model-specific proposal graphs in Qwen 3.5 model/runtime code, and existing exact acceptance policy reused outside it.
- Concretely: decode the current verifier token normally; sample/accept the first next token normally; run a small native-MTP graph from the same context using the just-computed seed hidden state plus the accepted token id to draft only the continuation token(s); then verify `[accepted_first_token] + drafted_continuation` with the ordinary main-model decode and exact sampler-accept logic.
- This is the cleanest tradeoff:
  - correctness: the first committed token is still baseline verifier output;
  - performance: no second `llama_context`, no hidden-state handoff between runtimes, no duplicated KV/runtime state;
  - maintainability: backend-agnostic ggml/model code, not a CUDA-only orchestration hack;
  - extensibility: native MTP becomes a reusable core facility for any later model with `NEXTN_*`-style heads.

## 5) Deliverable For Codex CLI

### Target architecture

- Native MTP is a capability of one verifier `llama_context`.
- Native MTP is not another `common_speculative_state`.
- Qwen 3.5 MTP compute lives in model/core graph code, not in `common/` and not in backend-specific server glue.
- The execution model should be:
  1. Normal verifier decode of the current token(s).
  2. Normal sampling of the first next token from verifier logits.
  3. Native MTP proposal of continuation token(s) from the same context, using:
     - the verifier seed hidden state for the just-processed token, and
     - the accepted first token id.
  4. Verifier decode of `[accepted_first_token] + drafted_continuation`.
  5. Exact accept/reject using the existing sampler-agreement logic and rollback of any unaccepted suffix.
- This is the key shape to preserve exactness: the first token stays baseline-verifier exact, and only the continuation is speculative.

### What not to do

- Do not create a second `llama_context` for native MTP.
- Do not pass hidden state between verifier and a sidecar runtime.
- Do not put backend-specific orchestration in `common/` or `tools/server/`.
- Do not force native MTP into the current pre-decode `common_speculative_draft()` abstraction.
- Do not silently reinterpret Qwen 3.5 main `n_layer` semantics to include MTP-only blocks unless verifier-layer count and MTP-layer count are split cleanly.
- Do not tie native MTP to the public embeddings feature flag.
- Do not bypass `output_ids`, `output_reorder()`, or row-resolution helpers.
- Do not recurse a 1-layer Qwen 3.5 MTP head to fabricate deeper drafts in v1.

### Phased implementation plan

1. Baseline freeze and checkpoint `mtp-00-baseline`.
   - Build clean upstream CPU and CUDA.
   - Keep `scripts/validate_mtp_cuda.py` as the preserved starting point.
   - Record baseline non-MTP server outputs/logs on the target Qwen 3.5 GGUF for later regression comparison.
   - No MTP runtime changes yet.
2. Qwen 3.5 converter / GGUF plumbing, still runtime-disabled.
   - In `convert_hf_to_gguf.py`:
     - stop dropping Qwen 3.5 `mtp*` tensors;
     - map HF `mtp_num_hidden_layers` -> GGUF `nextn_predict_layers`;
     - honor `mtp_use_dedicated_embeddings`;
     - normalize Qwen 3.5 MTP tensor names onto generic `NEXTN_*`.
   - In `gguf-py/gguf/constants.py`:
     - add `NEXTN_*` tensors to `MODEL_ARCH.QWEN35` and `MODEL_ARCH.QWEN35MOE`.
   - In `gguf-py/gguf/tensor_mapping.py`:
     - add Qwen 3.5 MTP aliases to generic `NEXTN_*`.
   - In `src/llama-arch.cpp`:
     - verify QWEN35 / QWEN35MOE tensor-name lists include the needed `NEXTN_*` tensors; add them if missing.
   - In `src/llama-model.cpp`:
     - load `nextn_predict_layers` for QWEN35 / QWEN35MOE;
     - load and attach Qwen 3.5 MTP tensors to native model structures.
   - Checkpoint: `mtp-01-qwen35-gguf-load-only`.
3. Create a core native-MTP facility in `src/`, disabled by default.
   - Add `src/llama-mtp.h` / `src/llama-mtp.cpp` (or equivalent focused core files).
   - Define a small runtime descriptor:
     - native-MTP supported or not;
     - max native draft depth;
     - dedicated/shared-embedding behavior;
     - proposal scratch/state per context/sequence.
   - Extend `llama_context` with native-MTP state and entry points, but do not change public behavior yet.
   - Checkpoint: `mtp-02-core-scaffold-disabled`.
4. Add a dedicated native-MTP graph type and graph I/O.
   - In `src/llama-graph.h` / `src/llama-graph.cpp`:
     - add `LLM_GRAPH_TYPE_MTP` (or an equivalently isolated native-MTP graph path);
     - add graph inputs for:
       - seed hidden rows,
       - accepted token ids;
     - add result fields for proposal logits / proposal ids.
   - Extend reuse logic so proposal-graph topology is explicit in `can_reuse()`.
   - Keep graph shape static w.r.t. max native draft depth.
   - Checkpoint: `mtp-03-mtp-graph-plumbing`.
5. Implement Qwen 3.5 native-MTP proposal builders.
   - In `src/models.h`:
     - declare new builder/helper(s), e.g. `llm_build_qwen35_mtp` and the MoE variant.
   - In `src/models/qwen35.cpp` and the file that owns `llm_build_qwen35moe`:
     - implement model-specific native-MTP graph code using Qwen 3.5 MTP weights;
     - input = verifier seed hidden row(s) + accepted token ids;
     - output = continuation-draft logits or top-1 ids;
     - use only ordinary ggml ops and backend scheduler paths.
   - V1 cap: native draft depth = model-native depth. For the official configs checked, that is 1 continuation layer.
   - Checkpoint: `mtp-04-qwen35-native-proposal`.
6. Integrate native MTP into `llama_context` execution.
   - In `src/llama-context.h` / `src/llama-context.cpp`:
     - add private native-MTP proposal APIs;
     - store MTP seed rows using `output_resolve_row()` semantics;
     - keep MTP seed handling private and independent from public embeddings.
   - Temporary bring-up is allowed to use host-buffered seed rows.
   - Final target is a persistent backend-managed seed buffer with no host round-trip.
   - Checkpoint: `mtp-05-context-native-mtp-cpu`.
7. Add the public `mtp` mode surface, without putting runtime logic into `common/speculative.cpp`.
   - In `common/common.h`:
     - add `COMMON_SPECULATIVE_TYPE_MTP`.
   - In `common/arg.cpp`:
     - add `mtp` to `--spec-type`.
   - In `common/speculative.h` / `common/speculative.cpp`:
     - update enum/string plumbing only;
     - do not add an MTP sidecar implementation there.
   - On non-MTP models:
     - fail fast at startup with a clear error instead of silently disabling.
   - Checkpoint: `mtp-06-public-flag-surface`.
8. Server integration for `-np 1` first.
   - In `tools/server/server-context.cpp`:
     - keep old draft-model/ngram speculative paths unchanged;
     - add a new native-MTP branch after normal verifier decode.
   - Native-MTP server loop:
     - decode current sampled token(s) normally;
     - sample the first next token normally;
     - ask `llama_context` for continuation draft(s);
     - build verifier batch `[accepted_first_token] + drafted_continuation`;
     - reuse existing exact accept helper on that verification batch;
     - rollback only the unaccepted suffix.
   - Add per-slot pending state as needed for:
     - accepted-first-token staging,
     - continuation draft storage,
     - verification-batch row mapping.
   - Checkpoint: `mtp-07-server-np1-exact`.
9. Harden `-np 2` and shared-slot behavior as a first-class gate.
   - Batch native-MTP proposal work across compatible slots.
   - Keep slot-local proposal buffers isolated.
   - Reuse existing batching compatibility rules.
   - If Qwen 3.5 hybrid/recurrent rollback is not exactly reversible with `llama_memory_seq_rm()` alone, add a temporary narrow fallback using per-sequence state save/restore rather than inventing a second runtime.
   - Checkpoint: `mtp-08-server-np2-exact`.
10. Performance cleanup after exactness is stable.
   - Remove host seed-buffer round-trips.
   - Reserve native-MTP graphs explicitly and tune reuse.
   - Keep proposal graph topology fixed for reuse.
   - Only after this phase should the branch be treated as upstream-ready.
   - Checkpoint: `mtp-09-upstream-ready`.

### Temporary compatibility steps

- LM-only / no-mmproj native MTP in v1.
- Dense Qwen 3.5 first if needed; MoE follows through the same interface.
- Cap `--draft-max` to the model's native MTP depth.
- Allow a narrow per-sequence state snapshot/restore fallback only if exact rollback requires it.
- Allow a host-buffered seed path only for bring-up, not for final GPU-ready design.

### Final-state cleanup

- Remove any host-buffered seed path.
- Remove any temporary snapshot fallback once exact rollback is proven.
- Keep all native-MTP runtime logic in `src/`, not `common/`.
- Share the same core native-MTP helper with any later CLI/completion integration rather than duplicating server logic.

### Files to edit/create + change summary

- `gguf-py/gguf/constants.py`
  - Add `NEXTN_*` tensors to `MODEL_ARCH.QWEN35` / `MODEL_ARCH.QWEN35MOE`.
- `gguf-py/gguf/tensor_mapping.py`
  - Add/adjust Qwen 3.5 MTP name mappings to generic `NEXTN_*`.
- `convert_hf_to_gguf.py`
  - Stop dropping `mtp*`; map `mtp_num_hidden_layers`; honor dedicated/shared embedding behavior; normalize Qwen 3.5 MTP names.
- `src/llama-arch.cpp`
  - Verify or add `NEXTN_*` coverage for QWEN35 / QWEN35MOE tensor-name lists.
- `src/llama-model.h`
  - Add capability hooks/descriptor access if needed.
- `src/llama-model.cpp`
  - Change QWEN35 / QWEN35MOE hparam-loading and tensor-loading cases.
- `src/models.h`
  - Declare native-MTP builder(s) for Qwen 3.5.
- `src/models/qwen35.cpp`
  - Implement Qwen 3.5 native-MTP graph builder/helper(s).
- `src/models/qwen35moe.cpp` or the upstream file that implements `llm_build_qwen35moe`
  - Implement the MoE native-MTP builder/helper(s).
- `src/llama-mtp.h` (new)
  - Native-MTP core declarations.
- `src/llama-mtp.cpp` (new)
  - Native-MTP core helpers.
- `src/llama-graph.h`
  - Add native-MTP graph type, inputs, outputs.
- `src/llama-graph.cpp`
  - Implement native-MTP graph inputs and reuse/static-topology logic.
- `src/llama-context.h`
  - Add native-MTP context state/API.
- `src/llama-context.cpp`
  - Change `decode`, `process_ubatch`, `graph_reserve`, `graph_params`, `graph_max_nodes`, and seed/output-row handling.
- `common/common.h`
  - Add `COMMON_SPECULATIVE_TYPE_MTP`.
- `common/arg.cpp`
  - Add `mtp` to `--spec-type`.
- `common/speculative.h`
  - Enum/string plumbing only.
- `common/speculative.cpp`
  - Enum/string plumbing only; no sidecar MTP state.
- `tools/server/server-context.cpp`
  - Add native-MTP slot flow, verification staging, `-np 1` / `-np 2` logic.
- `tools/server/tests/unit/test_speculative.py` or `tools/server/tests/unit/test_mtp.py` (new)
  - Add parser/startup/non-MTP rejection smoke tests.
- `docs/speculative.md`
  - Document native `mtp`, its constraints, and the exactness guarantee.
- `scripts/validate_mtp_cuda.py`
  - Preserve as the starting harness; optionally add non-breaking flags like `--modes baseline,mtp` / threshold checks later.
- `CMakeLists.txt` or the source-list file
  - Only if new `src/llama-mtp.cpp` is not auto-discovered.

Likely existing symbols to change:

- `llama_context::decode`
- `llama_context::process_ubatch`
- `llama_context::graph_reserve`
- `llama_context::graph_params`
- `llama_context::graph_max_nodes`
- `llama_context::output_resolve_row`
- `llm_graph_result::can_reuse`
- `llm_graph_context::build_inp_out_ids`
- `llm_graph_context::build_sampling`
- `llm_build_qwen35::llm_build_qwen35`
- `common_speculative_type_to_str`
- `common_speculative_type_from_name`
- speculative slot handling in `tools/server/server-context.cpp`

### Commands to run

Builds:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j
```

Existing server/speculative regression tests:

```bash
pytest tools/server/tests/unit/test_speculative.py -q
```

CPU exactness + `-np 1` / `-np 2`:

```bash
python3 scripts/validate_mtp_cuda.py \
  --binary build/bin/llama-server \
  --model /path/to/qwen35-mtp.gguf \
  --ngl 0 \
  --flash-attn off \
  --ctx-size 4096 \
  --batch-size 64 \
  --ubatch-size 64 \
  --threads 8 \
  --threads-batch 8
```

CUDA exactness + `-np 1` / `-np 2`:

```bash
python3 scripts/validate_mtp_cuda.py \
  --binary build-cuda/bin/llama-server \
  --model /path/to/qwen35-mtp.gguf \
  --ngl all \
  --flash-attn on \
  --ctx-size 4096 \
  --batch-size 64 \
  --ubatch-size 64 \
  --threads 8 \
  --threads-batch 8
```

Non-MTP model fast-fail check:

```bash
build/bin/llama-server \
  -m /path/to/non-mtp-model.gguf \
  --spec-type mtp \
  --no-webui \
  --no-warmup
```

### Validation checklist / pass-fail gates

Hard gates:

- Qwen 3.5 MTP GGUF loads successfully.
- Non-MTP models still load and decode normally when MTP is not requested.
- `--spec-type mtp` on a non-MTP model fails fast with a clear error.
- CPU harness exits 0 and prints exact greedy equality for:
  - baseline vs native MTP,
  - `-np 1`,
  - `-np 2`.
- CUDA harness exits 0 and prints exact greedy equality for:
  - baseline vs native MTP,
  - `-np 1`,
  - `-np 2`.
- Existing speculative/server tests stay green.
- Baseline greedy output with MTP disabled matches the pre-MTP checkpoint output on the same model/prompt/seed.

Promotion gates before upstream PR:

- Reference CUDA `np=1`: native MTP tok/s must beat baseline tok/s.
- Reference CUDA `np=2`: native MTP mean per-request tok/s must not regress beyond a small noise budget; a practical gate is "no worse than baseline by more than 10%".
- CPU native MTP tok/s must be recorded and compared, but CPU speedup is not a blocker for the first functional merge as long as:
  - CPU exactness passes, and
  - the disabled path remains unchanged.

### Rollback / checkpoint boundaries

- `mtp-00-baseline`: clean upstream build + baseline logs.
- `mtp-01-qwen35-gguf-load-only`: converter/loading only; no runtime behavior change.
- `mtp-02-core-scaffold-disabled`: native-MTP core exists but is dark.
- `mtp-03-mtp-graph-plumbing`: native-MTP graph type exists, still not user-visible.
- `mtp-04-qwen35-native-proposal`: direct proposal builder works.
- `mtp-05-context-native-mtp-cpu`: single-context CPU native-MTP path works.
- `mtp-06-public-flag-surface`: `--spec-type mtp` exists.
- `mtp-07-server-np1-exact`: server exactness for `-np 1`.
- `mtp-08-server-np2-exact`: shared-slot exactness for `-np 2`.
- `mtp-09-upstream-ready`: performance cleanup complete.

Rollback rule:

- If any checkpoint breaks exact greedy equality or disabled-path baseline behavior, stop and revert to the previous checkpoint instead of stacking more changes on top.

### Risks / gotchas

- Qwen 3.5 hybrid/recurrent memory makes rollback correctness more delicate than dense-only transformer models.
- Output-row ordering bugs are easy to reintroduce if native MTP reads seed rows without going through the existing output-id machinery.
- Host round-trips for seed hidden states may be acceptable for bring-up but will likely destroy CUDA performance if left in place.
- Overloading `n_layer` semantics with MTP-only blocks will create long-term maintenance pain in Qwen 3.5 graph loops; normalize or separate that state early.
- `-np 2` is the real stress case; do not treat it as cleanup work after `-np 1`.
