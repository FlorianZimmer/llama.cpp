CONTEXT_NOTES:

Branch context:

- Current branch: `feat/native-mtp-qwen35-dense-speedup`
- Public diff base used locally: `d6f3030047f85a98b009189e76f441fe818ea44d`
- This is a private mirror. Included local docs and files are the source of truth for branch state.

What this branch currently includes relative to public upstream:

- native-MTP GGUF conversion / metadata / loader plumbing
- native-MTP runtime state in `src/llama-mtp.*`
- native-MTP seed transport and graph inputs
- server integration for `--spec-type mtp`
- direct greedy verifier-token access plus raw-logit suppression
- dense Qwen 3.5 graph-builder support in `src/models/qwen35.cpp`
- CUDA validator / benchmark harness with per-step timing and acceptance parsing

What was just removed from the live V1 branch:

- `qwen35moe` native-MTP capability in `src/llama-mtp.cpp`
- branch-local native-MTP graph-builder hooks in `src/models/qwen35moe.cpp`
- MoE-specific replay-guard scope from the live V1 path
- A3B/MoE quant override files under `scripts/mtp_quant_overrides/`
- MoE/A3B as live docs / benchmark scope

Current dense-only runtime facts:

- runtime still drafts only `1` continuation token per step
- `np=1` exactness is the hard contract
- `np>1` is stability-only on the current hybrid path
- current branch still has:
  - greedy verifier accept fast path
  - optional raw-logit suppression for token-only accept batches
  - backend-resident seed transport
  - per-step runtime / acceptance profiling

Current dense benchmark state:

- `Qwen3.5-9B q8_0`
  - `primary np=1`: `150.94 -> 163.10 tok/s` (`1.081x`)
  - `good np=1`: `148.90 -> 153.65 tok/s` (`1.032x`)
  - `bad np=1`: `148.89 -> 147.56 tok/s` (`0.991x`)
- `Qwen3.5-9B UD-Q4_K_XL`
  - `primary np=1`: `175.39 -> 167.94 tok/s` (`0.957x`)
  - `good np=1`: `196.31 -> 182.85 tok/s` (`0.931x`)
  - `primary np=2`: `156.34 -> 157.28 tok/s` (`1.006x`)
- `Qwen3.5-27B UD-Q4_K_XL`
  - `primary np=1`: `72.26 -> 59.71 tok/s` (`0.826x`)
  - `good np=1`: `70.96 -> 68.41 tok/s` (`0.964x`)
  - `bad np=2`: `59.17 -> 32.92 tok/s` (`0.556x`)

Dense-only post-cleanup smoke rerun:

- `/tmp/native-mtp-v1-dense-only-smoke/qwen35-9b-q8_0-primary.json`
  - `primary np=1`: `148.46 -> 163.32 tok/s` (`1.100x`)
- `/tmp/native-mtp-v1-dense-only-smoke/qwen35-9b-ud-q4-primary.json`
  - `primary np=1`: `173.77 -> 163.41 tok/s` (`0.940x`)
- `/tmp/native-mtp-v1-dense-only-smoke/qwen35-27b-ud-q4-primary.json`
  - `primary np=1`: `72.09 -> 59.28 tok/s` (`0.822x`)

Important branch history that should affect the next plan:

1. Fast-path accept work already landed.
   - `5cfd26302` `native MTP: fast-path greedy verifier accept`
   - `65f3e244a` `native MTP: skip raw logits on token-only accept`
   - these produced small real wins, mainly on `9B q8_0`

2. Backend seed transport already landed.
   - `cbc7e258f` `native MTP: keep seed transport on backend`
   - this cleaned up architecture but did not create broad speedups by itself

3. Scratch reuse already landed.
   - `43b19c4b4` `native MTP: reuse host scratch state`

4. Dedicated draft-side caching / scheduler ideas were tried and dropped.
   - local reading from docs: the draft bucket barely moved and end-to-end gains did not survive repeated benchmarking

5. Broad hybrid replay guard was a real regression.
   - `069224a37` `native mtp: guard replayed hybrid steps and refresh benchmarks`
   - this fixed the A3B exactness issue, but it also regressed dense `9B q8_0` because dense `qwen35` is also classified as hybrid in libllama

6. Step visibility already ruled out an easy remaining server-local win.
   - `60d211f78` `native mtp: add step visibility and narrow v1 scope`
   - representative coverage:
     - `9B q8_0 primary np=1`: `15/15` pure-fast-path, `15/15` logits-suppressed
     - `9B UD-Q4 good np=2`: `180/186`
     - `27B UD-Q4 bad np=2`: `180/186`
   - conclusion:
     - dense losses are not mainly due to missed fast-path coverage in mixed accept batches

7. Narrowing the replay guard back off dense `qwen35` recovered the target win.
   - `3396cf9a5` `native mtp: narrow replay guard for dense qwen35`
   - current live branch is effectively the dense-only form of that decision

What we already tried on the “hybrid-state” question:

- rollback / restore / replay hardening
- replay-logit tracing
- disabling the greedy accept fast path as an isolation test
- one-step post-replay cooldown / plain-step guard
- broad guard on all hybrid Qwen 3.5 models
- narrowed guard to avoid regressing dense `qwen35`

What we did not yet try:

- deeper explicit speculative branch-state storage of the kind public stacks use
- deeper libllama integration of draft + verify + rewind beyond the current restore/replay path
- multi-token native drafting

That distinction matters. We did try the shallow replay/guard experiments already. We did not yet try the larger runtime-state redesign.

MoE / A3B history to keep in mind but not keep in scope:

- A3B originally exposed both a quantization-side weakness and a runtime exactness issue
- balanced quant rescue was necessary but not sufficient
- exactness was restored conservatively with a post-replay plain-step guard
- throughput still stayed materially below baseline on every checked quant
- result: MoE was removed from this V1 prep branch rather than kept as an active target

Current local maintenance bar:

- if dense native MTP cannot produce a consistent `>= 5%` `np=1` single-user win, it is probably not worth carrying as a complex upstream-facing speed feature

Files that matter most for this review:

- `tools/server/server-context.cpp`
- `src/llama-mtp.h`
- `src/llama-mtp.cpp`
- `src/llama-context.cpp`
- `src/llama-graph.h`
- `src/llama-graph.cpp`
- `src/models/qwen35.cpp`
- `include/llama.h`
- `scripts/validate_mtp_cuda.py`
- `scripts/audit_mtp_quantization.py`
- `docs/development/native-mtp-benchmarks.md`
- `docs/development/native-mtp-optimization-plan.md`
- `docs/development/native-mtp-model-prep.md`

If more surrounding context is needed beyond the included files:

- consult public `ggml-org/llama.cpp` upstream surgically
- use `repomix/work/04_UPSTREAM_DIFF_SUMMARY.md` first to see which files in this private branch differ from upstream and therefore must not be treated as upstream-equivalent
