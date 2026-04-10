USER_GOAL:
Review the current private native-MTP state in this llama.cpp mirror and answer two scope-defining questions for the next cycle:

1. Should we spend another branch on deeper hybrid replay/guard economics for Qwen 3.5 native MTP, or is that too large/risky for this first upstream-oriented implementation?
2. Should `Qwen3.5-35B-A3B` / `qwen35moe` stay in the native-MTP scope at all, or should it be removed/deferred because the expected upside is too small relative to the complexity and correctness cost?

DELIVERABLE_TYPE: RESULT

USER_REQUEST:
Please review this private-mirror native-MTP implementation and give a concrete recommendation about branch scope and next actions.

The main questions are:

1. We already know `Qwen3.5-9B q8_0` used to be clearly faster in an earlier branch state. Did the later correctness work regress that path in a fundamental way, and if so, is it worth targeting deeper hybrid replay/guard economics next?
2. How do other public MTP-capable inference stacks handle the kinds of issues we ran into here:
   - verifier fast-path / logits suppression
   - replay / rollback after rejected drafts
   - hybrid or recurrent state restoration
   - dense vs MoE runtime economics
   - exactness expectations at low draft depth
3. Is `Qwen3.5-35B-A3B` / `qwen35moe` worth keeping in scope for the first upstreamable native-MTP series, or should it be explicitly deferred or removed for now?

Please answer using the included local files as the source of truth for this private branch. For broader comparison, you may use public sources surgically.

Important: do not broaden into a generic literature survey. Focus on public implementations that are directly useful for this decision. Examples of useful targets if relevant:

- public upstream `ggml-org/llama.cpp` only where needed to compare with this branch
- public inference stacks / engines that concretely support MTP or close speculative decoding variants for Qwen-class models, especially where replay/rollback, verifier batching, or MoE support are documented or visible in code

What I want back:

1. A direct answer to whether the next step should be:
   - a deeper hybrid replay/guard branch
   - a narrower dense-only cleanup branch
   - or a stop/defer decision because the remaining ceiling is likely structural for the current one-token native-MTP design
2. A direct answer to whether `qwen35moe` should stay in scope for the first upstreamable series.
3. A short comparison against other public MTP-capable stacks:
   - how they handle replay / verifier / state problems
   - what they do differently from this branch if anything materially relevant
   - whether any of those ideas look realistically portable here
4. A pragmatic recommended path for this private mirror in priority order, with explicit “do now”, “separate later branch”, and “drop/defer” buckets.

Critical benchmark context:

- This branch already had an earlier faster `Qwen3.5-9B q8_0` state under the current harness, before the permanent replay guard:
  - from `/tmp/native-mtp-bench-20260410/qwen35-9b-q8_0.json`
  - `primary np=1`: `150.83 -> 163.23 tok/s` (`1.082x`)
  - `good np=1`: `148.88 -> 153.38 tok/s` (`1.030x`)
  - `primary np=2`: `128.08 -> 138.55 tok/s` (`1.082x`)
  - `good np=2`: `131.40 -> 145.13 tok/s` (`1.104x`)
- Current branch state after the permanent hybrid/recurrent post-replay guard:
  - from `/tmp/native-mtp-bench-20260410-post-replay-guard/qwen35-9b-q8_0.json`
  - `primary np=1`: `150.53 -> 150.31 tok/s` (`0.999x`)
  - `good np=1`: `148.62 -> 139.40 tok/s` (`0.938x`)
  - `primary np=2`: `127.85 -> 133.26 tok/s` (`1.042x`)
  - `good np=2`: `132.66 -> 137.12 tok/s` (`1.034x`)
  - `bad np=1`: `148.82 -> 125.10 tok/s` (`0.841x`)
  - `bad np=2`: `132.99 -> 115.83 tok/s` (`0.871x`)

Current branch-wide reading from the full 2026-04-10 matrix:

- no checked model or quant is net-positive on `np=1`
- only `Qwen3.5-9B q8_0` remains speed-positive at all, and only on the easier `np=2` cases
- `Qwen3.5-9B UD-Q4_K_XL` is slower everywhere
- `Qwen3.5-27B UD-Q4_K_XL` is slower everywhere
- `Qwen3.5-35B-A3B` is slower everywhere
- `Qwen3.5-35B-A3B Q4_K_M` is now `np=1` exact again after the replay guard fix, but still far from speed-positive

Very important local finding from the new visibility pass:

- speculative accept rows are already almost entirely pure fast-path verifier rows with logits suppressed
- representative coverage from `/tmp/native-mtp-step-01`:
  - `9B q8_0 primary np=1`: `15/15` pure-fast-path, `15/15` logits-suppressed
  - `9B UD-Q4 good np=2`: `180/186` pure-fast-path, `180/186` logits-suppressed
  - `27B UD-Q4 bad np=2`: `180/186` pure-fast-path, `180/186` logits-suppressed
- this means the earlier “maybe split pure verifier rows out of mixed chunks” idea is now probably not the main win

Another very important local finding:

- in this branch, `Qwen3.5` dense is currently classified as `hybrid` in libllama too, not just `Qwen3.5-MoE`
- so the current one-step post-replay guard is already the live policy on the checked 9B and 27B targets, not just on A3B
- a trial “dense cooldown” branch produced no distinct `cooldown_hits`; it collapsed into the same guard behavior and was dropped

MoE-specific reality check:

- `Qwen3.5-35B-A3B` native MTP is now functionally/correctness-clean on the checked `np=1` cases
- but it is still materially slower than baseline on every checked quant
- current local belief is that this may simply not be a good speed target even with a cleaner implementation, because the model is already a fast active-parameter MoE path and the current native-MTP depth is only one token

Please do not just say “more profiling needed”. I want a specific call:

- continue deeper on hybrid replay/guard economics now
- or stop here and narrow scope for the first upstreamable series

CONSTRAINTS:

- preserve the already-validated `np=1` exactness contract on the checked dense and A3B cases
- prefer upstream-friendly conclusions over heroic local-only hacks
- separate:
  - “worth doing in this private mirror next”
  - “worth exploring later in a dedicated branch”
  - “not worth keeping in scope for v1”
- if you use public upstream or other public projects, fetch them surgically and explain exactly why they matter
