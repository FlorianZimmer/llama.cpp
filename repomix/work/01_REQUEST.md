USER_GOAL:
Review the current dense-only native-MTP branch in this private llama.cpp mirror and produce a concrete implementation plan for recovering and broadening `np=1` single-user speedups on Qwen 3.5 dense models.

DELIVERABLE_TYPE: PLAN

USER_REQUEST:
Please review this private-mirror native-MTP implementation and give a practical next-step plan for dense Qwen 3.5 only.

The current branch has just been narrowed for V1 prep:

- `qwen35` native MTP is still in scope
- `qwen35moe` native MTP support has been removed from the live branch
- `Qwen3.5-35B-A3B` / MoE is no longer part of the live V1 code or benchmark scope
- the branch now focuses only on:
  - `Qwen3.5-9B q8_0`
  - `Qwen3.5-9B UD-Q4_K_XL`
  - `Qwen3.5-27B UD-Q4_K_XL`

What I need from you:

1. Review the current dense-only state and propose the best way forward to achieve a real `np=1` speedup on Qwen 3.5 dense models, ideally across the checked quants rather than only `9B q8_0`.
2. Make the plan benchmark-gated after every step against:
   - greedy baseline
   - the immediately previous native-MTP result
3. Be explicit about what we already tried that failed, regressed, or did not survive repeated end-to-end tok/s testing, so the plan does not waste cycles.
4. Tell us whether the remaining ceiling for the current one-token native-MTP design looks:
   - still attackable with one more dense-only runtime branch
   - or mostly structural unless we do deeper runtime-state work
5. If you recommend deeper work, separate:
   - what is still realistic for an upstream-friendly dense V1
   - what belongs in a larger follow-up branch

Important local branch facts:

- The active dense target is `Qwen3.5-9B q8_0`.
- Current branch result on the live dense-only state:
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

Critical history:

- We did have an earlier faster dense `9B q8_0` state under the same newer harness:
  - file: `/tmp/native-mtp-bench-20260410/qwen35-9b-q8_0.json`
  - `primary np=1`: `1.082x`
  - `good np=1`: `1.030x`
- A broad post-replay guard regressed that path badly:
  - file: `/tmp/native-mtp-bench-20260410-post-replay-guard/qwen35-9b-q8_0.json`
  - `primary np=1`: `0.999x`
  - `good np=1`: `0.938x`
- The current dense-only branch recovered the earlier `9B q8_0` win by removing that broad guard from dense `qwen35`.

Important measurement result:

- The visibility pass already showed speculative accept rows are almost entirely on the intended fast path with logits suppressed.
- Representative coverage:
  - `9B q8_0 primary np=1`: `15/15` pure-fast-path, `15/15` logits-suppressed
  - `9B UD-Q4 good np=2`: `180/186` pure-fast-path, `180/186` logits-suppressed
  - `27B UD-Q4 bad np=2`: `180/186` pure-fast-path, `180/186` logits-suppressed

That means the next plan should not assume “find another verifier fast path in server-context.cpp” is the main missing win.

What I want back:

1. A direct reading of whether we already exhausted the easy server-local wins.
2. A small-step plan for the next dense-only branch, benchmark-gated after every step.
3. A separate list of larger follow-up ideas only if they are truly outside a reasonable V1.
4. A specific call on whether the current one-token native-MTP design is likely capable of:
   - consistent `>= 5%` `np=1` speedups on dense Qwen 3.5
   - or only narrow wins like `9B q8_0`

If you use public references, be surgical. Focus on implementations or documentation that directly help this branch:

- public upstream `ggml-org/llama.cpp` only where needed
- public vLLM / SGLang / TensorRT-LLM only where they materially illuminate replay, verifier, state, or dense runtime economics for MTP-like paths

Please keep the plan practical and codebase-aware. Do not turn this into a generic research survey.
