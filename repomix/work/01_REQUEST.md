USER_GOAL:
Create a detailed implementation plan for the remaining upstream-friendly native-MTP performance work in this private llama.cpp mirror.

DELIVERABLE_TYPE: PLAN

USER_REQUEST:
Please create a detailed step-by-step implementation plan for the three remaining native-MTP optimization areas in this branch:

1. adaptive native-MTP backoff on replay-heavy workloads
2. replay-path reduction / cheaper rejection handling
3. small server hot-path cleanup

This should be one coherent plan that tries to finish all three in one go, but it must stay pragmatic:
- if one strategy does not pan out, set it aside and continue to the next
- only treat something as a blocker if it truly prevents progress on the others

The plan must require benchmarking after every landed step, always against:
- baseline greedy decode
- the immediately previous native-MTP implementation

Do not count an optimization as successful unless it shows a real end-to-end speedup on the validated exact cases, not just lower internal phase timings.

Please use the provided repo slice first. If you need more context from the public llama.cpp repo, fetch only a few additional relevant lines/files surgically. This is a very large codebase, so do not expand scope casually.

What I want from you:
- a prioritized phased implementation plan
- estimated payoff and scope for each step
- exact files / code areas to touch
- validation checkpoints after each step
- fallback / rollback rules if a strategy is not paying off
- explicit guidance for keeping the work upstream-friendly, simple, maintainable, and reusable for future native-MTP models rather than hard-coding Qwen 3.5 assumptions

CONSTRAINTS (optional):
- Preserve current exactness on validated cases:
  - CUDA Berlin `np=1/2`
  - CUDA Moon `np=1/2`
- The known Rust `np=2` stress case is still allowed to diverge because it reflects the documented hybrid/recurrent `np>1` exactness limitation.
- Avoid invasive scheduler/backend API refactors unless absolutely necessary.
- Prefer policy/runtime/server changes before backend/kernel surgery.
- Keep instrumentation/debug aids lightweight and removable.
- Be explicit about what should be measured before and after each step.
