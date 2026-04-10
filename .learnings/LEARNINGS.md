## [LRN-20260409-001] best_practice

**Logged**: 2026-04-09T20:05:00Z
**Priority**: high
**Status**: pending
**Area**: backend

### Summary
Hybrid/recurrent native-MTP verification batches cannot rely on `llama_memory_seq_rm()` alone to roll back a rejected draft suffix.

### Details
While validating native MTP on Qwen 3.5, speculative verifier batches that decoded `[accepted_token] + draft_suffix` hit M-RoPE position failures on the next step when a drafted suffix was rejected. The root cause was `llama_memory_recurrent::seq_rm()` rejecting partial tail erasure for the final recurrent state, which left the rejected suffix in live memory. The exactness-safe fallback is to snapshot per-sequence state before the speculative verifier batch, restore it on rejection, and replay only the accepted prefix tokens for that sequence.

### Suggested Action
Keep the per-sequence state save/restore fallback in place for hybrid/recurrent native-MTP paths until an exact reversible rollback path exists in the memory layer.

### Metadata
- Source: error
- Related Files: tools/server/server-context.cpp, src/llama-context.cpp
- Tags: mtp, rollback, recurrent, hybrid, exactness

---

## [LRN-20260409-002] best_practice

**Logged**: 2026-04-09T22:25:00Z
**Priority**: high
**Status**: pending
**Area**: backend

### Summary
Native-MTP rollback on hybrid/recurrent models should keep backup state on the recurrent backend instead of serializing `PARTIAL_ONLY` state to host on every draft step.

### Details
In-process timing on Qwen 3.5 native MTP showed that the dominant overhead was the per-step recurrent snapshot, not MTP draft generation. For a 48-token run at `-np 2`, the old path spent about 93-95 ms per request in `llama_state_seq_get_data_ext(..., LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)` alone, versus about 13-16 ms in draft generation and about 16 ms in replay. Replacing that host snapshot with a backend-local recurrent backup seq id collapsed snapshot/restore time to effectively zero while preserving the exact rollback flow for the short validated case.

### Suggested Action
Prefer recurrent-only backup/copy paths for native-MTP rollback on hybrid/recurrent models, and reserve host-side state serialization as the fallback when no backend-local backup is available.

### Metadata
- Source: conversation
- Related Files: src/llama-model.cpp, src/llama-context.cpp, tools/server/server-context.cpp
- Tags: mtp, profiling, performance, recurrent, hybrid
- See Also: LRN-20260409-001, ERR-20260409-002

---
