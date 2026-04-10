# Focused Code Excerpts

These excerpts are from large files that are too expensive to include whole. They cover the current native-MTP runtime shape, the server-side speculative / replay path, and the measurement hooks that future optimizations must preserve.

## `tools/server/server-context.cpp`

### Native MTP profiling counters and runtime flags on each slot

```cpp
48	static bool server_native_mtp_profile_enabled() {
49	    static const bool enabled = getenv("LLAMA_SERVER_MTP_PROFILE") != nullptr;
50	    return enabled;
51	}
...
74	    struct native_mtp_profile_data {
75	        int32_t n_draft_calls = 0;
76	        int32_t n_snapshot = 0;
77	        int32_t n_accept = 0;
78	        int32_t n_restore = 0;
79	        int32_t n_replay = 0;
80	
81	        int64_t t_draft_us = 0;
82	        int64_t t_snapshot_us = 0;
83	        int64_t t_accept_us = 0;
84	        int64_t t_restore_us = 0;
85	        int64_t t_replay_us = 0;
86	    };
...
96	    common_speculative * spec = nullptr;
97	    common_speculative_type spec_type = COMMON_SPECULATIVE_TYPE_NONE;
98	    bool native_mtp = false;
99	    int32_t native_mtp_max = 0;
...
197	    std::vector<uint8_t> native_mtp_state;
198	    llama_state_seq_flags native_mtp_state_flags = 0;
199	    llama_seq_id native_mtp_backup_id = -1;
200	    native_mtp_profile_data native_mtp_profile;
```

### Slot-level speculation helpers

```cpp
314	    common_speculative_type runtime_spec_type() const {
315	        if (native_mtp) {
316	            return COMMON_SPECULATIVE_TYPE_MTP;
317	        }
...
334	    bool can_speculate() const {
335	        if (task) {
336	            return task->params.speculative.type != COMMON_SPECULATIVE_TYPE_NONE &&
337	                has_spec_runtime(task->params.speculative.type);
338	        }
339	
340	        return runtime_spec_type() != COMMON_SPECULATIVE_TYPE_NONE;
341	    }
...
390	    int get_n_draft_max() const {
393	        if (!can_speculate()) {
394	            return 0;
395	        }
...
398	        int n_draft_max = task->params.speculative.n_max;
400	        if (uses_native_mtp()) {
401	            n_draft_max = std::min(n_draft_max, native_mtp_max);
402	        }
...
406	        n_draft_max = std::min(n_draft_max, n_ctx - prompt.n_tokens() - 2);
409	            n_draft_max = std::min(n_draft_max, n_remaining - 1);
...
414	        if (n_draft_max < task->params.speculative.n_min) {
415	            ...
416	            n_draft_max = 0;
417	        }
419	        return n_draft_max;
420	    }
```

### Where per-slot native-MTP phase timing is reported today

```cpp
522	        if (server_native_mtp_profile_enabled() && native_mtp && n_draft_total > 0) {
523	            const auto to_ms = [](int64_t us) { return us / 1000.0; };
524	            const int64_t t_native_total_us =
525	                    native_mtp_profile.t_draft_us +
526	                    native_mtp_profile.t_snapshot_us +
527	                    native_mtp_profile.t_accept_us +
528	                    native_mtp_profile.t_restore_us +
529	                    native_mtp_profile.t_replay_us;
...
539	                    to_ms(native_mtp_profile.t_draft_us),    native_mtp_profile.n_draft_calls,
540	                    to_ms(native_mtp_profile.t_snapshot_us), native_mtp_profile.n_snapshot,
541	                    to_ms(native_mtp_profile.t_accept_us),   native_mtp_profile.n_accept,
542	                    to_ms(native_mtp_profile.t_restore_us),  native_mtp_profile.n_restore,
543	                    to_ms(native_mtp_profile.t_replay_us),   native_mtp_profile.n_replay,
544	                    to_ms(t_native_total_us));
545	        }
```

### Batched native-MTP draft generation across generating slots

```cpp
2288	        std::unordered_map<int, llama_tokens> native_mtp_drafts;
...
2304	                const int n_draft_max = slot.get_n_draft_max();
2305	                if (!slot.uses_native_mtp() || n_draft_max <= 0) {
2306	                    continue;
2307	                }
...
2321	                const int32_t n_mtp = llama_native_mtp_draft_batch(
2322	                        ctx,
2323	                        mtp_seq_ids.data(),
2324	                        mtp_tokens.data(),
2325	                        mtp_pos.data(),
2326	                        mtp_slots.size(),
2327	                        mtp_batch_draft.data(),
2328	                        1);
...
2332	                    mtp_slot->native_mtp_profile.n_draft_calls += 1;
2333	                    mtp_slot->native_mtp_profile.t_draft_us += t_mtp_draft_us / (int64_t) mtp_slots.size();
...
2338	                        native_mtp_drafts[mtp_slots[i]->id] = { mtp_batch_draft[i] };
```

### Per-slot draft, snapshot, and draft-batch expansion

```cpp
2363	            const int n_draft_max = slot.get_n_draft_max();
2364	            if (n_draft_max > 0) {
...
2373	                if (slot.uses_native_mtp()) {
2374	                    auto it_draft = native_mtp_drafts.find(slot.id);
2375	                    if (it_draft != native_mtp_drafts.end()) {
2376	                        draft = it_draft->second;
2377	                    } else {
2378	                        draft.resize((size_t) std::min(1, n_draft_max));
...
2381	                        const int32_t n_mtp = llama_native_mtp_draft(
2382	                                slot.ctx,
2383	                                slot.id,
2384	                                slot.sampled,
2385	                                slot.prompt.tokens.pos_next(),
2386	                                draft.data(),
2387	                                draft.size());
...
2407	                const bool use_recurrent_backup = slot.uses_native_mtp_recurrent_backup();
2408	                const llama_state_seq_flags native_mtp_state_flags =
2409	                    slot.uses_native_mtp() && !use_recurrent_backup && llama_model_is_hybrid(llama_get_model(slot.ctx))
2410	                        ? LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY
2411	                        : 0;
2412	
2413	                if (slot.uses_native_mtp() && !draft.empty()) {
2414	                    const int64_t t_snapshot_start = ggml_time_us();
2415	                    const bool saved = use_recurrent_backup
2416	                            ? llama_memory_seq_cp_recr(llama_get_memory(ctx), slot.id, slot.native_mtp_backup_id, -1, -1)
2417	                            : slot.save_native_mtp_state(native_mtp_state_flags);
2418	                    slot.native_mtp_profile.n_snapshot += 1;
2419	                    slot.native_mtp_profile.t_snapshot_us += ggml_time_us() - t_snapshot_start;
2420	
2421	                    if (!saved) {
2422	                        ...
2423	                        draft.clear();
2424	                    }
2425	                }
...
2432	                if (slot.task->params.speculative.n_min > (int) draft.size()) {
2433	                    ...
2434	                } else {
2435	                    slot.n_draft_total += draft.size();
2436	                    for (size_t i = 0; i < draft.size(); i++) {
2437	                        slot.i_batch_dft.push_back(batch.n_tokens);
2438	                        common_batch_add(batch, draft[i], slot.prompt.tokens.pos_next(), { slot.id }, true);
2439	                        slot.prompt.tokens.push_back(draft[i]);
2440	                    }
2441	                    slot.drafted = std::move(draft);
2442	                }
```

### Replay helper used after rejection

```cpp
3042	        struct native_replay_entry {
3043	            server_slot * slot = nullptr;
3044	            size_t n_prompt_base = 0;
3045	        };
...
3059	        auto replay_native_mtp_prefix_batch = [&](const std::vector<native_replay_entry> & replay_slots) {
3060	            size_t n_replay_total = 0;
3061	            size_t n_replay_max = 0;
3062	            for (const auto & replay_slot : replay_slots) {
3063	                ...
3065	                const size_t n_replay = replay_slot.slot->prompt.n_tokens() - replay_slot.n_prompt_base;
3066	                n_replay_total += n_replay;
3067	                n_replay_max = std::max(n_replay_max, n_replay);
3068	            }
...
3074	            llama_batch replay = llama_batch_init((int32_t) replay_slots.size(), 0, 1);
...
3079	            for (size_t step = 0; step < n_replay_max; ++step) {
3080	                common_batch_clear(replay);
...
3085	                    const size_t n_replay = slot.prompt.n_tokens() - replay_slot.n_prompt_base;
3086	                    if (step >= n_replay) {
3087	                        continue;
3088	                    }
...
3091	                    const bool logits = step + 1 == n_replay;
3092	                    common_batch_add(replay, slot.prompt.tokens[replay_slot.n_prompt_base + step], pos, { slot.id }, logits);
...
3110	                if (llama_decode(ctx, replay) != 0) {
3111	                    ok = false;
3112	                    break;
3113	                }
3114	            }
...
3116	            llama_batch_free(replay);
3117	            return ok;
3118	        };
```

### Accept, rollback, and replay decision path

```cpp
3301	            // speculative decoding - main model sample and accept
3302	            std::vector<speculative_accept_result> speculative_results;
...
3314	                if (slot.uses_native_mtp()) {
3315	                    slot.native_mtp_profile.n_accept += 1;
3316	                    slot.native_mtp_profile.t_accept_us += ggml_time_us() - t_accept_start;
3317	                }
...
3326	            bool any_native_replay = false;
3327	            for (auto & spec_result : speculative_results) {
3328	                auto & slot = *spec_result.slot;
3329	                const size_t n_accepted_draft = spec_result.ids.size() - 1;
3330	                spec_result.n_prompt_base = slot.prompt.n_tokens() - spec_result.n_draft - 1;
3331	                spec_result.needs_native_replay = slot.uses_native_mtp() && n_accepted_draft < spec_result.n_draft;
3332	                any_native_replay = any_native_replay || spec_result.needs_native_replay;
...
3346	                slot.n_draft_accepted += spec_result.ids.size() - 1;
...
3353	                // rollback to the state before sampling the draft tokens
3354	                slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - spec_result.n_draft);
3355	
3356	                // add accepted tokens to the prompt
3357	                slot.prompt.tokens.insert({spec_result.ids.begin(), spec_result.ids.end() - 1});
3358	                slot.sampled = spec_result.ids.back(); // last accepted token
3359	            }
...
3361	            std::vector<native_replay_entry> native_replay_slots;
3362	            if (any_native_replay) {
3363	                for (auto & spec_result : speculative_results) {
```

## `src/llama-context.cpp`

### Current native-MTP draft entry point and backend-resident seed path

```cpp
701	int32_t llama_context::mtp_draft_batch(
...
712	    // Finish the verifier decode and seed copy before reusing the scheduler for MTP.
713	    synchronize();
...
720	    if (native_mtp.seed_mode == LLAMA_MTP_SEED_MODE_NONE) {
721	        return 0;
722	    }
...
730	    switch (native_mtp.seed_mode) {
731	        case LLAMA_MTP_SEED_MODE_NONE:
732	            return 0;
733	        case LLAMA_MTP_SEED_MODE_HOST:
...
748	        case LLAMA_MTP_SEED_MODE_BACKEND:
749	            if (!native_mtp.seed_backend.ready()) {
750	                return 0;
751	            }
...
765	                ggml_backend_tensor_copy(
766	                        native_mtp.seed_backend.seed_cache_rows[seq_ids[i]],
767	                        native_mtp.seed_backend.seed_batch_rows[i]);
768	            }
...
781	            mtp_seed_backend = native_mtp.seed_backend.seed_batch_dev;
782	            mtp_seed_generation = native_mtp.seed_backend.generation;
783	            break;
784	    }
...
836	    const auto gparams = graph_params(
837	            gf_res_prev.get(),
838	            ubatch,
839	            nullptr,
840	            LLM_GRAPH_TYPE_MTP,
841	            native_mtp.seed_mode,
842	            mtp_seed,
843	            mtp_seed_backend,
844	            mtp_seed_generation,
845	            tokens,
846	            n_seq);
```

### Current verifier hidden-state capture dispatcher

```cpp
1840	static void capture_mtp_seed_rows(
1841	        ggml_tensor * tensor,
1842	        const std::map<llama_seq_id, uint32_t> & seq_to_row,
1843	        llama_mtp_state & dst,
1844	        ggml_backend_sched_t sched) {
...
1854	    dst.next_seed_epoch();
1855	
1856	    if (!mtp_capture_seed_rows_backend(tensor, seq_to_row, dst, sched, row_size)) {
1857	        // Keep the old host round-trip path for host-backed and single-sequence fallback.
1858	        // On non-host backends with multiple live sequences, dropping back to host can still
1859	        // hit the existing hybrid/recurrent np>1 exactness limitation, so prefer no seed.
1860	        if (tensor->buffer != nullptr &&
1861	            !ggml_backend_buffer_is_host(tensor->buffer) &&
1862	            seq_to_row.size() > 1 &&
1863	            !mtp_backend_seed_force_host()) {
1864	            return;
1865	        }
1866	        mtp_capture_seed_rows_host(tensor, seq_to_row, dst, sched, row_size);
1867	    }
1868	}
...
2166	        if (native_mtp.enabled() && t_embd_raw && n_outputs > 0) {
2167	            const auto seq_to_output_row = build_seq_to_output_row(ubatch, 0);
2168	            capture_mtp_seed_rows(t_embd_raw, seq_to_output_row, native_mtp, sched.get());
2169	        }
```

## `src/llama-graph.cpp`

### Current backend-view MTP seed input

```cpp
206	void llm_graph_input_mtp_seed::set_input(const llama_ubatch * ubatch) {
...
209	    if (mode == LLAMA_MTP_SEED_MODE_HOST && t_seed && seed && n_mtp > 0) {
210	        ggml_backend_tensor_set(t_seed, seed, 0, (size_t) n_embd*n_mtp*ggml_element_size(t_seed));
211	    }
212	}
...
214	bool llm_graph_input_mtp_seed::can_reuse(const llm_graph_params & params) {
215	    mode = params.mtp_seed_mode;
216	    seed = params.mtp_seed;
217	    seed_backend = params.mtp_seed_backend;
218	    seed_generation = params.mtp_seed_generation;
219	    n_mtp = params.n_mtp;
...
229	        case LLAMA_MTP_SEED_MODE_HOST:
230	            return t_seed->view_src == nullptr;
231	        case LLAMA_MTP_SEED_MODE_BACKEND:
232	            return t_seed->view_src == seed_backend;
233	    }
...
1818	ggml_tensor * llm_graph_context::build_inp_mtp_seed() const {
1819	    auto inp = std::make_unique<llm_graph_input_mtp_seed>(n_embd, n_mtp, mtp_seed_mode, mtp_seed, mtp_seed_backend, mtp_seed_generation);
...
1823	    if (mtp_seed_mode == LLAMA_MTP_SEED_MODE_BACKEND) {
1824	        GGML_ASSERT(mtp_seed_backend != nullptr);
1825	
1826	        cur = ggml_view_2d(ctx0, mtp_seed_backend, n_embd, n_mtp, mtp_seed_backend->nb[1], 0);
1827	        GGML_ASSERT(cur != nullptr);
1828	        GGML_ASSERT(ggml_backend_view_init(cur) == GGML_STATUS_SUCCESS);
1829	    } else {
1830	        cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_mtp);
1831	    }
```

## `scripts/validate_mtp_cuda.py`

### Current exactness / throughput validation contract

```python
256    scenarios = [
257        ("baseline", 1, args.port_base + 0),
258        ("mtp", 1, args.port_base + 1),
259        ("baseline", 2, args.port_base + 2),
260        ("mtp", 2, args.port_base + 3),
261    ]
...
278    assert_equal_outputs(results[("baseline", 1)], results[("mtp", 1)])
279    assert_equal_outputs(results[("baseline", 2)], results[("mtp", 2)])
...
282    for n_parallel in (1, 2):
283        mtp = results[("mtp", n_parallel)]
284        if not any(resp.get("timings", {}).get("draft_n", 0) > 0 for resp in mtp.responses):
285            raise AssertionError(f"mtp np={n_parallel} did not report any native draft activity")
...
295    for n_parallel in (1, 2):
296        baseline = results[("baseline", n_parallel)]
297        mtp = results[("mtp", n_parallel)]
298        baseline_tps = [resp["timings"]["predicted_per_second"] for resp in baseline.responses]
299        mtp_tps = [resp["timings"]["predicted_per_second"] for resp in mtp.responses]
```

## Notes for the planner

- The next plan should assume the backend seed path is already landed and mostly not the bottleneck anymore.
- The strongest remaining lever looks server/runtime-side:
  - smarter decision to skip / shrink speculation on replay-heavy steps
  - cheaper replay after rejection
  - smaller per-step server hot-path cleanup
- Any proposed change should include a measurement point against:
  - baseline greedy decode
  - the previous native-MTP step on this branch
- If a proposed optimization only improves internal phase timings but not end-to-end tok/s on Berlin or Moon exact cases, it should be deprioritized or dropped.
