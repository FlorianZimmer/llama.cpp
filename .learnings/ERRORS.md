## [ERR-20260409-001] python3 -m pip install --user

**Logged**: 2026-04-09T19:12:14Z
**Priority**: medium
**Status**: pending
**Area**: tests

### Summary
`pip install --user` is blocked by the distro-managed Python environment on this machine.

### Error
```text
error: externally-managed-environment

× This environment is externally managed
╰─> To install Python packages system-wide, try apt install
    python3-xyz, where xyz is the package you are trying to
    install.
```

### Context
- Command attempted: `python3 -m pip install --user pytest`
- Environment: Debian/Ubuntu-managed Python 3.12 with PEP 668 protections enabled
- This affects ad hoc test dependency installation in the repo workspace

### Suggested Fix
Create and use a local virtual environment for Python test tools instead of `pip install --user`.

### Metadata
- Reproducible: yes
- Related Files: tools/server/tests/unit/test_speculative.py

---

## [ERR-20260409-002] perf stat / perf record

**Logged**: 2026-04-09T21:44:00Z
**Priority**: medium
**Status**: pending
**Area**: backend

### Summary
Kernel perf events are unavailable in this environment, so process-level `perf` sampling cannot be used for server profiling.

### Error
```text
Error:
Access to performance monitoring and observability operations is limited.
...
perf_event_paranoid setting is 4
```

### Context
- Command attempted: `perf stat -e cycles,instructions -- sleep 0.1`
- Related probe: `perf record -F 199 -g -p <pid> -- sleep 12` produced a zero-sized data file
- Environment: `/proc/sys/kernel/perf_event_paranoid = 4`

### Suggested Fix
Use in-process wall-time instrumentation for performance analysis on this machine, or rerun profiling on a host with `perf_event_paranoid <= 1` or the required capabilities.

### Metadata
- Reproducible: yes
- Related Files: tools/server/server-context.cpp, scripts/validate_mtp_cuda.py
- See Also: LRN-20260409-001

---
