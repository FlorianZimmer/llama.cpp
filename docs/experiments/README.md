# Experiment Index

This fork preserves a small number of branch families as research snapshots.

Current documented experiment lines:

- [Native MTP](native-mtp.md)
  - Qwen 3.5 native multi-token prediction work inside `llama.cpp`
  - current status: on hold
  - main branches:
    - `research/native-mtp-runtime-base`
    - `research/native-mtp-qwen35-dense-speedup`
- [XQuant](xquant.md)
  - XQuant-style KV rematerialization experiment
  - current status: on hold
  - main branch:
    - `research/xquant-on-hold`

Why this exists:

- keep the public fork understandable without turning `master` into an experiment branch
- preserve branch history and technical findings after active work stops
- make the current experiment scope visible from one place
