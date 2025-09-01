#!/usr/bin/env bash
# Microbenchmark helper for XQuant
# Usage: ./xq-bench.sh <model.gguf>

set -euo pipefail
MODEL=${1:?"model path required"}
LLAMA_BENCH_DIR="$(dirname "$0")/../llama-bench"
BIN="${LLAMA_BENCH_DIR}/llama-bench"

if [ ! -x "$BIN" ]; then
    echo "llama-bench binary not found at $BIN" >&2
    exit 1
fi

run_case() {
    local NAME=$1; shift
    echo "===== $NAME ====="
    "$BIN" -m "$MODEL" "$@"
    echo
}

run_case baseline ""
run_case xquant4 --xquant --xq-bits 4
run_case xquant-cl3 --xquant-cl --xq-bits 3
