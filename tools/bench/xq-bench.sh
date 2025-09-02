#!/usr/bin/env bash
# Microbenchmark helper for XQuant
# Usage: ./xq-bench.sh <model.gguf>
#
# Searches for the llama-bench binary relative to this script. The
# lookup order is:
#   1. ../llama-bench/llama-bench
#   2. $LLAMA_BENCH_BIN if set
#   3. ../../build/bin/llama-bench

set -euo pipefail
MODEL=${1:?"model path required"}
LLAMA_BENCH_DIR="$(dirname "$0")/../llama-bench"
BIN="${LLAMA_BENCH_DIR}/llama-bench"

if [ ! -x "$BIN" ]; then
    BIN="${LLAMA_BENCH_BIN:-"$(dirname "$0")/../../build/bin/llama-bench"}"
fi

if [ ! -x "$BIN" ]; then
    echo "llama-bench binary not found at $BIN" >&2
    exit 1
fi

run_case() {
    local NAME=$1; shift
    echo "===== $NAME ====="
    if ! "$BIN" -m "$MODEL" "$@"; then
        echo "benchmark failed for $NAME" >&2
    fi
    echo
}

run_case baseline
run_case xquant4 --xquant --xq-bits 4
run_case xquant-cl3 --xquant-cl --xq-bits 3
