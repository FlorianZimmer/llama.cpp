#!/usr/bin/env bash
# Perplexity smoke test for XQuant using WikiText-2
# Usage: ./xq-ppl.sh <model.gguf> <wikitext-2.txt>

set -euo pipefail
MODEL=${1:?"model path required"}
DATA=${2:?"dataset required"}
PERP_DIR="$(dirname "$0")/../perplexity"
BIN="${PERP_DIR}/perplexity"

if [ ! -x "$BIN" ]; then
    echo "perplexity binary not found at $BIN" >&2
    exit 1
fi

run_case() {
    local NAME=$1; shift
    echo "===== $NAME ====="
    "$BIN" -m "$MODEL" -f "$DATA" "$@"
    echo
}

run_case baseline ""
run_case xquant4 --xquant --xq-bits 4
run_case xquant-cl3 --xquant-cl --xq-bits 3
