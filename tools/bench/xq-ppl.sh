#!/usr/bin/env bash
# Perplexity smoke test for XQuant using WikiText-2
# Usage: ./xq-ppl.sh <model.gguf> <wikitext-2.txt>

set -euo pipefail
MODEL=${1:?"model path required"}
DATA=${2:?"dataset required"}
PERP_DIR="$(dirname "$0")/../llama-perplexity"
BIN="${PERP_DIR}/llama-perplexity"

if [ ! -x "$BIN" ]; then
    BIN="${LLAMA_PERPLEXITY_BIN:-"$(dirname "$0")/../../build/bin/llama-perplexity"}"
fi

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

run_case baseline

# XQuant with optional GQA latent caching via SVD factors
SVD_PATH="${MODEL%.gguf}.xqsvd"
XQ_ARGS=()
if [ -f "$SVD_PATH" ]; then
    XQ_ARGS+=(--xq-gqa-svd --xq-svd-path "$SVD_PATH")
fi

run_case xquant4 --xquant --xq-bits 4 "${XQ_ARGS[@]}"
run_case xquant-cl3 --xquant-cl --xq-bits 3 "${XQ_ARGS[@]}"
