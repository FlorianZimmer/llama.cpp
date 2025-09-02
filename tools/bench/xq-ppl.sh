#!/usr/bin/env bash
# Perplexity smoke test for XQuant using WikiText-2
# Usage: ./xq-ppl.sh <model.gguf> [wikitext-2.txt]

set -euo pipefail

ensure_wikitext2() {
  local OUT="${1:-./wikitext-2.txt}"
  if [ -s "$OUT" ]; then return 0; fi
  echo "[xq-ppl] downloading WikiText-2 to $OUT ..." >&2

  urls=(
    "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/test.txt"
    "https://huggingface.co/datasets/wikitext/resolve/main/wikitext-2-raw-v1/test.txt"
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" # last-resort placeholder
  )
  for u in "${urls[@]}"; do
    if command -v curl >/dev/null 2>&1; then
      curl -L --retry 3 --connect-timeout 10 -o "$OUT.tmp" "$u" && mv "$OUT.tmp" "$OUT" && break
    else
      wget -O "$OUT.tmp" "$u" && mv "$OUT.tmp" "$OUT" && break
    fi
  done

  if [ ! -s "$OUT" ]; then
    echo "[xq-ppl] ERROR: could not download WikiText-2 (checked ${#urls[@]} mirrors)" >&2
    exit 1
  fi
}

# usage: if second arg missing, fetch locally
MODEL="${1:-}"; WT2="${2:-}"
if [ -z "$MODEL" ]; then echo "Usage: $0 <model.gguf> [wikitext-2.txt]"; exit 1; fi
if [ -z "$WT2" ]; then WT2="$(dirname "$0")/wikitext-2.txt"; fi
ensure_wikitext2 "$WT2"
DATA="$WT2"
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
