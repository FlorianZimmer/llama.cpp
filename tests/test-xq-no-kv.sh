#!/usr/bin/env bash
set -euo pipefail
TARGET=${1:?"binary to check"}
if nm "$TARGET" | grep -q llama_kv_cache; then
    echo "found llama_kv_cache symbol" >&2
    exit 1
fi
