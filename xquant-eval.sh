#!/usr/bin/env bash

# xquant-eval.sh
# XQuant A/B runner (OFF vs ON) for macOS (bash 3.2+)
# Robust arg parsing, output logging, color support, CSV export

# Strict mode
set -eo pipefail
set -E
trap 'code=$?; echo "!! ERROR at line $LINENO: $BASH_COMMAND (exit $code)" >&2' ERR

# -------- Defaults --------
BIN_DIR="${BIN_DIR:-build/bin}"
RESULTS_DIR="${RESULTS_DIR:-results}"

MODEL=""
CTX=4096
P_LEN=1024
N_TOK=2048
BATCH=64
UBATCH=""
THREADS="$(sysctl -n hw.logicalcpu 2>/dev/null || echo 8)"
NGL=0
CTK=""
CTV=""

PPL_FILE=""
MAKE_DUMMY_PPL=false
GPU_LOG=false
DEBUG_MODE=false

# -------- Color Support --------
if [ -t 1 ] && tput setaf 1 >/dev/null 2>&1; then
    RED=$(tput setaf 1)
    GREEN=$(tput setaf 2)
    YELLOW=$(tput setaf 3)
    RESET=$(tput sgr0)
else
    RED=""
    GREEN=""
    YELLOW=""
    RESET=""
fi

# -------- Helpers --------
die() { echo "${RED}Error:${RESET} $*" >&2; exit 1; }
ensure_dir() { [ -d "$1" ] || mkdir -p "$1"; }
get_tool_path() {
    local dir="$1" base="$2"
    [ -x "$dir/$base" ] && echo "$dir/$base" && return
    [ -x "$dir/$base.exe" ] && echo "$dir/$base.exe" && return
    echo ""
}

help_has() {
    local help_text="$1" pattern="$2"
    printf '%s' "$help_text" | grep -qEi "$pattern" 2>/dev/null
}

fmt_bytes_mb() {
    local b="${1:-}"
    [ -z "$b" ] && echo "NA" && return
    awk -v b="$b" 'BEGIN{ if (b <= 0) print "NA"; else printf "%.1f MB", b/1048576.0 }'
}

to_mb_or_na_raw() {
    local b="${1:-}"
    [ -z "$b" ] && echo "NA" && return
    awk -v b="$b" 'BEGIN{ if (b <= 0) print "NA"; else printf "%.1f", b/1048576.0 }'
}

last_tokps() {
    local file="$1"
    [ -f "$file" ] || { echo ""; return; }
    awk '
        {
          for (i=1; i<=NF; i++) {
            if ($(i+1) ~ /^tok\/s$/ && $i ~ /^[0-9]+(\.[0-9]+)?$/) last=$i
          }
        }
        END { if (last!="") print last; }
    ' "$file"
}

compute_tokps_from_seconds() {
    local n="$1" secs="${2:-0}"
    awk -v n="$n" -v s="$secs" 'BEGIN{ if (s <= 0) print ""; else printf "%.2f", n/s }'
}

get_elapsed_from_timefile() {
    local file="$1"
    [ -f "$file" ] || { echo ""; return; }
    awk -F= '/^ELAPSED_SECONDS=/{print $2}' "$file" | tail -n1
}

get_maxrss_from_timefile() {
    local file="$1"
    [ -f "$file" ] || { echo ""; return; }
    awk -F= '/^MAXRSS=/{print $2}' "$file" | tail -n1
}

get_peak_vram_from_log() {
    local file="$1"
    [ -f "$file" ] || { echo ""; return; }
    awk 'NF==1 && $1 ~ /^[0-9]+(\.[0-9]+)?$/ { if ($1>max) max=$1 } END{ if (max!="") print max }' "$file"
}

vram_summary_string() {
    local gpu_log="$1" ngl="$2" have="$3" peak="${4:-}"
    if [ -n "$peak" ]; then
        echo "${peak} MB"
        return
    fi
    if [ "$gpu_log" != "true" ]; then
        echo "— (gpu-log off)"
        return
    fi
    if [ "$ngl" -le 0 ]; then
        echo "— (NGL=0/CPU)"
        return
    fi
    if [ "$have" != "true" ]; then
        echo "— (nvidia-smi missing)"
        return
    fi
    echo "—"
}

get_perplexity() {
    local file="$1"
    [ -f "$file" ] || { echo ""; return; }

    local text=$(cat "$file")
    local need=$(printf '%s' "$text" | sed -nE 's/.*you need at least[[:space:]]+([0-9]+)[[:space:]]+tokens.*/\1/ip' | tail -n1 | tr -d ' ')
    local have=$(printf '%s' "$text" | sed -nE 's/.*tokenizes to only[[:space:]]+([0-9]+)[[:space:]]+tokens.*/\1/ip' | tail -n1 | tr -d ' ')

    if [ -n "$need" ] && [ -n "$have" ]; then
        echo "NA (insufficient tokens: ${have}/${need})"
        return
    fi

    local val=$(printf '%s' "$text" | sed -nE 's/.*\bperplexity[[:space:]:=]+([0-9]+(\.[0-9]+)?).*/\1/ip' | tail -n1 | tr -d ' ')
    [ -n "$val" ] && echo "$val" && return

    val=$(printf '%s' "$text" | sed -nE 's/.*\bppl[[:space:]:=]+([0-9]+(\.[0-9]+)?).*/\1/ip' | tail -n1 | tr -d ' ')
    [ -n "$val" ] && echo "$val" && return

    val=$(printf '%s' "$text" | sed -nE 's/.*\bppx[[:space:]:=]+([0-9]+(\.[0-9]+)?).*/\1/ip' | tail -n1 | tr -d ' ')
    [ -n "$val" ] && echo "$val" && return

    awk '
        BEGIN{best=""}
        {
          low=tolower($0);
          if (low ~ /(^|[^a-z])ppl([^a-z]|$)/ || low ~ /\bperplexity\b/) {
            for (i=1;i<=NF;i++) if ($i ~ /^[0-9]+(\.[0-9]+)?$/) best=$i
          }
        }
        END{ if (best!="") print best }
    ' <<<"$text"
}

# -------- Usage --------
show_usage() {
    cat <<'USAGE'
Usage: xquant-eval.sh -m MODEL [options]

Required:
  -m, --model PATH          GGUF model path

Bench shape:
  -c, --ctx N               context length (default: 4096)
  -p, --prompt-len N        prompt tokens (default: 1024)
  -n, --n-tok N             tokens to generate (default: 2048)
  -b, --batch N             batch size (default: 64)
  --ubatch N                micro-batch size (if bench supports -ub)
  -t, --threads N           CPU threads (default: logical cores)
  -g, --ngl N               GPU layers (0=CPU, 99=all) (default: 0)
  --ctk TYPE                cache type for K (e.g., q4_0, f16)
  --ctv TYPE                cache type for V (e.g., q4_0, f16)

Paths:
  --bin-dir DIR             dir with llama-bench / llama-perplexity (default: build/bin)
  --results-dir DIR         output dir (default: results)

Perplexity:
  -f, --ppl-file PATH       wiki.test-60.raw (if omitted, PPL skipped unless --make-dummy)
  --make-dummy              create a 60-line dummy ppl file if -f omitted

GPU:
  --gpu-log                 sample VRAM via nvidia-smi (requires NGL>0)

Misc:
  --debug                   verbose shell tracing
  -h, --help                this help

Env toggles used: LLAMA_XQUANT (unset=OFF, 1=ON) and LLAMA_XQ_NOBASE (unset, 1).
USAGE
}

# -------- Arg Parsing --------
args=("$@")
i=0
argn=${#args[@]}

while [ $i -lt $argn ]; do
    a="${args[$i]}"
    case "$a" in
        -m|--model)       i=$((i+1)); [ $i -lt $argn ] || die "--model needs a value"; MODEL="${args[$i]}";;
        -c|--ctx)         i=$((i+1)); [ $i -lt $argn ] || die "--ctx needs a value"; CTX="${args[$i]}";;
        -p|--prompt-len)  i=$((i+1)); [ $i -lt $argn ] || die "--prompt-len needs a value"; P_LEN="${args[$i]}";;
        -n|--n-tok)       i=$((i+1)); [ $i -lt $argn ] || die "--n-tok needs a value"; N_TOK="${args[$i]}";;
        -b|--batch)       i=$((i+1)); [ $i -lt $argn ] || die "--batch needs a value"; BATCH="${args[$i]}";;
        --ubatch)         i=$((i+1)); [ $i -lt $argn ] || die "--ubatch needs a value"; UBATCH="${args[$i]}";;
        -t|--threads)     i=$((i+1)); [ $i -lt $argn ] || die "--threads needs a value"; THREADS="${args[$i]}";;
        -g|--ngl)         i=$((i+1)); [ $i -lt $argn ] || die "--ngl needs a value"; NGL="${args[$i]}";;
        --ctk)            i=$((i+1)); [ $i -lt $argn ] || die "--ctk needs a value"; CTK="${args[$i]}";;
        --ctv)            i=$((i+1)); [ $i -lt $argn ] || die "--ctv needs a value"; CTV="${args[$i]}";;
        -f|--ppl-file)   i=$((i+1)); [ $i -lt $argn ] || die "--ppl-file needs a value"; PPL_FILE="${args[$i]}";;
        --flash-attn)     i=$((i+1)); [ $i -lt $argn ] || die "--flash-attn needs a value"; bench_flash_mode="${args[$i]}";;
        --make-dummy)     MAKE_DUMMY_PPL=true;;
        --gpu-log)        GPU_LOG=true;;
        --bin-dir)        i=$((i+1)); [ $i -lt $argn ] || die "--bin-dir needs a value"; BIN_DIR="${args[$i]}";;
        --results-dir)    i=$((i+1)); [ $i -lt $argn ] || die "--results-dir needs a value"; RESULTS_DIR="${args[$i]}";;
        --debug)          DEBUG_MODE=true;;
        -h|--help)        show_usage; exit 0;;
        *) echo "Unknown arg: $a" >&2; show_usage; exit 1;;
    esac
    i=$((i+1))
done

[ "${DEBUG_MODE}" = "true" ] && { set -x; echo "${YELLOW}>>> ARGS:${RESET} $*"; }

[ -n "$MODEL" ] || die "--model is required"

# Enable strict mode after parsing
set -u

# -------- Discover Tools --------
bench_path="$(get_tool_path "$BIN_DIR" "llama-bench")"
[ -n "$bench_path" ] || die "$BIN_DIR/llama-bench not found"

ppl_path=""
have_ppl=true
ppl_path="$(get_tool_path "$BIN_DIR" "llama-perplexity")"
if [ -z "$ppl_path" ]; then
    echo "${YELLOW}Note:${RESET} $BIN_DIR/llama-perplexity not found; perplexity step will be skipped."
    have_ppl=false
fi

have_nvsmi=false
command -v nvidia-smi >/dev/null 2>&1 && have_nvsmi=true

# -------- Run Directory --------
ts="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${RESULTS_DIR}/xquant_${ts}"
ensure_dir "$RUN_DIR"

echo "${GREEN}>>> Writing logs to:${RESET} $RUN_DIR"
echo "${GREEN}>>> Model:${RESET} $MODEL"
echo "${GREEN}>>> CTX=$CTX P_LEN=$P_LEN N_TOK=$N_TOK BATCH=$BATCH THREADS=$THREADS NGL=$NGL${RESET}"

# -------- Prepare PPL File if Needed --------
if [ -z "$PPL_FILE" ] && [ "$MAKE_DUMMY_PPL" = "true" ]; then
    PPL_FILE="${RUN_DIR}/wiki.test-60.raw"
    { for _ in $(seq 1 60); do echo 'The quick brown fox jumps over the lazy dog.'; done; } > "$PPL_FILE"
    echo "${GREEN}>>> Created dummy ppl file:${RESET} $PPL_FILE"
fi
if [ -n "$PPL_FILE" ] && [ ! -f "$PPL_FILE" ]; then
    die "ppl file not found: $PPL_FILE"
fi

# -------- Auto-detect Flags from Help --------
bench_help="$("$bench_path" -h 2>&1 || true)"
ppl_help=""
if [ "$have_ppl" = true ]; then
    ppl_help="$("$ppl_path" -h 2>&1 || true)"
fi

bench_has_ctx=false
help_has "$bench_help" '(^|\s)-c(,|\s)|--ctx' && bench_has_ctx=true

bench_has_ub=false
help_has "$bench_help" '(^|\s)-ub(\s|,)|--ubatch' && bench_has_ub=true

bench_has_ctk=false
help_has "$bench_help" '-ctk|--cache-type-k' && bench_has_ctk=true

bench_has_ctv=false
help_has "$bench_help" '-ctv|--cache-type-v' && bench_has_ctv=true

# Initialize these variables to avoid "unbound variable" error
ppl_has_ctx=false
ppl_has_batch=false
ppl_has_chunks=false
ppl_has_ngl=false
ppl_has_ctk=false
ppl_has_ctv=false

flash_mode_from_help() {
    local h="$1"
    if printf '%s' "$h" | grep -qEi '\-\-flash-attn[^[:alnum:]_/-]*<0\|1>'; then
        echo valued
    elif printf '%s' "$h" | grep -qEi '(^|[^-])\-\-flash-attn([^[:alnum:]_-]|$)'; then
        echo flag
    else
        echo none
    fi
}

bench_flash_mode=$(flash_mode_from_help "$bench_help")
ppl_flash_mode=$(flash_mode_from_help "$ppl_help")

# Auto-detect ppl flags
help_has "$ppl_help" '(^|\s)-c(,|\s)|--ctx' && ppl_has_ctx=true
help_has "$ppl_help" '(^|\s)-b(,|\s)|--batch(-size)?' && ppl_has_batch=true
help_has "$ppl_help" '--chunks' && ppl_has_chunks=true
help_has "$ppl_help" '-ngl' && ppl_has_ngl=true
help_has "$ppl_help" '-ctk' && ppl_has_ctk=true
help_has "$ppl_help" '-ctv' && ppl_has_ctv=true

# -------- Process Runner with RSS + Optional VRAM --------
run_logged_process() {
    local out_file="$1"
    local time_file="$2"
    local vram_enabled="$3"
    local exe="$4"
    shift 4

    local vram_log="${out_file%.out}.vram"
    local vram_job=""

    "$exe" "$@" >"$out_file" 2>&1 &
    local pid=$!

    if [ "$vram_enabled" = "true" ] && [ "$have_nvsmi" = "true" ]; then
        (
            while kill -0 "$pid" 2>/dev/null; do
                nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 >>"$vram_log" || true
                sleep 1
            done
        ) &
        vram_job=$!
    fi

    local start end
    start="$(date +%s)"
    local max_kb=0

    while kill -0 "$pid" 2>/dev/null; do
        local rss
        rss="$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
        if [ -n "$rss" ] && [[ "$rss" =~ ^[0-9]+$ ]]; then
            [ "$rss" -gt "$max_kb" ] && max_kb="$rss"
        fi
        sleep 5
    done

    wait "$pid" || true
    end="$(date +%s)"
    local elapsed=$(( end - start ))
    local max_bytes=$(( max_kb * 1024 ))

    {
        printf "ELAPSED_SECONDS=%s\n" "$elapsed"
        printf "MAXRSS=%s\n" "$max_bytes"
    } >"$time_file"

    if [ -n "${vram_job:-}" ]; then
        wait "$vram_job" 2>/dev/null || true
    fi
}

# -------- Bench Mode --------
run_bench_mode() {
    local mode="$1"
    local out="${RUN_DIR}/bench_${mode}.out"
    local timefile="${RUN_DIR}/bench_${mode}.time"

    local args=( -m "$MODEL" -p "$P_LEN" -n "$N_TOK" -b "$BATCH" -t "$THREADS" -ngl "$NGL" )

    case "$bench_flash_mode" in
        valued) args+=( --flash-attn 1 ) ;;
        flag)   args+=( --flash-attn )   ;;
    esac

    if [ -n "$UBATCH" ] && [ "$bench_has_ub" = true ]; then args+=( -ub "$UBATCH" ); fi
    if [ -n "$CTK" ] && [ "$bench_has_ctk" = true ]; then args+=( -ctk "$CTK" ); fi
    if [ -n "$CTV" ] && [ "$bench_has_ctv" = true ]; then args+=( -ctv "$CTV" ); fi

    local vram_enable="false"
    if [ "$GPU_LOG" = "true" ] && [ "$NGL" -gt 0 ] && [ "$have_nvsmi" = "true" ]; then
        vram_enable="true"
    fi

    echo "${GREEN}>>> BENCH ($mode) ...${RESET}"
    if [ "$DEBUG_MODE" = true ]; then
        echo "[debug] bench cmd: $bench_path ${args[*]}"
    fi

    if [ "$mode" = "on" ]; then
        ( export LLAMA_XQUANT=1 LLAMA_XQ_NOBASE=0; run_logged_process "$out" "$timefile" "$vram_enable" "$bench_path" "${args[@]}" )
    else
        ( unset LLAMA_XQUANT LLAMA_XQ_NOBASE; run_logged_process "$out" "$timefile" "$vram_enable" "$bench_path" "${args[@]}" )
    fi
}

# -------- PPL Mode --------
run_ppl_mode() {
    local mode="$1"
    [ "$have_ppl" = true ] && [ -n "$PPL_FILE" ] || { echo "${YELLOW}>>> PPL ($mode) skipped.${RESET}"; return; }

    local out="${RUN_DIR}/ppl_${mode}.out"
    local args=( -m "$MODEL" -f "$PPL_FILE" )

    case "$ppl_flash_mode" in
        valued) args+=( --flash-attn 1 ) ;;
        flag)   args+=( --flash-attn )   ;;
    esac

    [ "$ppl_has_ctk" = true ] && [ -n "$CTK" ] && args+=( -ctk "$CTK" )
    [ "$ppl_has_ctv" = true ] && [ -n "$CTV" ] && args+=( -ctv "$CTV" )
    [ "$ppl_has_ctx" = true ] && args+=( -c "$CTX" )
    [ "$ppl_has_batch" = true ] && args+=( -b "$BATCH" )
    [ "$ppl_has_chunks" = true ] && args+=( --chunks 1 )
    [ "$ppl_has_ngl" = true ] && args+=( -ngl "$NGL" )

    echo "${GREEN}>>> PPL ($mode) ...${RESET}"
    if [ "$DEBUG_MODE" = true ]; then
        echo "[debug] ppl cmd: $ppl_path ${args[*]}"
    fi

    if [ "$mode" = "on" ]; then
        ( export LLAMA_XQUANT=1 LLAMA_XQ_NOBASE=0; "$ppl_path" "${args[@]}" >"$out" 2>&1 ) || true
    else
        ( unset LLAMA_XQUANT LLAMA_XQ_NOBASE; "$ppl_path" "${args[@]}" >"$out" 2>&1 ) || true
    fi
}

# -------- RUN --------
run_bench_mode off
run_bench_mode on
#run_ppl_mode off
#run_ppl_mode on

# -------- PARSE --------
bench_off_out="${RUN_DIR}/bench_off.out"
bench_on_out="${RUN_DIR}/bench_on.out"
bench_off_time="${RUN_DIR}/bench_off.time"
bench_on_time="${RUN_DIR}/bench_on.time"

tok_off="$(last_tokps "$bench_off_out")"
tok_on="$(last_tokps "$bench_on_out")"

if [ -z "$tok_off" ]; then
    sec_off="$(get_elapsed_from_timefile "$bench_off_time")"
    [ -n "$sec_off" ] && tok_off="$(compute_tokps_from_seconds "$N_TOK" "$sec_off")"
fi

if [ -z "$tok_on" ]; then
    sec_on="$(get_elapsed_from_timefile "$bench_on_time")"
    [ -n "$sec_on" ] && tok_on="$(compute_tokps_from_seconds "$N_TOK" "$sec_on")"
fi

speedup="NA"
if [ -n "${tok_off:-}" ] && [ -n "${tok_on:-}" ]; then
    speedup="$(awk -v off="$tok_off" -v on="$tok_on" 'BEGIN{ if (off>0) printf "%.2f", on/off; else print "NA" }')"
fi

rss_off_bytes="$(get_maxrss_from_timefile "$bench_off_time")"
rss_on_bytes="$(get_maxrss_from_timefile "$bench_on_time")"

off_peak="$(get_peak_vram_from_log "${bench_off_out%.out}.vram")"
on_peak="$(get_peak_vram_from_log "${bench_on_out%.out}.vram")"

vram_off="$(vram_summary_string "$GPU_LOG" "$NGL" "$have_nvsmi" "$off_peak")"
vram_on="$(vram_summary_string "$GPU_LOG" "$NGL" "$have_nvsmi" "$on_peak")"

ppl_off="NA"; ppl_on="NA"
ppl_off_file="${RUN_DIR}/ppl_off.out"
ppl_on_file="${RUN_DIR}/ppl_on.out"
[ -f "$ppl_off_file" ] && { v="$(get_perplexity "$ppl_off_file")"; [ -n "$v" ] && ppl_off="$v"; }
[ -f "$ppl_on_file" ] && { v="$(get_perplexity "$ppl_on_file")"; [ -n "$v" ] && ppl_on="$v"; }

# -------- SUMMARY --------
echo
echo "${GREEN}==================== XQuant A/B Summary ====================${RESET}"
echo "Model     : $MODEL"
echo "Params    : CTX=$CTX P_LEN=$P_LEN N_TOK=$N_TOK BATCH=$BATCH THREADS=$THREADS NGL=$NGL"
echo "Bench OFF : ${tok_off:-NA} tok/s | MaxRSS=$(fmt_bytes_mb "$rss_off_bytes") | VRAM=$vram_off"
echo "Bench ON  : ${tok_on:-NA} tok/s | MaxRSS=$(fmt_bytes_mb "$rss_on_bytes") | VRAM=$vram_on"
echo "Speedup   : $speedup x"
echo "PPL OFF   : $ppl_off"
echo "PPL ON    : $ppl_on"
echo "Logs      : $RUN_DIR"
echo "${GREEN}============================================================${RESET}"

# -------- CSV --------
CSV="${RUN_DIR}/summary.csv"
echo "model,ctx,p_len,n_tok,batch,threads,ngl,tok_off,tok_on,speedup,maxrss_off_mb,maxrss_on_mb,vram_off_mb,vram_on_mb,ppl_off,ppl_on" >"$CSV"

printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"%s","%s",%s,%s\n' \
    "$MODEL" "$CTX" "$P_LEN" "$N_TOK" "$BATCH" "$THREADS" "$NGL" \
    "${tok_off:-NA}" "${tok_on:-NA}" "$speedup" \
    "$(to_mb_or_na_raw "$rss_off_bytes")" "$(to_mb_or_na_raw "$rss_on_bytes")" \
    "${vram_off% MB}" "${vram_on% MB}" \
    "$ppl_off" "$ppl_on" >>"$CSV"

echo "${GREEN}CSV       :${RESET} $CSV"