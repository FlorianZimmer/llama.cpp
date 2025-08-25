#!/usr/bin/env bash
# xquant-eval-safe.sh — ultra-robust macOS runner for XQuant A/B
# - No 'set -e' or traps. Every step logs status and exits only with a final summary.
# - Uses explicit error checks and continues when possible.

# -------- defaults --------
BIN_DIR="${BIN_DIR:-build/bin}"
RESULTS_DIR="${RESULTS_DIR:-results}"

MODEL=""
CTX=4096
P_LEN=1024
N_TOK=2048
BATCH=1
UBATCH=""
THREADS="$(sysctl -n hw.logicalcpu 2>/dev/null || echo 8)"
NGL=0
CTK=""
CTV=""

PPL_FILE=""
MAKE_DUMMY_PPL=false
GPU_LOG=false
DEBUG_MODE=false

echo "[BOOT] xquant-eval-safe.sh starting (bash ${BASH_VERSION:-unknown})"

# -------- helpers --------
die(){ echo "Error: $*" >&2; exit 1; }
ensure_dir(){ [ -d "$1" ] || mkdir -p "$1"; }
get_tool_path(){
  local dir="$1" base="$2"
  if [ -x "$dir/$base" ]; then echo "$dir/$base"
  elif [ -x "$dir/$base.exe" ]; then echo "$dir/$base.exe"
  else echo ""; fi
}
help_has(){ printf "%s" "$1" | grep -qiE "$2" >/dev/null 2>&1; } # never fatal
fmt_bytes_mb(){ local b="${1:-}"; [ -z "$b" ] && { echo "NA"; return; }; awk -v b="$b" 'BEGIN{ if (b+0<=0) print "NA"; else printf "%.1f MB", b/1048576.0 }'; }
to_mb_or_na_raw(){ local b="${1:-}"; [ -z "$b" ] && { echo "NA"; return; }; awk -v b="$b" 'BEGIN{ if (b+0<=0) print "NA"; else printf "%.1f", b/1048576.0 }'; }
last_tokps(){
  local f="$1"; [ -f "$f" ] || { echo ""; return; }
  awk '{for(i=1;i<=NF;i++) if($(i+1)~/^tok\/s$/ && $i~/^[0-9]+(\.[0-9]+)?$/) last=$i} END{if(last!="")print last}' "$f"
}
compute_tokps_from_seconds(){ awk -v n="$1" -v s="${2:-0}" 'BEGIN{ if (s<=0) print ""; else printf "%.2f", n/s }'; }
get_elapsed_from_timefile(){ local f="$1"; [ -f "$f" ] && awk -F= '/^ELAPSED_SECONDS=/{print $2}' "$f" | tail -n1 || echo ""; }
get_maxrss_from_timefile(){ local f="$1"; [ -f "$f" ] && awk -F= '/^MAXRSS=/{print $2}' "$f" | tail -n1 || echo ""; }
get_peak_vram_from_log(){ local f="$1"; [ -f "$f" ] && awk 'NF==1 && $1~/^[0-9]+(\.[0-9]+)?$/{if($1>m)m=$1} END{if(m!="")print m}' "$f" || echo ""; }
vram_summary_string(){
  local gpu="$1" ngl="$2" have="$3" peak="${4:-}"
  [ -n "$peak" ] && { echo "${peak} MB"; return; }
  [ "$gpu" != "true" ] && { echo "— (gpu-log off)"; return; }
  [ "$ngl" -le 0 ] && { echo "— (NGL=0/CPU)"; return; }
  [ "$have" != "true" ] && { echo "— (nvidia-smi missing)"; return; }
  echo "—"
}
get_perplexity(){
  local file="$1"; [ -f "$file" ] || { echo ""; return; }
  local text; text="$(cat "$file")"
  local need have val
  need="$(printf "%s" "$text" | sed -nE 's/.*you need at least[[:space:]]+([0-9]+)[[:space:]]+tokens.*/\1/ip' | tail -n1 | tr -d ' ')"
  have="$(printf "%s" "$text" | sed -nE 's/.*tokenizes to only[[:space:]]+([0-9]+)[[:space:]]+tokens.*/\1/ip' | tail -n1 | tr -d ' ')"
  if [ -n "$need" ] && [ -n "$have" ]; then echo "NA (insufficient tokens: ${have}/${need})"; return; fi
  for rx in perplexity ppl ppx; do
    val="$(printf "%s" "$text" | sed -nE "s/.*\\b${rx}[[:space:]:=]+([0-9]+(\\.[0-9]+)?).*/\\1/ip" | tail -n1 | tr -d ' ')"
    [ -n "$val" ] && { echo "$val"; return; }
  done
  awk '
    BEGIN{best=""}
    { low=tolower($0); if (low ~ /(^|[^a-z])ppl([^a-z]|$)/ || low ~ /\bperplexity\b/)
      for (i=1;i<=NF;i++) if ($i ~ /^[0-9]+(\.[0-9]+)?$/) best=$i }
    END{ if (best!="") print best }' <<<"$text"
}

# -------- arg parsing --------
show_usage(){
cat <<'USAGE'
Usage: xquant-eval-safe.sh -m MODEL [options]
  -m, --model PATH     GGUF model path (required)
  -c, --ctx N          context (default 4096)
  -p, --prompt-len N   prompt tokens (default 1024)
  -n, --n-tok N        generate tokens (default 2048)
  -b, --batch N        batch size (default 1)
  --ubatch N           micro-batch (if supported)
  -t, --threads N      CPU threads (default: logical cores)
  -g, --ngl N          GPU layers (default 0)
  --ctk TYPE           KV K type (e.g. q4_0, f16)
  --ctv TYPE           KV V type (e.g. q4_0, f16)
  --bin-dir DIR        dir with llama-bench / llama-perplexity (default build/bin)
  --results-dir DIR    output dir (default results)
  -f, --ppl-file PATH  wiki.test-60.raw path
  --make-dummy         create dummy ppl file if missing
  --gpu-log            sample VRAM via nvidia-smi
  --debug              verbose tracing
  -h, --help           this help
USAGE
}

args=("$@"); i=0
while [ $i -lt $# ]; do
  a="${args[$i]}"
  case "$a" in
    -m|--model)        ((i++)); MODEL="${args[$i]:-}";;
    -c|--ctx)          ((i++)); CTX="${args[$i]:-}";;
    -p|--prompt-len)   ((i++)); P_LEN="${args[$i]:-}";;
    -n|--n-tok)        ((i++)); N_TOK="${args[$i]:-}";;
    -b|--batch)        ((i++)); BATCH="${args[$i]:-}";;
    --ubatch)          ((i++)); UBATCH="${args[$i]:-}";;
    -t|--threads)      ((i++)); THREADS="${args[$i]:-}";;
    -g|--ngl)          ((i++)); NGL="${args[$i]:-}";;
    --ctk)             ((i++)); CTK="${args[$i]:-}";;
    --ctv)             ((i++)); CTV="${args[$i]:-}";;
    -f|--ppl-file)     ((i++)); PPL_FILE="${args[$i]:-}";;
    --make-dummy)      MAKE_DUMMY_PPL=true;;
    --gpu-log)         GPU_LOG=true;;
    --bin-dir)         ((i++)); BIN_DIR="${args[$i]:-}";;
    --results-dir)     ((i++)); RESULTS_DIR="${args[$i]:-}";;
    --debug)           DEBUG_MODE=true;;
    -h|--help)         show_usage; exit 0;;
    *) echo "Unknown arg: $a"; show_usage; exit 1;;
  esac
  ((i++))
done

$DEBUG_MODE && { echo "[DEBUG] enabled"; set -x; }

[ -n "$MODEL" ] || die "--model is required"

# -------- discover tools --------
bench_path="$(get_tool_path "$BIN_DIR" "llama-bench")"
if [ -z "$bench_path" ]; then
  echo "[WARN] $BIN_DIR/llama-bench not found or not executable."
  echo "       BIN_DIR='$BIN_DIR'     PWD='$(pwd)'"
  echo "       Contents of BIN_DIR:"; ls -la "$BIN_DIR" 2>/dev/null || true
  die "llama-bench missing"
fi
echo "[OK] Found llama-bench at: $bench_path"

ppl_path="$(get_tool_path "$BIN_DIR" "llama-perplexity")"
have_ppl="true"
if [ -z "$ppl_path" ]; then
  echo "[NOTE] $BIN_DIR/llama-perplexity not found; PPL step will be skipped."
  have_ppl="false"
else
  echo "[OK] Found llama-perplexity at: $ppl_path"
fi

have_nvsmi="false"; if command -v nvidia-smi >/dev/null 2>&1; then have_nvsmi="true"; fi

# -------- run dir --------
ts="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${RESULTS_DIR}/xquant_${ts}"
ensure_dir "$RUN_DIR"
echo "[OK] Logs -> $RUN_DIR"

echo ">>> Model: $MODEL"
echo ">>> CTX=$CTX P_LEN=$P_LEN N_TOK=$N_TOK BATCH=$BATCH THREADS=$THREADS NGL=$NGL"

# -------- ppl file --------
if [ -z "$PPL_FILE" ] && [ "$MAKE_DUMMY_PPL" = "true" ]; then
  PPL_FILE="${RUN_DIR}/wiki.test-60.raw"
  for _ in $(seq 1 60); do echo 'The quick brown fox jumps over the lazy dog.'; done >"$PPL_FILE"
  echo "[OK] Created dummy ppl file: $PPL_FILE"
fi
if [ -n "$PPL_FILE" ] && [ ! -f "$PPL_FILE" ]; then
  echo "[WARN] ppl file not found: $PPL_FILE — skipping ppl step."
  have_ppl="false"
fi

# -------- flag auto-detection --------
bench_help="$("$bench_path" -h 2>&1 || true)"
ppl_help=""; [ "$have_ppl" = "true" ] && ppl_help="$("$ppl_path" -h 2>&1 || true)"

bench_has_ctx=false;  help_has "$bench_help" '(^|\s)-c(,|\s)|--ctx' && bench_has_ctx=true
bench_has_ub=false;   help_has "$bench_help" '(^|\s)-ub(\s|,)|--ubatch' && bench_has_ub=true
bench_has_ctk=false;  help_has "$bench_help" '-ctk|--cache-type-k' && bench_has_ctk=true
bench_has_ctv=false;  help_has "$bench_help" '-ctv|--cache-type-v' && bench_has_ctv=true

ppl_has_ctx=false;    help_has "$ppl_help" '(^|\s)-c(,|\s)|--ctx' && ppl_has_ctx=true
ppl_has_batch=false;  help_has "$ppl_help" '(^|\s)-b(,|\s)|--batch(-size)?' && ppl_has_batch=true
ppl_has_chunks=false; help_has "$ppl_help" '--chunks' && ppl_has_chunks=true
ppl_has_ngl=false;    help_has "$ppl_help" '-ngl' && ppl_has_ngl=true
ppl_has_ctk=false;    help_has "$ppl_help" '-ctk' && ppl_has_ctk=true
ppl_has_ctv=false;    help_has "$ppl_help" '-ctv' && ppl_has_ctv=true

echo "[INFO] bench flags: ctx=$bench_has_ctx ub=$bench_has_ub ctk=$bench_has_ctk ctv=$bench_has_ctv"
[ "$have_ppl" = "true" ] && echo "[INFO] ppl flags  : ctx=$ppl_has_ctx batch=$ppl_has_batch chunks=$ppl_has_chunks ngl=$ppl_has_ngl ctk=$ppl_has_ctk ctv=$ppl_has_ctv"

# -------- runner --------
run_logged_process() {
  local out_file="$1"; shift
  local time_file="$1"; shift
  local vram_enabled="$1"; shift
  local exe="$1"; shift
  local vram_log="${out_file%.out}.vram"

  echo "[RUN] $exe $*"
  "$exe" "$@" >"$out_file" 2>&1 &
  local pid=$!
  echo "[RUN] pid=$pid -> $out_file"

  # VRAM sampler
  local vram_job=""
  if [ "$vram_enabled" = "true" ] && [ "$have_nvsmi" = "true" ]; then
    (
      while kill -0 "$pid" 2>/dev/null; do
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 >>"$vram_log" || true
        sleep 0.5
      done
    ) & vram_job=$!
  fi

  # RSS sampler (ps rss in KB)
  local start end; start="$(date +%s)"
  local max_kb=0
  while kill -0 "$pid" 2>/dev/null; do
    local rss; rss="$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    if [[ "$rss" =~ ^[0-9]+$ ]] && [ "$rss" -gt "$max_kb" ]; then max_kb="$rss"; fi
    sleep 0.2
  done
  wait "$pid" 2>/dev/null
  local rc=$?
  end="$(date +%s)"
  local elapsed=$(( end - start ))
  local max_bytes=$(( max_kb * 1024 ))
  printf "ELAPSED_SECONDS=%s\nMAXRSS=%s\n" "$elapsed" "$max_bytes" >"$time_file"
  echo "[DONE] pid=$pid rc=$rc elapsed=${elapsed}s maxrss=${max_bytes}B"
}

run_bench_mode() {
  local mode="$1"
  local out="${RUN_DIR}/bench_${mode}.out"
  local timefile="${RUN_DIR}/bench_${mode}.time"
  local args=( -m "$MODEL" -p "$P_LEN" -n "$N_TOK" -b "$BATCH" -t "$THREADS" -ngl "$NGL" --flash-attn 1 )
  $bench_has_ctx && args+=( -c "$CTX" )
  [ -n "$UBATCH" ] && $bench_has_ub && args+=( -ub "$UBATCH" )
  [ -n "$CTK" ] && $bench_has_ctk && args+=( -ctk "$CTK" )
  [ -n "$CTV" ] && $bench_has_ctv && args+=( -ctv "$CTV" )

  local vram_enable="false"
  [ "$GPU_LOG" = "true" ] && [ "$NGL" -gt 0 ] && [ "$have_nvsmi" = "true" ] && vram_enable="true"

  echo ">>> BENCH ($mode) ..."
  if [ "$mode" = "on" ]; then
    ( LLAMA_XQUANT=1 LLAMA_XQ_NOBASE=1; run_logged_process "$out" "$timefile" "$vram_enable" "$bench_path" "${args[@]}" )
  else
    ( unset LLAMA_XQUANT LLAMA_XQ_NOBASE; run_logged_process "$out" "$timefile" "$vram_enable" "$bench_path" "${args[@]}" )
  fi
}

run_ppl_mode() {
  local mode="$1"
  [ "$have_ppl" = "true" ] && [ -n "$PPL_FILE" ] || { echo ">>> PPL ($mode) skipped."; return; }

  local args=( -m "$MODEL" -f "$PPL_FILE" --flash-attn 1 )
  $ppl_has_ctk   && [ -n "$CTK" ] && args+=( -ctk "$CTK" )
  $ppl_has_ctv   && [ -n "$CTV" ] && args+=( -ctv "$CTV" )
  $ppl_has_ctx   && args+=( -c "$CTX" )
  $ppl_has_batch && args+=( -b "$BATCH" )
  $ppl_has_chunks && args+=( --chunks 1 )
  $ppl_has_ngl   && args+=( -ngl "$NGL" )

  local out="${RUN_DIR}/ppl_${mode}.out"
  echo ">>> PPL ($mode) ..."
  if [ "$mode" = "on" ]; then
    ( LLAMA_XQUANT=1 LLAMA_XQ_NOBASE=1; "$ppl_path" "${args[@]}" >"$out" 2>&1 || true )
  else
    ( unset LLAMA_XQUANT LLAMA_XQ_NOBASE; "$ppl_path" "${args[@]}" >"$out" 2>&1 || true )
  fi
}

# -------- RUN --------
run_bench_mode off
run_bench_mode on
run_ppl_mode   off
run_ppl_mode   on

# -------- PARSE & REPORT --------
bench_off_out="${RUN_DIR}/bench_off.out"; bench_on_out="${RUN_DIR}/bench_on.out"
bench_off_time="${RUN_DIR}/bench_off.time"; bench_on_time="${RUN_DIR}/bench_on.time"

tok_off="$(last_tokps "$bench_off_out")"; tok_on="$(last_tokps "$bench_on_out")"
[ -z "$tok_off" ] && { sec_off="$(get_elapsed_from_timefile "$bench_off_time")"; [ -n "$sec_off" ] && tok_off="$(compute_tokps_from_seconds "$N_TOK" "$sec_off")"; }
[ -z "$tok_on" ]  && { sec_on="$(get_elapsed_from_timefile "$bench_on_time")"; [ -n "$sec_on" ] && tok_on="$(compute_tokps_from_seconds "$N_TOK" "$sec_on")"; }

speedup="NA"; if [ -n "${tok_off:-}" ] && [ -n "${tok_on:-}" ] && awk "BEGIN{exit !($tok_off>0)}"; then
  speedup="$(awk -v off="$tok_off" -v on="$tok_on" 'BEGIN{ if (off>0) printf "%.2f", on/off; else print "NA" }')"
fi

rss_off_bytes="$(get_maxrss_from_timefile "$bench_off_time")"
rss_on_bytes="$(get_maxrss_from_timefile "$bench_on_time")"

off_peak="$(get_peak_vram_from_log "${bench_off_out%.out}.vram")"
on_peak="$(get_peak_vram_from_log  "${bench_on_out%.out}.vram")"
vram_off="$(vram_summary_string "$GPU_LOG" "$NGL" "$have_nvsmi" "$off_peak")"
vram_on="$(vram_summary_string  "$GPU_LOG" "$NGL" "$have_nvsmi" "$on_peak")"

ppl_off="NA"; ppl_on="NA"
ppl_off_file="${RUN_DIR}/ppl_off.out"; [ -f "$ppl_off_file" ] && { v="$(get_perplexity "$ppl_off_file")"; [ -n "$v" ] && ppl_off="$v"; }
ppl_on_file="${RUN_DIR}/ppl_on.out";   [ -f "$ppl_on_file"  ] && { v="$(get_perplexity "$ppl_on_file")";  [ -n "$v" ] && ppl_on="$v"; }

echo
echo "==================== XQuant A/B Summary ===================="
echo "Model     : $MODEL"
echo "Params    : CTX=$CTX P_LEN=$P_LEN N_TOK=$N_TOK BATCH=$BATCH THREADS=$THREADS NGL=$NGL"
echo "Bench OFF : ${tok_off:-NA} tok/s | MaxRSS=$(fmt_bytes_mb "$rss_off_bytes") | VRAM=$vram_off"
echo "Bench ON  : ${tok_on:-NA} tok/s | MaxRSS=$(fmt_bytes_mb "$rss_on_bytes") | VRAM=$vram_on"
echo "Speedup   : $speedup x"
echo "PPL OFF   : $ppl_off"
echo "PPL ON    : $ppl_on"
echo "Logs      : $RUN_DIR"
echo "============================================================"

CSV="${RUN_DIR}/summary.csv"
echo "model,ctx,p_len,n_tok,batch,threads,ngl,tok_off,tok_on,speedup,maxrss_off_mb,maxrss_on_mb,vram_off_mb,vram_on_mb,ppl_off,ppl_on" >"$CSV"
printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"%s","%s",%s,%s\n' \
  "$MODEL" "$CTX" "$P_LEN" "$N_TOK" "$BATCH" "$THREADS" "$NGL" \
  "${tok_off:-NA}" "${tok_on:-NA}" "$speedup" \
  "$(to_mb_or_na_raw "$rss_off_bytes")" "$(to_mb_or_na_raw "$rss_on_bytes")" \
  "${vram_off% MB}" "${vram_on% MB}" \
  "$ppl_off" "$ppl_on" >>"$CSV"
echo "CSV       : $CSV"