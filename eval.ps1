# xquant_eval.ps1
# XQuant A/B runner (OFF vs ON) for Windows PowerShell
# Mirrors the original bash behavior, including flag auto-detection and GPU VRAM sampling.

param(
  # Capture everything the user passes and parse it manually
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$argv
)

$ErrorActionPreference = 'Stop'
if (-not $argv) { $argv = @() }


$ErrorActionPreference = 'Stop'

function Build-ArgumentString {
    param([object[]]$Parts)

    $safe = @()
    foreach ($p in $Parts) {
        if ($null -eq $p) { continue }
        $s = [string]$p
        # quote if it has space or a quote
        if ($s -match '[\s"]') {
            $s = '"' + ($s -replace '"','`"') + '"'
        }
        $safe += $s
    }
    return ($safe -join ' ')
}
function Format-RssMb([object]$bytes) {
    if ($null -eq $bytes -or $bytes -eq '') { return 'NA' }
    try {
        $v = [double]$bytes
        return ("{0:N1} MB" -f ($v / 1MB))
    } catch { return 'NA' }
}

function Format-RssMb([object]$bytes) {
    if ($null -eq $bytes -or $bytes -eq '') { return 'NA' }
    try {
        $v = [double]$bytes
        return ("{0:N1} MB" -f ($v / 1MB))
    } catch { return 'NA' }
}

function Show-Usage {
@"
Usage: xquant_eval.ps1 -m MODEL [options]

Required:
  -m, --model PATH          GGUF model path (e.g., F:\LLMs\...)

Bench shape:
  -c, --ctx N               context length (default: 4096)
  -p, --prompt-len N        prompt tokens (default: 1024)
  -n, --n-tok N             tokens to generate (default: 2048)
  -b, --batch N             batch size (default: 64)
  --ubatch N                micro-batch size (only if bench supports -ub)
  -t, --threads N           CPU threads (default: logical cores)
  -g, --ngl N               GPU layers (0=CPU, 99=all if GPU build) (default: 0)
  --ctk TYPE                cache type for K (e.g., q4_0, f16)
  --ctv TYPE                cache type for V (e.g., q4_0, f16)

Paths:
  --bin-dir DIR             directory with llama-bench / llama-perplexity (default: build\bin)
  --results-dir DIR         where to write logs (default: results)

Perplexity:
  -f, --ppl-file PATH       path to wiki.test-60.raw (if omitted, ppl step skipped unless --make-dummy)
  --make-dummy              create a 60-line dummy ppl file if -f omitted

GPU:
  --gpu-log                 sample VRAM with nvidia-smi (requires NGL>0)

Misc:
  --debug                   verbose error info
  -h, --help                this help

Env toggle used: LLAMA_XQUANT (unset=OFF, 1=ON)
"@ | Write-Host
}

# ---- defaults ----
$BIN_DIR     = $env:BIN_DIR     ; if (-not $BIN_DIR)     { $BIN_DIR = 'build\bin' }
$RESULTS_DIR = $env:RESULTS_DIR ; if (-not $RESULTS_DIR) { $RESULTS_DIR = 'results' }

$MODEL   = ''
$CTX     = 4096
$P_LEN   = 1024
$N_TOK   = 2048
$BATCH   = 1
$UBATCH  = $null
$THREADS = [Environment]::ProcessorCount
$NGL     = 0
$CTK     = $null
$CTV     = $null

$PPL_FILE       = $null
$MAKE_DUMMY_PPL = $false
$GPU_LOG        = $false
$DEBUG_MODE     = $false

# ---- arg parsing (PowerShell-friendly) ----
for ($i = 0; $i -lt $argv.Count; $i++) {
    $a = $argv[$i]
    switch -regex ($a) {
        '^(?:-m|--model)$'        { $MODEL   = $argv[++$i]; continue }
        '^(?:-c|--ctx)$'          { $CTX     = [int]$argv[++$i]; continue }
        '^(?:-p|--prompt-len)$'   { $P_LEN   = [int]$argv[++$i]; continue }
        '^(?:-n|--n-tok)$'        { $N_TOK   = [int]$argv[++$i]; continue }
        '^(?:-b|--batch)$'        { $BATCH   = [int]$argv[++$i]; continue }
        '^--ubatch$'              { $UBATCH  = [int]$argv[++$i]; continue }
        '^(?:-t|--threads)$'      { $THREADS = [int]$argv[++$i]; continue }
        '^(?:-g|--ngl)$'          { $NGL     = [int]$argv[++$i]; continue }
        '^--ctk$'                 { $CTK     = $argv[++$i]; continue }
        '^--ctv$'                 { $CTV     = $argv[++$i]; continue }
        '^(?:-f|--ppl-file)$'     { $PPL_FILE = $argv[++$i]; continue }
        '^--make-dummy$'          { $MAKE_DUMMY_PPL = $true; continue }
        '^--gpu-log$'             { $GPU_LOG = $true; continue }
        '^--bin-dir$'             { $BIN_DIR = $argv[++$i]; continue }
        '^--results-dir$'         { $RESULTS_DIR = $argv[++$i]; continue }
        '^--debug$'               { $DEBUG_MODE = $true; continue }
        '^(?:-h|--help)$'         { Show-Usage; exit 0 }
        default {
            Write-Host "Unknown arg: $a" -ForegroundColor Yellow
            Show-Usage; exit 1
        }
    }
}


if ($DEBUG_MODE) {
    $global:LASTEXITCODE = 0
    if ($PSStyle) { # <-- Add this check
        $PSStyle.OutputRendering = 'Host'
    } # <-- And this closing brace
    Write-Host "[debug] args: $($argv -join ' ')" -ForegroundColor Gray
}
if (-not $MODEL) { Write-Error "Error: --model is required"; exit 2 }

# ---- helpers ----
function Get-ToolPath([string]$Dir, [string]$BaseName) {
    $candidates = @(
        (Join-Path $Dir "$BaseName.exe"),
        (Join-Path $Dir $BaseName)
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return (Resolve-Path $c).Path }
    }
    return $null
}

function Ensure-Dir([string]$path) {
    if (-not (Test-Path $path)) { New-Item -ItemType Directory -Path $path | Out-Null }
}

function Write-TimeLog([string]$path, [double]$secs, [nullable[int64]]$maxrss) {
    $lines = @("ELAPSED_SECONDS=$([Math]::Round($secs,2))")
    if ($maxrss -ne $null) { $lines += "MAXRSS=$maxrss" }
    Set-Content -Path $path -Value $lines -Encoding ascii
}

function Get-LastTokps([string]$file) {
    if (-not (Test-Path $file)) { return $null }
    $matches = Select-String -Path $file -Pattern '([0-9]+(?:\.[0-9]+)?)\s+tok/s' -AllMatches
    if ($matches) {
        $m = $matches.Matches | Select-Object -Last 1
        if ($m.Groups.Count -ge 2) { return [double]$m.Groups[1].Value }
    }
    return $null
}

function Get-ElapsedFromTimeLog([string]$file) {
    if (-not (Test-Path $file)) { return $null }
    $line = Get-Content -Path $file | Where-Object { $_ -match '^ELAPSED_SECONDS=' } | Select-Object -Last 1
    if ($line) { return [double]($line -replace '^ELAPSED_SECONDS=','') }
    return $null
}

function Get-MaxRssFromTimeLog([string]$file) {
    if (-not (Test-Path $file)) { return $null }
    $line = Get-Content -Path $file | Where-Object { $_ -match '^MAXRSS=' } | Select-Object -Last 1
    if ($line) { return ($line -replace '^MAXRSS=','') }
    return $null
}

function Compute-TokpsFromSeconds([int]$nTok, [double]$secs) {
    if ($secs -le 0) { return $null }
    return [Math]::Round($nTok / $secs, 2)
}

function Get-PeakVramFromLog([string]$file) {
    if (-not (Test-Path $file)) { return $null }
    $vals = Get-Content -Path $file | Where-Object { $_ -match '^[0-9]+(\.[0-9]+)?$' } | ForEach-Object { [double]$_ }
    if ($vals) { return ($vals | Measure-Object -Maximum).Maximum }
    return $null
}

function Get-VramSummaryString {
    param(
        [bool]$GpuLog,
        [int]$Ngl,
        [bool]$HaveNvSmi,
        [double]$PeakVramMb = $null
    )
    if ($PeakVramMb -ne $null) { return ("{0}" -f $PeakVramMb) + " MB" }
    if (-not $GpuLog)          { return "— (gpu-log off)" }
    if ($Ngl -le 0)            { return "— (NGL=0/CPU)" }
    if (-not $HaveNvSmi)       { return "— (nvidia-smi missing)" }
    return "—"
}

function Get-Perplexity([string]$file) {
    if (-not (Test-Path $file)) { return $null }
    $text = Get-Content -Path $file -Raw

    # Detect "insufficient tokens" and return a descriptive NA
    $need = [regex]::Match($text, '(?im)you need at least\s+(\d+)\s+tokens')
    $have = [regex]::Match($text, '(?im)tokenizes to only\s+(\d+)\s+tokens')
    if ($need.Success -and $have.Success) {
        $needN = $need.Groups[1].Value
        $haveN = $have.Groups[1].Value
        return "NA (insufficient tokens: $haveN/$needN)"
    }

    # 1) strong, specific patterns first
    $rxs = @(
        '(?im)\bperplexity\s*[:=]\s*(?<val>\d+(?:\.\d+)?)',
        '(?im)\bppl\s*[:=]\s*(?<val>\d+(?:\.\d+)?)',
        '(?im)\bppx\s*[:=]\s*(?<val>\d+(?:\.\d+)?)'
    )
    foreach ($rx in $rxs) {
        $m = [regex]::Matches($text, $rx)
        if ($m.Count -gt 0) {
            $last = $m[$m.Count-1].Groups['val'].Value
            if ($last -and ([double]$last) -gt 0) { return $last }
        }
    }
    # 2) fallback: look at lines mentioning ppl/perplexity and take the last sensible number
    $best = $null
    foreach ($ln in ($text -split "`r?`n")) {
        $low = $ln.ToLower()
        if ($low -match '\bperplexity\b' -or $low -match '(^|[^a-z])ppl([^a-z]|$)') {
            $nums = [regex]::Matches($ln, '\d+(?:\.\d+)?') | ForEach-Object { $_.Value }
            foreach ($n in $nums) {
                if ([double]$n -gt 0) { $best = $n } # keep last positive
            }
        }
    }
    return $best
}
# ---- discover tools ----
$benchPath = Get-ToolPath $BIN_DIR 'llama-bench'
if (-not $benchPath) { Write-Error "Error: $BIN_DIR\llama-bench(.exe) not found"; exit 3 }

$pplPath = Get-ToolPath $BIN_DIR 'llama-perplexity'
if (-not $pplPath) {
    Write-Host "Note: $BIN_DIR\llama-perplexity(.exe) not found; perplexity step will be skipped." -ForegroundColor Yellow
}

$haveNvSmi = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)

# ---- run directory ----
$ts = Get-Date -Format 'yyyyMMdd-HHmmss'
$RUN_DIR = Join-Path $RESULTS_DIR "xquant_$ts"
Ensure-Dir $RUN_DIR

Write-Host ">>> Writing logs to: $RUN_DIR"
Write-Host ">>> Model: $MODEL"
Write-Host ">>> CTX=$CTX P_LEN=$P_LEN N_TOK=$N_TOK BATCH=$BATCH THREADS=$THREADS NGL=$NGL"

# ---- prepare ppl file if needed ----
if (-not $PPL_FILE -and $MAKE_DUMMY_PPL) {
    $PPL_FILE = Join-Path $RUN_DIR 'wiki.test-60.raw'
    $lines = 1..60 | ForEach-Object { 'The quick brown fox jumps over the lazy dog.' }
    Set-Content -Path $PPL_FILE -Value $lines -Encoding ascii
    Write-Host ">>> Created dummy ppl file: $PPL_FILE"
}
if ($PPL_FILE -and -not (Test-Path $PPL_FILE)) {
    Write-Error "Error: ppl file not found: $PPL_FILE"; exit 4
}

# ---- flag auto-detection helpers ----
$benchHelp = (& $benchPath -h 2>&1 | Out-String)
$pplHelp   = if ($pplPath) { & $pplPath -h 2>&1 | Out-String } else { '' }

function Help-Has([string]$help, [string]$pattern) {
    return [bool]([regex]::Match(
        $help, $pattern,
        [System.Text.RegularExpressions.RegexOptions]::IgnoreCase
    ).Success)
}

$benchHasCtx    = Help-Has $benchHelp '(^|\s)-c,?\s*--ctx|--ctx'
$benchHasUb     = Help-Has $benchHelp '-ub,?\s*--ubatch-size|--ubatch'
$benchHasCTK    = Help-Has $benchHelp '-ctk,?\s*--cache-type-k|--cache-type-k'
$benchHasCTV    = Help-Has $benchHelp '-ctv,?\s*--cache-type-v|--cache-type-v'

$pplHasCtx      = Help-Has $pplHelp '(^|\s)-c,?\s*--ctx|--ctx'
$pplHasBatch    = Help-Has $pplHelp '(^|\s)-b,?\s*--batch|--batch-size'
$pplHasChunks   = Help-Has $pplHelp '--chunks'
$pplHasNGL      = Help-Has $pplHelp '-ngl'
$pplHasCTK      = Help-Has $pplHelp '-ctk'
$pplHasCTV      = Help-Has $pplHelp '-ctv'

# ---- process runner with optional VRAM sampling ----
# ---- process runner with optional VRAM sampling ----
function Invoke-LoggedProcess {
    param(
        [string]$Exe,
        [object[]]$Arguments,
        [string]$OutFile,
        [string]$TimeFile,
        [bool]$EnableVramSampling = $false
    )

    # Build single argument string (never null)
    $argString = Build-ArgumentString $Arguments

    # ---- START: PowerShell 5.1 Compatibility Fix ----
    # Create a temporary path for the stderr file and redirect to it.
    # This avoids the error in PS 5.1 while working perfectly in PS 7.
    $errFile = [System.IO.Path]::ChangeExtension($OutFile, '.stderr')

    $proc = Start-Process -FilePath $Exe -ArgumentList $argString -NoNewWindow -PassThru `
        -RedirectStandardOutput $OutFile -RedirectStandardError $errFile
    # ---- END: PowerShell 5.1 Compatibility Fix ----


    # Start VRAM sampling job if requested
    $vramJob = $null
    if ($EnableVramSampling) {
        $vramLog = [System.IO.Path]::ChangeExtension($OutFile, '.vram')
        $childPid = $proc.Id
        $vramJob = Start-Job -ScriptBlock {
            param([int]$ChildPid, [string]$VramLog)
            try {
                while (Get-Process -Id $ChildPid -ErrorAction SilentlyContinue) {
                    $o = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
                    if ($LASTEXITCODE -eq 0 -and $o) {
                        $o | Out-File -FilePath $VramLog -Append -Encoding ascii
                    }
                    Start-Sleep -Milliseconds 500
                }
            } catch { }
        } -ArgumentList $childPid, $vramLog
    }

    # Start RSS sampling job (Windows-friendly, robust after exit)
    $rssJob = $null
    try {
        $childPid = $proc.Id
        $rssJob = Start-Job -ScriptBlock {
            param([int]$ChildPid)
            $max = [long]0
            try {
                while ($true) {
                    $p = Get-Process -Id $ChildPid -ErrorAction SilentlyContinue
                    if (-not $p) { break }
                    if ($p.WorkingSet64 -gt $max) { $max = [long]$p.WorkingSet64 }
                    Start-Sleep -Milliseconds 200
                }
            } catch { }
            return $max
        } -ArgumentList $childPid
    } catch { }

    # Inline RSS sampler: track WorkingSet64 until the process exits (robust on PS7)
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $maxRss = [long]0
    while ($true) {
        try {
            $proc.Refresh()
            if ($proc.WorkingSet64 -gt $maxRss) { $maxRss = [long]$proc.WorkingSet64 }
            if ($proc.HasExited) { break }
        } catch { break }
        Start-Sleep -Milliseconds 100
    }
    $sw.Stop()

    # ---- START: PowerShell 5.1 Compatibility Fix ----
    # Merge the temporary stderr file into the main log and then clean it up.
    if (Test-Path $errFile) {
        Add-Content -Path $OutFile -Value (Get-Content -Path $errFile -Raw)
        Remove-Item -Path $errFile -Force
    }
    # ---- END: PowerShell 5.1 Compatibility Fix ----


    # finalize VRAM job
    if ($vramJob) {
        try { Receive-Job -Job $vramJob -ErrorAction SilentlyContinue | Out-Null } catch { }
        Remove-Job -Job $vramJob -Force -ErrorAction SilentlyContinue
    }

    # Write peak RSS we sampled
    Write-TimeLog -path $TimeFile -secs $sw.Elapsed.TotalSeconds -maxrss $maxRss
}

# ---- bench mode ----
function Run-BenchMode {
    param([ValidateSet('off','on')] [string]$Mode)

    $out  = Join-Path $RUN_DIR "bench_$Mode.out"
    $time = Join-Path $RUN_DIR "bench_$Mode.time"

    if ($Mode -eq 'on') { $env:LLAMA_XQUANT = '1' } else { Remove-Item Env:LLAMA_XQUANT -ErrorAction SilentlyContinue }
    if ($Mode -eq 'on') { $env:LLAMA_XQ_NOBASE = '1' } else { Remove-Item Env:LLAMA_XQ_NOBASE -ErrorAction SilentlyContinue }
    Write-Host ">>> BENCH ($Mode) ..."

    # Arguments only (exe is passed separately)
    $benchArgs = @('-m', $MODEL, '-p', $P_LEN, '-n', $N_TOK, '-b', $BATCH, '-t', $THREADS, '-ngl', $NGL)
    if ($benchHasCtx) { $benchArgs += @('-c', $CTX) }
    if ($UBATCH -and $benchHasUb) { $benchArgs += @('-ub', $UBATCH) }
    if ($CTK -and $benchHasCTK) { $benchArgs += @('-ctk', $CTK) }
    if ($CTV -and $benchHasCTV) { $benchArgs += @('-ctv', $CTV) }

    if ($DEBUG_MODE) {
        Write-Host "[debug] bench cmd: $benchPath $(Build-ArgumentString $benchArgs)" -ForegroundColor DarkGray
    }

    $enableVram = ($GPU_LOG -and $NGL -gt 0 -and $haveNvSmi)
    # Corrected line:
    Invoke-LoggedProcess -Exe $benchPath -Arguments $benchArgs -OutFile $out -TimeFile $time -EnableVramSampling:$enableVram
}


# ---- ppl mode ----
function Run-PplMode {
    param([ValidateSet('off','on')] [string]$Mode)

    if (-not $pplPath -or -not $PPL_FILE) {
        Write-Host ">>> PPL ($Mode) skipped."
        return
    }
    if ($Mode -eq 'on') { $env:LLAMA_XQUANT = '1' } else { Remove-Item Env:LLAMA_XQUANT -ErrorAction SilentlyContinue }
    if ($Mode -eq 'on') { $env:LLAMA_XQ_NOBASE = '1' } else { Remove-Item Env:LLAMA_XQ_NOBASE -ErrorAction SilentlyContinue }
    Write-Host ">>> PPL ($Mode) ..."

    $pplArgs = @('-m', $MODEL, '-f', $PPL_FILE)
    if ($pplHasCTK -and $CTK) { $pplArgs += @('-ctk', $CTK) }
    if ($pplHasCTV -and $CTV) { $pplArgs += @('-ctv', $CTV) }
    if ($pplHasCtx)           { $pplArgs += @('-c', $CTX) }
    if ($pplHasBatch)         { $pplArgs += @('-b', $BATCH) }
    if ($pplHasChunks)        { $pplArgs += @('--chunks', 1) }
    if ($pplHasNGL)           { $pplArgs += @('-ngl', $NGL) }

    if ($DEBUG_MODE) {
        Write-Host "[debug] ppl cmd: $pplPath $(Build-ArgumentString $pplArgs)" -ForegroundColor DarkGray
    }

    $out = Join-Path $RUN_DIR "ppl_$Mode.out"
    $argString = Build-ArgumentString $pplArgs

    # ---- START: FINAL FIX ----
    $errFile = [System.IO.Path]::ChangeExtension($out, '.stderr')
    $proc = Start-Process -FilePath $pplPath -ArgumentList $argString -NoNewWindow `
        -RedirectStandardOutput $out -RedirectStandardError $errFile -PassThru
    Wait-Process -Id $proc.Id

    if (Test-Path $errFile) {
        Add-Content -Path $out -Value (Get-Content -Path $errFile -Raw)
        Remove-Item -Path $errFile -Force
    }
    # ---- END: FINAL FIX ----
}



# ---- RUN ----
Run-BenchMode off
Run-BenchMode on

Run-PplMode off
Run-PplMode on

# ---- PARSE ----
$benchOffOut  = Join-Path $RUN_DIR 'bench_off.out'
$benchOnOut   = Join-Path $RUN_DIR 'bench_on.out'
$benchOffTime = Join-Path $RUN_DIR 'bench_off.time'
$benchOnTime  = Join-Path $RUN_DIR 'bench_on.time'

$tokOff = Get-LastTokps $benchOffOut
$tokOn  = Get-LastTokps  $benchOnOut

if (-not $tokOff) {
    $secOff = Get-ElapsedFromTimeLog $benchOffTime
    if ($secOff) { $tokOff = Compute-TokpsFromSeconds $N_TOK $secOff }
}
if (-not $tokOn) {
    $secOn = Get-ElapsedFromTimeLog $benchOnTime
    if ($secOn) { $tokOn = Compute-TokpsFromSeconds $N_TOK $secOn }
}

$speedup = 'NA'
if ($tokOff -and $tokOn -and $tokOff -gt 0) {
    $speedup = '{0:N2}' -f ($tokOn / $tokOff)
}

$rssOff = Get-MaxRssFromTimeLog $benchOffTime
$rssOn  = Get-MaxRssFromTimeLog $benchOnTime

# GPU VRAM peak (MB) + reasoned summary
$offPeak = Get-PeakVramFromLog ([System.IO.Path]::ChangeExtension($benchOffOut, '.vram'))
$onPeak  = Get-PeakVramFromLog  ([System.IO.Path]::ChangeExtension($benchOnOut,  '.vram'))
$vramOff = Get-VramSummaryString -GpuLog:$GPU_LOG -Ngl:$NGL -HaveNvSmi:$haveNvSmi -PeakVramMb:$offPeak
$vramOn  = Get-VramSummaryString -GpuLog:$GPU_LOG -Ngl:$NGL -HaveNvSmi:$haveNvSmi -PeakVramMb:$onPeak

# PPL (if present)
$pplOff = 'NA'; $pplOn = 'NA'
$pplOffFile = Join-Path $RUN_DIR 'ppl_off.out'
$pplOnFile  = Join-Path $RUN_DIR 'ppl_on.out'
if (Test-Path $pplOffFile) { $v = Get-Perplexity $pplOffFile; if ($v) { $pplOff = $v } }
if (Test-Path $pplOnFile)  { $v = Get-Perplexity $pplOnFile ; if ($v) { $pplOn  = $v } }

# ---- SUMMARY ----
Write-Host ""
Write-Host "==================== XQuant A/B Summary ===================="
Write-Host ("Model     : {0}" -f $MODEL)
Write-Host ("Params    : CTX={0} P_LEN={1} N_TOK={2} BATCH={3} THREADS={4} NGL={5}" -f $CTX,$P_LEN,$N_TOK,$BATCH,$THREADS,$NGL)
Write-Host ("Bench OFF : {0} tok/s | MaxRSS={1} | VRAM={2}" -f ($(if($tokOff){$tokOff}else{'NA'}), (Format-RssMb $rssOff), $vramOff))
Write-Host ("Bench ON  : {0} tok/s | MaxRSS={1} | VRAM={2}" -f ($(if($tokOn){$tokOn}else{'NA'}),  (Format-RssMb $rssOn),  $vramOn))
Write-Host ("Speedup   : {0} x" -f $speedup)
Write-Host ("PPL OFF   : {0}" -f $pplOff)
Write-Host ("PPL ON    : {0}" -f $pplOn)
Write-Host ("Logs      : {0}" -f $RUN_DIR)
Write-Host "============================================================"

# ---- CSV ----
$CSV = Join-Path $RUN_DIR 'summary.csv'
"model,ctx,p_len,n_tok,batch,threads,ngl,tok_off,tok_on,speedup,maxrss_off_mb,maxrss_on_mb,vram_off_mb,vram_on_mb,ppl_off,ppl_on" | Set-Content -Path $CSV -Encoding ascii

function _ToMbOrNA([object]$bytes) {
    if ($null -eq $bytes -or $bytes -eq '') { return 'NA' }
    try { return [math]::Round(([double]$bytes / 1MB), 1) } catch { return 'NA' }
}

$csvLine = '{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},"{12}","{13}",{14},{15}' -f `
    $MODEL,$CTX,$P_LEN,$N_TOK,$BATCH,$THREADS,$NGL, `
    ($(if($tokOff){$tokOff}else{'NA'})),($(if($tokOn){$tokOn}else{'NA'})),$speedup, `
    (_ToMbOrNA $rssOff),(_ToMbOrNA $rssOn), `
    ($vramOff -replace '\s*MB$',''),($vramOn -replace '\s*MB$',''),$pplOff,$pplOn

Add-Content -Path $CSV -Value $csvLine -Encoding ascii
Write-Host "CSV       : $CSV"
