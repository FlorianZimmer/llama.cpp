#!/usr/bin/env python3

import argparse
import concurrent.futures
import json
import signal
import statistics
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = "Hello"
DEFAULT_SEED = 12345
DEFAULT_N_PREDICT = 8


@dataclass
class ScenarioResult:
    repeat: int
    mode: str
    n_parallel: int
    outputs: list[str]
    responses: list[dict[str, Any]]
    log_path: Path


def post_json(url: str, payload: dict[str, Any], timeout: float = 60.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(url: str, timeout: float = 5.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_for_server(base_url: str, proc: subprocess.Popen[str], timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error = "server did not become healthy"

    while time.time() < deadline:
        ret = proc.poll()
        if ret is not None:
            raise RuntimeError(f"server exited early with code {ret}")

        try:
            get_json(f"{base_url}/health", timeout=2.0)
            return
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            time.sleep(0.25)

    raise RuntimeError(last_error)


def stop_server(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return

    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def completion_payload(prompt: str, n_predict: int, seed: int) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "seed": seed,
        "cache_prompt": False,
    }


def format_toks(resp: dict[str, Any]) -> str:
    timings = resp.get("timings", {})
    predicted = timings.get("predicted_per_second")
    if predicted is None:
        return "n/a"
    return f"{predicted:.2f}"


def describe_response(resp: dict[str, Any]) -> str:
    timings = resp.get("timings", {})
    extra = []
    extra.append(f"tok/s={format_toks(resp)}")
    if "draft_n" in timings and "draft_n_accepted" in timings:
        extra.append(f"draft={timings['draft_n']}")
        extra.append(f"accepted={timings['draft_n_accepted']}")
    extra.append(f"slot={resp.get('id_slot', '?')}")
    return ", ".join(extra)


def summarize_predicted_per_second(responses: list[dict[str, Any]]) -> dict[str, float | list[float]]:
    values = [float(resp["timings"]["predicted_per_second"]) for resp in responses]
    summary: dict[str, float | list[float]] = {
        "values": values,
        "mean": statistics.fmean(values) if values else 0.0,
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }
    if len(values) > 1:
        summary["stddev"] = statistics.pstdev(values)
    else:
        summary["stddev"] = 0.0
    return summary


def launch_server(
    binary: Path,
    model: Path,
    host: str,
    port: int,
    n_parallel: int,
    mode: str,
    args: argparse.Namespace,
    log_dir: Path,
) -> tuple[subprocess.Popen[str], Path]:
    cmd = [
        str(binary),
        "-m",
        str(model),
        "-ngl",
        args.ngl,
        "-fa",
        args.flash_attn,
        "-c",
        str(args.ctx_size),
        "-b",
        str(args.batch_size),
        "-ub",
        str(args.ubatch_size),
        "-t",
        str(args.threads),
        "-tb",
        str(args.threads_batch),
        "--host",
        host,
        "--port",
        str(port),
        "--no-webui",
        "--no-warmup",
        "--perf",
        "-np",
        str(n_parallel),
    ]

    if mode == "mtp":
        cmd.extend(["--spec-type", "mtp", "--draft-max", str(args.draft_max)])

    log_path = log_dir / f"{mode}-np{n_parallel}-r{args._repeat_idx}.log"
    fout = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=fout,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc, log_path


def run_requests(base_url: str, prompt: str, n_predict: int, seed: int, n_parallel: int) -> list[dict[str, Any]]:
    payloads = [
        completion_payload(prompt=prompt, n_predict=n_predict, seed=seed + i)
        for i in range(n_parallel)
    ]

    def do_request(payload: dict[str, Any]) -> dict[str, Any]:
        return post_json(f"{base_url}/completion", payload, timeout=120.0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel) as tp:
        futures = [tp.submit(do_request, payload) for payload in payloads]
        return [f.result() for f in futures]


def run_scenario(
    binary: Path,
    model: Path,
    host: str,
    port: int,
    mode: str,
    n_parallel: int,
    repeat: int,
    args: argparse.Namespace,
    log_dir: Path,
) -> ScenarioResult:
    proc, log_path = launch_server(binary, model, host, port, n_parallel, mode, args, log_dir)
    base_url = f"http://{host}:{port}"

    try:
        wait_for_server(base_url, proc, timeout_s=args.startup_timeout)
        responses = run_requests(base_url, args.prompt, args.n_predict, args.seed, n_parallel)
    finally:
        stop_server(proc)

    outputs = [resp["content"] for resp in responses]
    return ScenarioResult(
        repeat=repeat,
        mode=mode,
        n_parallel=n_parallel,
        outputs=outputs,
        responses=responses,
        log_path=log_path,
    )


def assert_equal_outputs(lhs: ScenarioResult, rhs: ScenarioResult) -> None:
    if lhs.outputs != rhs.outputs:
        raise AssertionError(
            f"output mismatch for np={lhs.n_parallel}: {lhs.mode}={lhs.outputs!r} vs {rhs.mode}={rhs.outputs!r}"
        )


def print_result(res: ScenarioResult) -> None:
    print(f"{res.mode} np={res.n_parallel} repeat={res.repeat}")
    for idx, resp in enumerate(res.responses):
        print(f"  req{idx}: {describe_response(resp)}")
        print(f"  req{idx}: content={resp['content']!r}")
    print(f"  log={res.log_path}")


def summarize_runs(runs: list[ScenarioResult]) -> dict[str, Any]:
    per_request_values: list[list[float]] = []
    for run in runs:
        timings = [float(resp["timings"]["predicted_per_second"]) for resp in run.responses]
        if not per_request_values:
            per_request_values = [[] for _ in timings]
        for idx, value in enumerate(timings):
            per_request_values[idx].append(value)

    request_summaries = [
        {
            "request_index": idx,
            "mean": statistics.fmean(values) if values else 0.0,
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
            "stddev": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
        for idx, values in enumerate(per_request_values)
    ]

    run_means = [
        statistics.fmean([float(resp["timings"]["predicted_per_second"]) for resp in run.responses])
        for run in runs
    ]

    return {
        "repeats": len(runs),
        "run_mean_predicted_per_second": {
            "mean": statistics.fmean(run_means) if run_means else 0.0,
            "min": min(run_means) if run_means else 0.0,
            "max": max(run_means) if run_means else 0.0,
            "stddev": statistics.pstdev(run_means) if len(run_means) > 1 else 0.0,
            "values": run_means,
        },
        "per_request": request_summaries,
    }


def write_json_report(path: Path, args: argparse.Namespace, results: dict[tuple[str, int], list[ScenarioResult]]) -> None:
    scenarios: list[dict[str, Any]] = []
    for (mode, n_parallel), runs in sorted(results.items()):
        scenarios.append(
            {
                "mode": mode,
                "n_parallel": n_parallel,
                "summary": summarize_runs(runs),
                "runs": [
                    {
                        "repeat": run.repeat,
                        "outputs": run.outputs,
                        "timings_summary": summarize_predicted_per_second(run.responses),
                        "responses": run.responses,
                        "log_path": str(run.log_path),
                    }
                    for run in runs
                ],
            }
        )

    payload = {
        "binary": str(Path(args.binary).resolve()),
        "model": str(Path(args.model).resolve()),
        "prompt": args.prompt,
        "seed": args.seed,
        "n_predict": args.n_predict,
        "repeat": args.repeat,
        "allow_known_np2_divergence": args.allow_known_np2_divergence,
        "scenarios": scenarios,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate CUDA native MTP against baseline llama-server output and throughput.")
    parser.add_argument("--binary", default="build-cuda/bin/llama-server", help="Path to llama-server")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port-base", type=int, default=18100, help="Base port for spawned servers")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt used for greedy validation")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base seed")
    parser.add_argument("--n-predict", type=int, default=DEFAULT_N_PREDICT, help="Generated tokens per request")
    parser.add_argument("--ctx-size", type=int, default=4096, help="Context size")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--ubatch-size", type=int, default=64, help="Ubatch size")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads")
    parser.add_argument("--threads-batch", type=int, default=8, help="CPU batch threads")
    parser.add_argument("--ngl", default="all", help="Value for -ngl")
    parser.add_argument("--flash-attn", default="on", choices=["on", "off", "auto"], help="Value for -fa")
    parser.add_argument("--startup-timeout", type=float, default=180.0, help="Seconds to wait for server health")
    parser.add_argument("--draft-max", type=int, default=1, help="Value for --draft-max in mtp mode")
    parser.add_argument("--log-dir", default=None, help="Directory for per-scenario logs")
    parser.add_argument("--repeat", type=int, default=1, help="How many times to rerun each scenario")
    parser.add_argument("--json-out", default=None, help="Optional path for a machine-readable JSON report")
    parser.add_argument(
        "--allow-known-np2-divergence",
        action="store_true",
        help="Allow the known native-MTP np>1 exactness limitation while still requiring valid responses",
    )
    args = parser.parse_args()

    binary = Path(args.binary).resolve()
    model = Path(args.model).resolve()
    if not binary.is_file():
        raise FileNotFoundError(f"llama-server not found: {binary}")
    if not model.is_file():
        raise FileNotFoundError(f"model not found: {model}")

    log_dir = Path(args.log_dir).resolve() if args.log_dir else Path(tempfile.mkdtemp(prefix="mtp-cuda-validate-"))
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"binary={binary}")
    print(f"model={model}")
    print(f"logs={log_dir}")

    scenarios = [
        ("baseline", 1),
        ("mtp", 1),
        ("baseline", 2),
        ("mtp", 2),
    ]

    results: dict[tuple[str, int], list[ScenarioResult]] = {}

    for repeat in range(args.repeat):
        args._repeat_idx = repeat
        for scenario_idx, (mode, n_parallel) in enumerate(scenarios):
            port = args.port_base + repeat * 16 + scenario_idx
            print(f"\n== running {mode} np={n_parallel} repeat={repeat + 1}/{args.repeat} on port {port} ==")
            result = run_scenario(
                binary=binary,
                model=model,
                host=args.host,
                port=port,
                mode=mode,
                n_parallel=n_parallel,
                repeat=repeat,
                args=args,
                log_dir=log_dir,
            )
            print_result(result)
            results.setdefault((mode, n_parallel), []).append(result)

    print("\n== validating exact greedy equality ==")
    baseline_np1_runs = results[("baseline", 1)]
    mtp_np1_runs = results[("mtp", 1)]
    baseline_np2_runs = results[("baseline", 2)]
    mtp_np2_runs = results[("mtp", 2)]

    for repeat, (baseline_run, mtp_run) in enumerate(zip(baseline_np1_runs, mtp_np1_runs, strict=True)):
        assert_equal_outputs(baseline_run, mtp_run)
        if not args.allow_known_np2_divergence:
            assert_equal_outputs(baseline_np2_runs[repeat], mtp_np2_runs[repeat])

    for n_parallel in (1, 2):
        for mtp_run in results[("mtp", n_parallel)]:
            if not any(resp.get("timings", {}).get("draft_n", 0) > 0 for resp in mtp_run.responses):
                raise AssertionError(f"mtp np={n_parallel} repeat={mtp_run.repeat} did not report any native draft activity")

    for repeat, baseline_run in enumerate(baseline_np1_runs):
        baseline_output = baseline_run.outputs[0]
        for key, scenario_runs in results.items():
            mode, n_parallel = key
            if args.allow_known_np2_divergence and mode == "mtp" and n_parallel == 2:
                continue
            for run in scenario_runs:
                if run.repeat != repeat:
                    continue
                for out in run.outputs:
                    if out != baseline_output:
                        raise AssertionError(
                            f"scenario {key} repeat={repeat} output {out!r} does not match baseline np=1 output {baseline_output!r}"
                        )

    if args.allow_known_np2_divergence:
        print("all required greedy outputs match exactly; mtp np=2 divergence was allowed for this run")
    else:
        print("all greedy outputs match exactly")

    print("\n== summary ==")
    for n_parallel in (1, 2):
        baseline_summary = summarize_runs(results[("baseline", n_parallel)])
        mtp_summary = summarize_runs(results[("mtp", n_parallel)])
        print(
            f"np={n_parallel}: baseline run-mean tok/s={baseline_summary['run_mean_predicted_per_second']['values']}, "
            f"mtp run-mean tok/s={mtp_summary['run_mean_predicted_per_second']['values']}"
        )

    if args.json_out:
        write_json_report(Path(args.json_out).resolve(), args, results)
        print(f"json={Path(args.json_out).resolve()}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
