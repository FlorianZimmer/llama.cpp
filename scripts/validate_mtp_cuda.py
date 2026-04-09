#!/usr/bin/env python3

import argparse
import concurrent.futures
import json
import signal
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

    log_path = log_dir / f"{mode}-np{n_parallel}.log"
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
    print(f"{res.mode} np={res.n_parallel}")
    for idx, resp in enumerate(res.responses):
        print(f"  req{idx}: {describe_response(resp)}")
        print(f"  req{idx}: content={resp['content']!r}")
    print(f"  log={res.log_path}")


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
        ("baseline", 1, args.port_base + 0),
        ("mtp", 1, args.port_base + 1),
        ("baseline", 2, args.port_base + 2),
        ("mtp", 2, args.port_base + 3),
    ]

    results: dict[tuple[str, int], ScenarioResult] = {}

    for mode, n_parallel, port in scenarios:
        print(f"\n== running {mode} np={n_parallel} on port {port} ==")
        results[(mode, n_parallel)] = run_scenario(
            binary=binary,
            model=model,
            host=args.host,
            port=port,
            mode=mode,
            n_parallel=n_parallel,
            args=args,
            log_dir=log_dir,
        )
        print_result(results[(mode, n_parallel)])

    print("\n== validating exact greedy equality ==")
    assert_equal_outputs(results[("baseline", 1)], results[("mtp", 1)])
    assert_equal_outputs(results[("baseline", 2)], results[("mtp", 2)])

    for n_parallel in (1, 2):
        mtp = results[("mtp", n_parallel)]
        if not any(resp.get("timings", {}).get("draft_n", 0) > 0 for resp in mtp.responses):
            raise AssertionError(f"mtp np={n_parallel} did not report any native draft activity")

    baseline_output = results[("baseline", 1)].outputs[0]
    for key, res in results.items():
        for out in res.outputs:
            if out != baseline_output:
                raise AssertionError(f"scenario {key} output {out!r} does not match baseline np=1 output {baseline_output!r}")

    print("all greedy outputs match exactly")

    print("\n== summary ==")
    for n_parallel in (1, 2):
        baseline = results[("baseline", n_parallel)]
        mtp = results[("mtp", n_parallel)]
        baseline_tps = [resp["timings"]["predicted_per_second"] for resp in baseline.responses]
        mtp_tps = [resp["timings"]["predicted_per_second"] for resp in mtp.responses]
        print(
            f"np={n_parallel}: baseline tok/s={baseline_tps}, "
            f"mtp tok/s={mtp_tps}"
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
