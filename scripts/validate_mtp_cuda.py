#!/usr/bin/env python3

import argparse
import concurrent.futures
import json
import os
import re
import signal
import statistics
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = "Hello"
DEFAULT_SEED = 12345
DEFAULT_N_PREDICT = 8

PROFILE_RE = re.compile(
    r"native MTP profile:"
    r" draft=\s*(?P<draft_ms>[\d.]+) ms \((?P<draft_calls>\d+) calls\),"
    r" snapshot=\s*(?P<snapshot_ms>[\d.]+) ms \((?P<snapshot_calls>\d+) saves\),"
    r" accept=\s*(?P<accept_ms>[\d.]+) ms \((?P<accept_calls>\d+) accepts\),"
    r" restore=\s*(?P<restore_ms>[\d.]+) ms \((?P<restore_calls>\d+) restores\),"
    r" replay=\s*(?P<replay_ms>[\d.]+) ms \((?P<replay_calls>\d+) replays\),"
    r" total=\s*(?P<total_ms>[\d.]+) ms"
)

ACCEPTANCE_RE = re.compile(
    r"draft acceptance rate =\s*(?P<ratio>[\d.]+)\s*\(\s*(?P<accepted>\d+)\s+accepted /\s*(?P<generated>\d+)\s+generated\)"
)

STEP_RE = re.compile(
    r"native MTP step:"
    r" step=(?P<step>\d+)"
    r" drafted=(?P<drafted>\d+)"
    r" accepted=(?P<accepted>\d+)"
    r" replay=(?P<replay>[01])"
    r" fast=(?P<fast>[01])"
    r" logits_suppressed=(?P<logits_suppressed>[01])"
    r" forced_plain=(?P<forced_plain>[01])"
    r" cooldown=(?P<cooldown>[01])"
    r" guard=(?P<guard>[01])"
    r" draft=(?P<draft_us>\d+)\s+us"
    r" snapshot=(?P<snapshot_us>\d+)\s+us"
    r" accept=(?P<accept_us>\d+)\s+us"
    r" restore=(?P<restore_us>\d+)\s+us"
    r" replay_us=(?P<replay_us>\d+)"
    r" total=(?P<total_us>\d+)\s+us"
)

CASE_PRESETS: dict[str, dict[str, Any]] = {
    "primary": {
        "prompt": "Write one short sentence about Berlin.",
        "seed": 42,
        "n_predict": 12,
        "description": "Primary exact CUDA gate for the 9B UD-Q4_K_XL regression case.",
    },
    "good": {
        "prompt": "Write two short sentences about the Moon.",
        "seed": 31415,
        "n_predict": 64,
        "description": "Previously good exact/stable CUDA case.",
    },
    "bad": {
        "prompt": "List three reasons Rust is used for systems programming.",
        "seed": 777,
        "n_predict": 64,
        "description": "Replay-heavy stability case.",
    },
}


@dataclass(frozen=True)
class CaseConfig:
    name: str
    prompt: str
    seed: int
    n_predict: int
    description: str


@dataclass(frozen=True)
class ScenarioKey:
    case: str
    repeat: int
    mode: str
    n_parallel: int


@dataclass
class ProfileTotals:
    draft_ms: float = 0.0
    draft_calls: int = 0
    snapshot_ms: float = 0.0
    snapshot_calls: int = 0
    accept_ms: float = 0.0
    accept_calls: int = 0
    restore_ms: float = 0.0
    restore_calls: int = 0
    replay_ms: float = 0.0
    replay_calls: int = 0
    total_ms: float = 0.0
    acceptance_generated: int = 0
    acceptance_accepted: int = 0
    steps: list[dict[str, int]] = field(default_factory=list)

    def add_match(self, match: re.Match[str]) -> None:
        self.draft_ms += float(match.group("draft_ms"))
        self.draft_calls += int(match.group("draft_calls"))
        self.snapshot_ms += float(match.group("snapshot_ms"))
        self.snapshot_calls += int(match.group("snapshot_calls"))
        self.accept_ms += float(match.group("accept_ms"))
        self.accept_calls += int(match.group("accept_calls"))
        self.restore_ms += float(match.group("restore_ms"))
        self.restore_calls += int(match.group("restore_calls"))
        self.replay_ms += float(match.group("replay_ms"))
        self.replay_calls += int(match.group("replay_calls"))
        self.total_ms += float(match.group("total_ms"))

    def add_acceptance(self, match: re.Match[str]) -> None:
        self.acceptance_accepted += int(match.group("accepted"))
        self.acceptance_generated += int(match.group("generated"))

    def add_step(self, match: re.Match[str]) -> None:
        self.steps.append(
            {
                "step": int(match.group("step")),
                "drafted": int(match.group("drafted")),
                "accepted": int(match.group("accepted")),
                "replay": int(match.group("replay")),
                "fast": int(match.group("fast")),
                "logits_suppressed": int(match.group("logits_suppressed")),
                "forced_plain": int(match.group("forced_plain")),
                "cooldown": int(match.group("cooldown")),
                "guard": int(match.group("guard")),
                "draft_us": int(match.group("draft_us")),
                "snapshot_us": int(match.group("snapshot_us")),
                "accept_us": int(match.group("accept_us")),
                "restore_us": int(match.group("restore_us")),
                "replay_us": int(match.group("replay_us")),
                "total_us": int(match.group("total_us")),
            }
        )

    def per_call(self, total: float, calls: int) -> float | None:
        return None if calls == 0 else total / calls

    def to_dict(self) -> dict[str, Any]:
        acceptance_rate = None
        if self.acceptance_generated > 0:
            acceptance_rate = self.acceptance_accepted / self.acceptance_generated

        speculative_steps = [step for step in self.steps if step["forced_plain"] == 0]
        step_count = len(speculative_steps)
        step_drafted = sum(step["drafted"] for step in speculative_steps)
        step_accepted = sum(step["accepted"] for step in speculative_steps)
        step_acceptance_rate = None if step_drafted == 0 else step_accepted / step_drafted
        step_totals_us = {
            "draft": sum(step["draft_us"] for step in speculative_steps),
            "snapshot": sum(step["snapshot_us"] for step in speculative_steps),
            "accept": sum(step["accept_us"] for step in speculative_steps),
            "restore": sum(step["restore_us"] for step in speculative_steps),
            "replay": sum(step["replay_us"] for step in speculative_steps),
            "total": sum(step["total_us"] for step in speculative_steps),
        }
        step_mean_us = {
            key: None if step_count == 0 else value / step_count
            for key, value in step_totals_us.items()
        }

        return {
            "totals_ms": {
                "draft": self.draft_ms,
                "snapshot": self.snapshot_ms,
                "accept": self.accept_ms,
                "restore": self.restore_ms,
                "replay": self.replay_ms,
                "total": self.total_ms,
            },
            "calls": {
                "draft": self.draft_calls,
                "snapshot": self.snapshot_calls,
                "accept": self.accept_calls,
                "restore": self.restore_calls,
                "replay": self.replay_calls,
            },
            "per_call_ms": {
                "draft": self.per_call(self.draft_ms, self.draft_calls),
                "snapshot": self.per_call(self.snapshot_ms, self.snapshot_calls),
                "accept": self.per_call(self.accept_ms, self.accept_calls),
                "restore": self.per_call(self.restore_ms, self.restore_calls),
                "replay": self.per_call(self.replay_ms, self.replay_calls),
            },
            "acceptance": {
                "accepted": self.acceptance_accepted,
                "generated": self.acceptance_generated,
                "rate": acceptance_rate,
            },
            "step_summary": {
                "count": step_count,
                "drafted": step_drafted,
                "accepted": step_accepted,
                "replay_steps": sum(step["replay"] for step in speculative_steps),
                "acceptance_rate": step_acceptance_rate,
                "totals_us": step_totals_us,
                "mean_us": step_mean_us,
            },
            "step_visibility": {
                "all_steps": len(self.steps),
                "speculative_steps": step_count,
                "pure_fast_path_steps": sum(step["fast"] for step in speculative_steps),
                "logits_suppressed_steps": sum(step["logits_suppressed"] for step in speculative_steps),
                "forced_plain_steps": sum(step["forced_plain"] for step in self.steps),
                "cooldown_hits": sum(step["cooldown"] for step in self.steps),
                "guard_hits": sum(step["guard"] for step in self.steps),
            },
            "steps": self.steps,
        }


@dataclass
class ScenarioResult:
    key: ScenarioKey
    outputs: list[str]
    responses: list[dict[str, Any]]
    log_path: Path
    predicted_per_second: list[float]
    profile: ProfileTotals | None

    def mean_tok_s(self) -> float:
        return statistics.fmean(self.predicted_per_second)


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


def parse_case_names(text: str) -> list[str]:
    names = [item.strip() for item in text.split(",") if item.strip()]
    if not names:
        raise ValueError("at least one case must be provided")
    for name in names:
        if name != "custom" and name not in CASE_PRESETS:
            raise ValueError(f"unknown case: {name}")
    return names


def parse_n_parallels(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one n_parallel must be provided")
    return values


def get_case_config(name: str, args: argparse.Namespace) -> CaseConfig:
    if name == "custom":
        return CaseConfig(
            name="custom",
            prompt=args.prompt,
            seed=args.seed,
            n_predict=args.n_predict,
            description="Custom single-case run.",
        )

    preset = CASE_PRESETS[name]
    return CaseConfig(
        name=name,
        prompt=preset["prompt"],
        seed=preset["seed"],
        n_predict=preset["n_predict"],
        description=preset["description"],
    )


def parse_profile(log_path: Path) -> ProfileTotals | None:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    profile = ProfileTotals()
    matched = False

    for match in PROFILE_RE.finditer(text):
        profile.add_match(match)
        matched = True

    for match in ACCEPTANCE_RE.finditer(text):
        profile.add_acceptance(match)

    for match in STEP_RE.finditer(text):
        profile.add_step(match)

    return profile if matched else None


def launch_server(
    binary: Path,
    model: Path,
    host: str,
    port: int,
    n_parallel: int,
    mode: str,
    args: argparse.Namespace,
    log_dir: Path,
    log_name: str,
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

    log_path = log_dir / f"{log_name}.log"
    fout = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    if args.mtp_profile:
        env["LLAMA_SERVER_MTP_PROFILE"] = "1"
    proc = subprocess.Popen(
        cmd,
        stdout=fout,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return proc, log_path


def run_requests(case: CaseConfig, base_url: str, n_parallel: int) -> list[dict[str, Any]]:
    payloads = [
        completion_payload(prompt=case.prompt, n_predict=case.n_predict, seed=case.seed + i)
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
    case: CaseConfig,
    mode: str,
    n_parallel: int,
    repeat: int,
    args: argparse.Namespace,
    log_dir: Path,
) -> ScenarioResult:
    key = ScenarioKey(case=case.name, repeat=repeat, mode=mode, n_parallel=n_parallel)
    log_name = f"{case.name}-r{repeat}-{mode}-np{n_parallel}"
    proc, log_path = launch_server(binary, model, host, port, n_parallel, mode, args, log_dir, log_name)
    base_url = f"http://{host}:{port}"

    try:
        wait_for_server(base_url, proc, timeout_s=args.startup_timeout)
        responses = run_requests(case, base_url, n_parallel)
    finally:
        stop_server(proc)

    outputs = [resp["content"] for resp in responses]
    predicted_per_second = [float(resp["timings"]["predicted_per_second"]) for resp in responses]
    profile = parse_profile(log_path) if args.mtp_profile else None

    return ScenarioResult(
        key=key,
        outputs=outputs,
        responses=responses,
        log_path=log_path,
        predicted_per_second=predicted_per_second,
        profile=profile,
    )


def assert_equal_outputs(lhs: ScenarioResult, rhs: ScenarioResult) -> None:
    if lhs.outputs != rhs.outputs:
        raise AssertionError(
            f"output mismatch for case={lhs.key.case} repeat={lhs.key.repeat} np={lhs.key.n_parallel}: "
            f"{lhs.key.mode}={lhs.outputs!r} vs {rhs.key.mode}={rhs.outputs!r}"
        )


def require_stable_outputs(res: ScenarioResult) -> None:
    for idx, out in enumerate(res.outputs):
        if not out:
            raise AssertionError(
                f"scenario case={res.key.case} repeat={res.key.repeat} mode={res.key.mode} np={res.key.n_parallel} "
                f"returned empty output for req{idx}"
            )


def print_result(res: ScenarioResult) -> None:
    print(f"{res.key.case} repeat={res.key.repeat} {res.key.mode} np={res.key.n_parallel}")
    for idx, resp in enumerate(res.responses):
        print(f"  req{idx}: {describe_response(resp)}")
        print(f"  req{idx}: content={resp['content']!r}")
    if res.profile:
        profile_dict = res.profile.to_dict()
        print(f"  profile={json.dumps({k: v for k, v in profile_dict.items() if k != 'steps'}, sort_keys=True)}")
    print(f"  log={res.log_path}")


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def aggregate_mode(results: list[ScenarioResult]) -> dict[str, Any]:
    per_repeat_mean_tok_s = [res.mean_tok_s() for res in results]
    per_response_tok_s = [tok_s for res in results for tok_s in res.predicted_per_second]
    draft_counts = [resp.get("timings", {}).get("draft_n", 0) for res in results for resp in res.responses]
    accepted_counts = [resp.get("timings", {}).get("draft_n_accepted", 0) for res in results for resp in res.responses]

    profile_totals = ProfileTotals()
    have_profile = False
    for res in results:
        if res.profile is None:
            continue
        have_profile = True
        profile_totals.draft_ms += res.profile.draft_ms
        profile_totals.draft_calls += res.profile.draft_calls
        profile_totals.snapshot_ms += res.profile.snapshot_ms
        profile_totals.snapshot_calls += res.profile.snapshot_calls
        profile_totals.accept_ms += res.profile.accept_ms
        profile_totals.accept_calls += res.profile.accept_calls
        profile_totals.restore_ms += res.profile.restore_ms
        profile_totals.restore_calls += res.profile.restore_calls
        profile_totals.replay_ms += res.profile.replay_ms
        profile_totals.replay_calls += res.profile.replay_calls
        profile_totals.total_ms += res.profile.total_ms
        profile_totals.acceptance_accepted += res.profile.acceptance_accepted
        profile_totals.acceptance_generated += res.profile.acceptance_generated
        profile_totals.steps.extend(res.profile.steps)

    data: dict[str, Any] = {
        "repeat_count": len(results),
        "per_repeat_mean_tok_s": per_repeat_mean_tok_s,
        "tok_s": stats(per_repeat_mean_tok_s),
        "per_response_tok_s": stats(per_response_tok_s),
        "draft_n_total": sum(draft_counts),
        "draft_n_accepted_total": sum(accepted_counts),
        "draft_acceptance_rate": None if sum(draft_counts) == 0 else sum(accepted_counts) / sum(draft_counts),
        "logs": [str(res.log_path) for res in results],
    }

    if have_profile:
        data["profile"] = profile_totals.to_dict()

    return data


def compute_speedup(baseline: dict[str, Any], mtp: dict[str, Any]) -> float | None:
    baseline_median = baseline["tok_s"]["median"]
    if baseline_median == 0:
        return None
    return mtp["tok_s"]["median"] / baseline_median


def compare_against_previous(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for case_name, case_data in current["cases"].items():
        prev_case = previous.get("cases", {}).get(case_name)
        if prev_case is None:
            continue
        case_cmp: dict[str, Any] = {}
        for np_key, np_data in case_data["n_parallel"].items():
            prev_np = prev_case.get("n_parallel", {}).get(np_key)
            if prev_np is None:
                continue
            cur_mtp = np_data.get("mtp")
            prev_mtp = prev_np.get("mtp")
            if cur_mtp is None or prev_mtp is None:
                continue
            prev_median = prev_mtp["tok_s"]["median"]
            if prev_median == 0:
                continue
            case_cmp[np_key] = {
                "prev_mtp_median_tok_s": prev_median,
                "cur_mtp_median_tok_s": cur_mtp["tok_s"]["median"],
                "speedup_vs_prev_mtp": cur_mtp["tok_s"]["median"] / prev_median,
            }
        if case_cmp:
            comparisons[case_name] = case_cmp
    return comparisons


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate native MTP against baseline llama-server output and throughput."
    )
    parser.add_argument("--binary", default="build-cuda/bin/llama-server", help="Path to llama-server")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port-base", type=int, default=18100, help="Base port for spawned servers")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt used for custom case")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base seed for custom case")
    parser.add_argument("--n-predict", type=int, default=DEFAULT_N_PREDICT, help="Generated tokens per request for custom case")
    parser.add_argument("--cases", default="custom", help="Comma-separated case names: custom,primary,good,bad")
    parser.add_argument("--n-parallels", default="1,2", help="Comma-separated list of -np values to run")
    parser.add_argument("--allow-known-np2-divergence", action="store_true", help="Do not enforce exact equality for n_parallel > 1")
    parser.add_argument("--repeat", "--repeats", dest="repeats", type=int, default=1, help="Number of repeats per scenario")
    parser.add_argument("--ctx-size", type=int, default=4096, help="Context size")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--ubatch-size", type=int, default=64, help="Ubatch size")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads")
    parser.add_argument("--threads-batch", type=int, default=8, help="CPU batch threads")
    parser.add_argument("--ngl", default="all", help="Value for -ngl")
    parser.add_argument("--flash-attn", default="on", choices=["on", "off", "auto"], help="Value for -fa")
    parser.add_argument("--startup-timeout", type=float, default=180.0, help="Seconds to wait for server health")
    parser.add_argument("--draft-max", type=int, default=1, help="Value for --draft-max in mtp mode")
    parser.add_argument("--mtp-profile", action="store_true", help="Set LLAMA_SERVER_MTP_PROFILE=1 and parse native MTP phase timing")
    parser.add_argument("--log-dir", default=None, help="Directory for per-scenario logs")
    parser.add_argument("--json-out", default=None, help="Write machine-readable summary JSON here")
    parser.add_argument("--compare-json", default=None, help="Optional previous summary JSON for step-to-step MTP comparison")
    args = parser.parse_args()

    binary = Path(args.binary).resolve()
    model = Path(args.model).resolve()
    if not binary.is_file():
        raise FileNotFoundError(f"llama-server not found: {binary}")
    if not model.is_file():
        raise FileNotFoundError(f"model not found: {model}")

    case_names = parse_case_names(args.cases)
    n_parallels = parse_n_parallels(args.n_parallels)
    cases = [get_case_config(name, args) for name in case_names]

    log_dir = Path(args.log_dir).resolve() if args.log_dir else Path(tempfile.mkdtemp(prefix="mtp-cuda-validate-"))
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"binary={binary}")
    print(f"model={model}")
    print(f"logs={log_dir}")

    results: dict[ScenarioKey, ScenarioResult] = {}
    scenario_counter = 0

    for case in cases:
        print(f"\n## case={case.name}: {case.description}")
        for repeat in range(args.repeats):
            for n_parallel in n_parallels:
                for mode in ("baseline", "mtp"):
                    port = args.port_base + scenario_counter
                    scenario_counter += 1
                    print(f"\n== running case={case.name} repeat={repeat} {mode} np={n_parallel} on port {port} ==")
                    res = run_scenario(
                        binary=binary,
                        model=model,
                        host=args.host,
                        port=port,
                        case=case,
                        mode=mode,
                        n_parallel=n_parallel,
                        repeat=repeat,
                        args=args,
                        log_dir=log_dir,
                    )
                    results[res.key] = res
                    print_result(res)

    print("\n== validating outputs ==")
    for case in cases:
        for repeat in range(args.repeats):
            for n_parallel in n_parallels:
                baseline = results[ScenarioKey(case.name, repeat, "baseline", n_parallel)]
                mtp = results[ScenarioKey(case.name, repeat, "mtp", n_parallel)]

                require_stable_outputs(baseline)
                require_stable_outputs(mtp)

                exact_required = n_parallel == 1 or not args.allow_known_np2_divergence
                if exact_required:
                    assert_equal_outputs(baseline, mtp)

                if not any(resp.get("timings", {}).get("draft_n", 0) > 0 for resp in mtp.responses):
                    raise AssertionError(
                        f"mtp case={case.name} repeat={repeat} np={n_parallel} did not report any native draft activity"
                    )

    print("output validation passed")

    summary: dict[str, Any] = {
        "binary": str(binary),
        "model": str(model),
        "config": {
            "cases": case_names,
            "n_parallels": n_parallels,
            "repeats": args.repeats,
            "allow_known_np2_divergence": args.allow_known_np2_divergence,
            "ctx_size": args.ctx_size,
            "batch_size": args.batch_size,
            "ubatch_size": args.ubatch_size,
            "threads": args.threads,
            "threads_batch": args.threads_batch,
            "ngl": args.ngl,
            "flash_attn": args.flash_attn,
            "draft_max": args.draft_max,
            "mtp_profile": args.mtp_profile,
            "log_dir": str(log_dir),
        },
        "cases": {},
    }

    for case in cases:
        case_summary: dict[str, Any] = {
            "description": case.description,
            "prompt": case.prompt,
            "seed": case.seed,
            "n_predict": case.n_predict,
            "n_parallel": {},
        }
        for n_parallel in n_parallels:
            mode_runs = {
                mode: [
                    results[ScenarioKey(case.name, repeat, mode, n_parallel)]
                    for repeat in range(args.repeats)
                ]
                for mode in ("baseline", "mtp")
            }
            np_summary = {
                mode: aggregate_mode(runs)
                for mode, runs in mode_runs.items()
            }
            np_summary["speedup_vs_baseline"] = compute_speedup(np_summary["baseline"], np_summary["mtp"])
            np_summary["exact_required"] = n_parallel == 1 or not args.allow_known_np2_divergence
            case_summary["n_parallel"][str(n_parallel)] = np_summary
        summary["cases"][case.name] = case_summary

    if args.compare_json:
        previous = json.loads(Path(args.compare_json).read_text(encoding="utf-8"))
        summary["comparison_vs_previous"] = compare_against_previous(summary, previous)

    print("\n== summary ==")
    for case_name, case_summary in summary["cases"].items():
        print(f"case={case_name}")
        for np_key, np_summary in case_summary["n_parallel"].items():
            baseline_median = np_summary["baseline"]["tok_s"]["median"]
            mtp_median = np_summary["mtp"]["tok_s"]["median"]
            speedup = np_summary["speedup_vs_baseline"]
            print(
                f"  np={np_key}: baseline median tok/s={baseline_median:.2f}, "
                f"mtp median tok/s={mtp_median:.2f}, "
                f"speedup={speedup:.3f}x" if speedup is not None else
                f"  np={np_key}: baseline median tok/s={baseline_median:.2f}, mtp median tok/s={mtp_median:.2f}"
            )

    if args.json_out:
        json_path = Path(args.json_out).resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\njson={json_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
