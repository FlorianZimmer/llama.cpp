#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "numpy is required; run this with ./.venv-tests/bin/python or an environment that has numpy installed"
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))

import gguf
from gguf.constants import GGMLQuantizationType
from gguf.gguf_reader import GGUFReader


@dataclass(frozen=True)
class Thresholds:
    name: str
    rel_rmse_max: float
    cosine_min: float


@dataclass
class TensorCandidateMetrics:
    label: str
    path: str
    tensor_name: str
    type_name: str
    n_bytes: int
    mae: float
    rmse: float
    rel_rmse: float
    max_abs: float
    cosine: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit native-MTP GGUF tensor quality against a BF16/F16/F32 reference and emit tensor override recommendations."
    )
    parser.add_argument("--reference", required=True, help="Reference GGUF path, ideally BF16/F16/F32.")
    parser.add_argument(
        "--candidate",
        dest="candidates",
        action="append",
        required=True,
        help="Candidate in label=path form. May be specified multiple times.",
    )
    parser.add_argument(
        "--baseline",
        help="Candidate label to treat as the base quant when writing override files.",
    )
    parser.add_argument(
        "--match-regex",
        default=r"\.nextn\.",
        help="Regex used to select MTP tensors from the reference GGUF.",
    )
    parser.add_argument(
        "--balanced-rel-rmse",
        type=float,
        default=0.05,
        help="Balanced recommendation threshold for relative RMSE.",
    )
    parser.add_argument(
        "--balanced-cosine",
        type=float,
        default=0.999,
        help="Balanced recommendation threshold for cosine similarity.",
    )
    parser.add_argument(
        "--strict-rel-rmse",
        type=float,
        default=0.02,
        help="Strict recommendation threshold for relative RMSE.",
    )
    parser.add_argument(
        "--strict-cosine",
        type=float,
        default=0.9999,
        help="Strict recommendation threshold for cosine similarity.",
    )
    parser.add_argument("--write-json", help="Write the full audit report to this JSON file.")
    parser.add_argument(
        "--write-balanced-type-file",
        help="Write a llama-quantize --tensor-type-file for the balanced recommendation.",
    )
    parser.add_argument(
        "--write-strict-type-file",
        help="Write a llama-quantize --tensor-type-file for the strict recommendation.",
    )
    return parser.parse_args()


def parse_candidate(spec: str) -> tuple[str, Path]:
    label, sep, raw_path = spec.partition("=")
    if sep == "" or not label or not raw_path:
        raise SystemExit(f"invalid --candidate value {spec!r}; expected label=path")
    return label, Path(raw_path)


def qtype_name(value: int) -> str:
    try:
        return GGMLQuantizationType(value).name
    except ValueError:
        return str(value)


def load_tensor_map(path: Path, name_re: re.Pattern[str]) -> tuple[GGUFReader, dict[str, object]]:
    reader = GGUFReader(str(path))
    tensors = {tensor.name: tensor for tensor in reader.tensors if name_re.search(tensor.name)}
    return reader, tensors


def dequantize_tensor(tensor: object) -> np.ndarray:
    qtype = GGMLQuantizationType(int(tensor.tensor_type))
    return gguf.dequantize(tensor.data, qtype).astype(np.float32, copy=False)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    denom = float(np.linalg.norm(a_flat) * np.linalg.norm(b_flat))
    if denom == 0.0:
        return 1.0 if float(np.linalg.norm(a_flat - b_flat)) == 0.0 else 0.0
    return float(np.dot(a_flat, b_flat) / denom)


def compute_metrics(
    label: str,
    path: Path,
    tensor_name: str,
    tensor: object,
    reference_arr: np.ndarray,
) -> TensorCandidateMetrics:
    arr = dequantize_tensor(tensor)
    if arr.shape != reference_arr.shape:
        raise SystemExit(
            f"shape mismatch for {tensor_name}: reference {reference_arr.shape} vs {label} {arr.shape}"
        )
    diff = arr - reference_arr
    rmse = float(math.sqrt(float(np.mean(diff * diff))))
    ref_rms = float(math.sqrt(float(np.mean(reference_arr * reference_arr))))
    rel_rmse = 0.0 if ref_rms == 0.0 else rmse / ref_rms
    return TensorCandidateMetrics(
        label=label,
        path=str(path),
        tensor_name=tensor_name,
        type_name=qtype_name(int(tensor.tensor_type)),
        n_bytes=int(tensor.n_bytes),
        mae=float(np.mean(np.abs(diff))),
        rmse=rmse,
        rel_rmse=rel_rmse,
        max_abs=float(np.max(np.abs(diff))),
        cosine=cosine_similarity(arr, reference_arr),
    )


def passes_thresholds(metrics: TensorCandidateMetrics, thresholds: Thresholds) -> bool:
    return metrics.rel_rmse <= thresholds.rel_rmse_max and metrics.cosine >= thresholds.cosine_min


def pick_recommendation(
    metrics_by_tensor: dict[str, list[TensorCandidateMetrics]],
    thresholds: Thresholds,
) -> tuple[dict[str, TensorCandidateMetrics], list[str]]:
    picks: dict[str, TensorCandidateMetrics] = {}
    warnings: list[str] = []
    for tensor_name, candidates in metrics_by_tensor.items():
        passing = [candidate for candidate in candidates if passes_thresholds(candidate, thresholds)]
        if passing:
            choice = min(passing, key=lambda item: (item.n_bytes, item.rel_rmse, -item.cosine))
            picks[tensor_name] = choice
            continue
        choice = min(candidates, key=lambda item: (item.rel_rmse, -item.cosine, item.n_bytes))
        picks[tensor_name] = choice
        warnings.append(
            f"{thresholds.name}: no candidate met thresholds for {tensor_name}; using best available {choice.label}:{choice.type_name}"
        )
    return picks, warnings


def write_type_file(
    output_path: Path,
    baseline_label: str,
    metrics_by_tensor: dict[str, list[TensorCandidateMetrics]],
    recommendation: dict[str, TensorCandidateMetrics],
) -> None:
    baseline_by_tensor = {
        tensor_name: next(candidate for candidate in candidates if candidate.label == baseline_label)
        for tensor_name, candidates in metrics_by_tensor.items()
    }
    lines: list[str] = []
    for tensor_name in sorted(recommendation):
        baseline = baseline_by_tensor[tensor_name]
        picked = recommendation[tensor_name]
        if baseline.type_name == picked.type_name:
            continue
        lines.append(f"^{re.escape(tensor_name)}$={picked.type_name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def print_report(
    reference_path: Path,
    metrics_by_tensor: dict[str, list[TensorCandidateMetrics]],
    balanced: dict[str, TensorCandidateMetrics],
    strict: dict[str, TensorCandidateMetrics],
    warnings: list[str],
) -> None:
    print(f"reference: {reference_path}")
    print(f"mtp_tensors: {len(metrics_by_tensor)}")
    for tensor_name in sorted(metrics_by_tensor):
        print(f"\n{tensor_name}")
        print("  candidate         type    bytes      rel_rmse   cosine      mae         max_abs")
        for item in sorted(metrics_by_tensor[tensor_name], key=lambda metric: (metric.n_bytes, metric.label)):
            print(
                f"  {item.label:<16} {item.type_name:<7} {item.n_bytes:>9} "
                f"{item.rel_rmse:>10.6f} {item.cosine:>10.6f} {item.mae:>11.8f} {item.max_abs:>11.8f}"
            )
        print(f"  balanced -> {balanced[tensor_name].label}:{balanced[tensor_name].type_name}")
        print(f"  strict   -> {strict[tensor_name].label}:{strict[tensor_name].type_name}")
    if warnings:
        print("\nwarnings:")
        for warning in warnings:
            print(f"  - {warning}")


def main() -> None:
    args = parse_args()
    thresholds_balanced = Thresholds(
        name="balanced",
        rel_rmse_max=args.balanced_rel_rmse,
        cosine_min=args.balanced_cosine,
    )
    thresholds_strict = Thresholds(
        name="strict",
        rel_rmse_max=args.strict_rel_rmse,
        cosine_min=args.strict_cosine,
    )

    candidate_specs = [parse_candidate(spec) for spec in args.candidates]
    candidate_labels = [label for label, _ in candidate_specs]
    if len(candidate_labels) != len(set(candidate_labels)):
        raise SystemExit("candidate labels must be unique")
    if args.baseline and args.baseline not in set(candidate_labels):
        raise SystemExit(f"baseline label {args.baseline!r} is not present in --candidate")

    name_re = re.compile(args.match_regex)
    reference_path = Path(args.reference)
    _, reference_tensors = load_tensor_map(reference_path, name_re)
    if not reference_tensors:
        raise SystemExit(f"no tensors matched {args.match_regex!r} in {reference_path}")

    candidate_maps: dict[str, dict[str, object]] = {}
    for label, path in candidate_specs:
        _, tensors = load_tensor_map(path, name_re)
        missing = sorted(set(reference_tensors) - set(tensors))
        if missing:
            raise SystemExit(f"{label} is missing tensors present in reference: {missing}")
        candidate_maps[label] = tensors

    ref_arrays = {name: dequantize_tensor(tensor) for name, tensor in reference_tensors.items()}
    metrics_by_tensor: dict[str, list[TensorCandidateMetrics]] = {}
    for tensor_name in sorted(reference_tensors):
        rows: list[TensorCandidateMetrics] = []
        reference_arr = ref_arrays[tensor_name]
        for label, path in candidate_specs:
            rows.append(compute_metrics(label, path, tensor_name, candidate_maps[label][tensor_name], reference_arr))
        metrics_by_tensor[tensor_name] = rows

    balanced, warnings_balanced = pick_recommendation(metrics_by_tensor, thresholds_balanced)
    strict, warnings_strict = pick_recommendation(metrics_by_tensor, thresholds_strict)
    warnings = warnings_balanced + warnings_strict

    if args.write_balanced_type_file:
        if not args.baseline:
            raise SystemExit("--baseline is required when writing override files")
        write_type_file(Path(args.write_balanced_type_file), args.baseline, metrics_by_tensor, balanced)
    if args.write_strict_type_file:
        if not args.baseline:
            raise SystemExit("--baseline is required when writing override files")
        write_type_file(Path(args.write_strict_type_file), args.baseline, metrics_by_tensor, strict)

    print_report(reference_path, metrics_by_tensor, balanced, strict, warnings)

    if args.write_json:
        report = {
            "reference": str(reference_path),
            "match_regex": args.match_regex,
            "thresholds": {
                "balanced": asdict(thresholds_balanced),
                "strict": asdict(thresholds_strict),
            },
            "tensors": {
                tensor_name: [asdict(item) for item in items] for tensor_name, items in metrics_by_tensor.items()
            },
            "recommendations": {
                "balanced": {tensor_name: asdict(item) for tensor_name, item in balanced.items()},
                "strict": {tensor_name: asdict(item) for tensor_name, item in strict.items()},
            },
            "warnings": warnings,
        }
        output_path = Path(args.write_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
