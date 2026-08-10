#!/usr/bin/env python3
"""Extract reviewer-facing structural diagnostics from a real pilot archive.

The command never estimates live performance.  It reads the manifests and
controller records emitted by ``run_training_fastpath_pilot.py`` and reports
(1) how similar the six-feature ordering is to a duration-only baseline and
(2) which FastPath branch actually executed.  Outputs are sanitized: node,
namespace, Pod UID, wall-clock timestamps and service names are omitted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scheduler.constants import ANNOTATION_MAP  # noqa: E402
from scheduler.rank import (  # noqa: E402
    JobFeatures,
    compute_duration_only_ranks,
    compute_ranks,
)

REPETITION_PATTERN = re.compile(r"-r(?P<repetition>\d+)-")


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation requires equal vectors with at least two values")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_scale == 0 or right_scale == 0:
        return 0.0
    return numerator / (left_scale * right_scale)


def _rankdata(values: Sequence[float]) -> list[float]:
    """Return ascending average ranks; equal values receive the same rank."""

    ranked = [0.0] * len(values)
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and ordered[stop][1] == ordered[start][1]:
            stop += 1
        average_rank = (start + 1 + stop) / 2.0
        for index, _ in ordered[start:stop]:
            ranked[index] = average_rank
        start = stop
    return ranked


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    return _pearson(_rankdata(left), _rankdata(right))


def _kendall_tau_b(left: Sequence[float], right: Sequence[float]) -> float:
    concordant = discordant = ties_left = ties_right = 0
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
            delta_left = left[first] - left[second]
            delta_right = right[first] - right[second]
            if delta_left == 0 and delta_right == 0:
                continue
            if delta_left == 0:
                ties_left += 1
            elif delta_right == 0:
                ties_right += 1
            elif delta_left * delta_right > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_left) * (concordant + discordant + ties_right)
    )
    return (concordant - discordant) / denominator if denominator else 0.0


def _repetition(path: Path) -> int:
    match = REPETITION_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"cannot derive repetition from {path.name!r}")
    return int(match.group("repetition"))


def _manifest_jobs(path: Path) -> list[tuple[int, JobFeatures]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows: list[tuple[int, JobFeatures]] = []
    for fallback_index, pod in enumerate(document.get("items", [])):
        metadata = pod["metadata"]
        annotations = metadata["annotations"]
        labels = metadata.get("labels", {})
        index = int(labels.get("pilot.kubeml/job-index", fallback_index))
        rows.append(
            (
                index,
                JobFeatures(
                    job_id=f"job-{index:02d}",
                    **{
                        feature: float(annotations[annotation])
                        for feature, annotation in ANNOTATION_MAP.items()
                    },
                ),
            )
        )
    if not rows:
        raise ValueError(f"manifest {path} has no Pods")
    return sorted(rows)


def _order(ranks: Mapping[str, float]) -> list[str]:
    return sorted(ranks, key=lambda job_id: (-ranks[job_id], job_id))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_pilot(pilot_dir: Path, output_dir: Path) -> dict[str, object]:
    manifests = sorted(pilot_dir.glob("tfp-k-r*-manifest.json"), key=_repetition)
    gate_records = sorted(pilot_dir.glob("gate-tfp-k-r*.json"), key=_repetition)
    if not manifests or len(manifests) != len(gate_records):
        raise ValueError("pilot archive must contain matching KubeML manifests and gate records")

    feature_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    for manifest in manifests:
        repetition = _repetition(manifest)
        indexed_jobs = _manifest_jobs(manifest)
        jobs = [job for _, job in indexed_jobs]
        full = compute_ranks(jobs)
        duration = compute_duration_only_ranks(jobs)
        full_order = _order(full)
        duration_order = _order(duration)
        full_position = {job_id: index + 1 for index, job_id in enumerate(full_order)}
        duration_position = {job_id: index + 1 for index, job_id in enumerate(duration_order)}
        vectors = [full[job.job_id] for job in jobs]
        duration_vectors = [duration[job.job_id] for job in jobs]
        shifts = [abs(full_position[job.job_id] - duration_position[job.job_id]) for job in jobs]
        top_k = min(3, len(jobs))
        diagnostic_rows.append(
            {
                "repetition": repetition,
                "jobs": len(jobs),
                "score_pearson": _pearson(vectors, duration_vectors),
                "spearman_rho": _spearman(vectors, duration_vectors),
                "kendall_tau_b": _kendall_tau_b(vectors, duration_vectors),
                "exact_order_match": full_order == duration_order,
                "jobs_with_changed_position": sum(shift > 0 for shift in shifts),
                "mean_absolute_position_shift": statistics.fmean(shifts),
                "maximum_position_shift": max(shifts),
                "top3_overlap": len(set(full_order[:top_k]) & set(duration_order[:top_k])),
            }
        )
        for index, job in indexed_jobs:
            feature_rows.append(
                {
                    "repetition": repetition,
                    "job_index": index,
                    "T": job.T,
                    "R": job.R,
                    "M": job.M,
                    "G": job.G,
                    "C": job.C,
                    "P": job.P,
                    "six_feature_score": full[job.job_id],
                    "duration_only_score": duration[job.job_id],
                    "six_feature_position": full_position[job.job_id],
                    "duration_only_position": duration_position[job.job_id],
                }
            )

    fast_path_rows: list[dict[str, object]] = []
    for path in gate_records:
        repetition = _repetition(path)
        document = json.loads(path.read_text(encoding="utf-8"))
        events = document.get("events", [])
        decision = next(event for event in events if event.get("event") == "fast_path_decision")
        tail = next(
            (event for event in events if event.get("event") == "training_tail_balance"),
            {},
        )
        fast_path_rows.append(
            {
                "repetition": repetition,
                "selected": decision["selected"],
                "reason": decision["reason"],
                "cpu_threshold": decision["cpu_threshold"],
                "current_cpu_utilization": decision["current_cpu_utilization"],
                "burst_cpu_demand_cores": decision["burst_cpu_demand_cores"],
                "allocatable_cpu_cores": decision["allocatable_cpu_cores"],
                "projected_cpu_utilization": decision["projected_cpu_utilization"],
                "headroom_release_count": decision["headroom_release_count"],
                "initial_release_count": decision["initial_release_count"],
                "tail_balance_selected": tail.get("selected", False),
                "tail_parallelism": tail.get("parallelism"),
                "protected_prefix": tail.get("protected_prefix"),
                "predicted_tail_makespan_before": tail.get("predicted_makespan_before"),
                "predicted_tail_makespan_after": tail.get("predicted_makespan_after"),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "pilot_feature_matrix.csv", feature_rows)
    _write_csv(output_dir / "rank_policy_diagnostics.csv", diagnostic_rows)
    _write_csv(output_dir / "fastpath_branch_evidence.csv", fast_path_rows)

    def mean(field: str, rows: Sequence[Mapping[str, object]]) -> float:
        return statistics.fmean(float(row[field]) for row in rows)

    summary: dict[str, object] = {
        "schema_version": "1.0",
        "kind": "kubeml-pilot-structural-diagnostics",
        "evidence": {
            "source": "archived real-cluster manifests and gate-controller records",
            "repetitions": len(diagnostic_rows),
            "jobs_per_repetition": diagnostic_rows[0]["jobs"],
            "performance_claim": False,
        },
        "six_feature_vs_duration_only": {
            "mean_score_pearson": mean("score_pearson", diagnostic_rows),
            "mean_spearman_rho": mean("spearman_rho", diagnostic_rows),
            "mean_kendall_tau_b": mean("kendall_tau_b", diagnostic_rows),
            "exact_order_matches": sum(bool(row["exact_order_match"]) for row in diagnostic_rows),
            "mean_jobs_with_changed_position": mean("jobs_with_changed_position", diagnostic_rows),
            "mean_absolute_position_shift": mean("mean_absolute_position_shift", diagnostic_rows),
            "mean_top3_overlap": mean("top3_overlap", diagnostic_rows),
            "interpretation": (
                "Structural comparison only. It quantifies policy overlap but does not "
                "measure the duration-only policy on the cluster."
            ),
        },
        "fast_path": {
            "reasons": {
                reason: sum(row["reason"] == reason for row in fast_path_rows)
                for reason in sorted({str(row["reason"]) for row in fast_path_rows})
            },
            "mean_current_cpu_utilization": mean("current_cpu_utilization", fast_path_rows),
            "mean_projected_cpu_utilization": mean("projected_cpu_utilization", fast_path_rows),
            "burst_cpu_demand_cores": fast_path_rows[0]["burst_cpu_demand_cores"],
            "allocatable_cpu_cores": fast_path_rows[0]["allocatable_cpu_cores"],
            "cpu_threshold": fast_path_rows[0]["cpu_threshold"],
            "tail_balance_selected_repetitions": sum(
                bool(row["tail_balance_selected"]) for row in fast_path_rows
            ),
        },
    }
    (output_dir / "structural_diagnostics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(analyze_pilot(args.pilot_dir, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
