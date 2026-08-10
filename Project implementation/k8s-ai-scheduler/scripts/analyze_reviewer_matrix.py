#!/usr/bin/env python3
"""Validate and summarize a completed reviewer-hardening matrix.

The analyzer fails closed: an incomplete matrix cannot produce a publication
table. Shared-node runs can be analyzed only with an explicit exploratory flag,
and the generated artifacts retain that evidence boundary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

EXPECTED_ARMS = (
    "native-default",
    "baseline",
    "duration-only",
    "kubeml",
    "six-feature-no-tail",
    "six-feature-no-fastpath",
    "reversed",
)
MINIMUM_REPETITIONS = 30
METRICS = (
    "avg_jct_seconds",
    "p95_jct_seconds",
    "makespan_seconds",
    "max_to_median_jct_ratio",
    "p95_to_median_jct_ratio",
)
COMPARISONS = {
    "end-to-end-vs-native": ("native-default", "kubeml"),
    "end-to-end-vs-synchronized": ("baseline", "kubeml"),
    "six-feature-increment": ("duration-only", "kubeml"),
    "tail-balance-increment": ("six-feature-no-tail", "kubeml"),
    "fastpath-increment": ("six-feature-no-fastpath", "six-feature-no-tail"),
    "intended-vs-reversed": ("reversed", "kubeml"),
}


class MatrixValidationError(ValueError):
    pass


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise MatrixValidationError("cannot compute a percentile of an empty sample")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def bootstrap_ci(values: list[float], *, resamples: int = 20_000) -> tuple[float, float]:
    if len(values) < 2:
        raise MatrixValidationError("paired confidence intervals require at least two blocks")
    rng = random.Random(20260810)
    estimates = [
        mean(values[rng.randrange(len(values))] for _ in values) for _ in range(resamples)
    ]
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def exact_two_sided_sign_p(values: Iterable[float]) -> float:
    nonzero = [value for value in values if not math.isclose(value, 0.0, abs_tol=1e-12)]
    if not nonzero:
        return 1.0
    positives = sum(value > 0 for value in nonzero)
    tail = min(positives, len(nonzero) - positives)
    probability = sum(math.comb(len(nonzero), k) for k in range(tail + 1)) / (2 ** len(nonzero))
    return min(1.0, 2.0 * probability)


def validate_report(
    report: dict[str, Any], *, allow_nonclaimable: bool = False
) -> tuple[int, dict[tuple[int, str], dict[str, Any]]]:
    repetitions = int(report.get("repetitions", 0))
    if repetitions < MINIMUM_REPETITIONS:
        raise MatrixValidationError(
            f"matrix requires at least {MINIMUM_REPETITIONS} complete blocks"
        )
    runs = report.get("runs")
    if not isinstance(runs, list):
        raise MatrixValidationError("result document has no runs list")
    indexed: dict[tuple[int, str], dict[str, Any]] = {}
    for run in runs:
        key = (int(run.get("repetition", -1)), str(run.get("arm", "")))
        if key in indexed:
            raise MatrixValidationError(f"duplicate run block: {key}")
        indexed[key] = run
        missing_metrics = set(METRICS) - set(run.get("metrics") or {})
        if missing_metrics:
            raise MatrixValidationError(f"run {key} is missing metrics: {sorted(missing_metrics)}")
    expected = {
        (repetition, arm)
        for repetition in range(repetitions)
        for arm in EXPECTED_ARMS
    }
    missing = expected - set(indexed)
    extra = set(indexed) - expected
    if missing or extra:
        raise MatrixValidationError(
            f"matrix is not a complete {repetitions} x 7 design; "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    if not report.get("eligible_for_article_claim") and not allow_nonclaimable:
        raise MatrixValidationError(
            "matrix is marked non-claimable; pass --allow-nonclaimable only for exploratory output"
        )
    return repetitions, indexed


def summarize(
    report: dict[str, Any],
    *,
    allow_nonclaimable: bool = False,
    bootstrap_resamples: int = 20_000,
) -> dict[str, Any]:
    repetitions, indexed = validate_report(
        report, allow_nonclaimable=allow_nonclaimable
    )
    arm_means = {
        arm: {
            metric: mean(
                float(indexed[(repetition, arm)]["metrics"][metric])
                for repetition in range(repetitions)
            )
            for metric in METRICS
        }
        for arm in EXPECTED_ARMS
    }
    effects: list[dict[str, Any]] = []
    for comparison, (reference, candidate) in COMPARISONS.items():
        for metric in METRICS:
            paired = []
            for repetition in range(repetitions):
                reference_value = float(indexed[(repetition, reference)]["metrics"][metric])
                candidate_value = float(indexed[(repetition, candidate)]["metrics"][metric])
                paired.append(100.0 * (reference_value - candidate_value) / reference_value)
            ci_low, ci_high = bootstrap_ci(paired, resamples=bootstrap_resamples)
            effects.append(
                {
                    "comparison": comparison,
                    "reference": reference,
                    "candidate": candidate,
                    "metric": metric,
                    "n": len(paired),
                    "mean_improvement_pct": mean(paired),
                    "sd_pct": stdev(paired),
                    "bootstrap_ci95_low_pct": ci_low,
                    "bootstrap_ci95_high_pct": ci_high,
                    "sign_test_two_sided_p": exact_two_sided_sign_p(paired),
                    "improved_blocks": sum(value > 0 for value in paired),
                }
            )
    return {
        "schema_version": "1.0",
        "kind": "reviewer-hardening-analysis",
        "publication_ready": bool(report.get("eligible_for_article_claim")),
        "evidence_boundary": report.get("reason"),
        "repetitions": repetitions,
        "arms": list(EXPECTED_ARMS),
        "arm_means": arm_means,
        "effects": effects,
    }


def write_outputs(summary: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "reviewer_matrix_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    effects = summary["effects"]
    with (output / "reviewer_matrix_effects.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(effects[0]))
        writer.writeheader()
        writer.writerows(effects)
    primary = {
        row["comparison"]: row
        for row in effects
        if row["metric"] == "avg_jct_seconds"
    }
    rows = []
    for comparison in COMPARISONS:
        result = primary[comparison]
        rows.append(
            f"{comparison.replace('-', ' ')} & {result['mean_improvement_pct']:.2f} & "
            f"[{result['bootstrap_ci95_low_pct']:.2f}, "
            f"{result['bootstrap_ci95_high_pct']:.2f}] & "
            f"{result['sign_test_two_sided_p']:.4f} \\\\"
        )
    label = "publishable" if summary["publication_ready"] else "exploratory only"
    table = "\n".join(
        [
            f"% Evidence status: {label}",
            "\\begin{tabular}{lrrr}",
            "\\toprule",
            "Comparison & Mean improvement (\\%) & Bootstrap 95\\% CI & Sign-test $p$ \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "",
        ]
    )
    (output / "reviewer_matrix_table.tex").write_text(table, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-nonclaimable", action="store_true")
    args = parser.parse_args(argv)
    report = json.loads(args.result.read_text(encoding="utf-8"))
    summary = summarize(report, allow_nonclaimable=args.allow_nonclaimable)
    write_outputs(summary, args.output_dir)
    print(args.output_dir / "reviewer_matrix_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
