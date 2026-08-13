#!/usr/bin/env python3
"""Create a complete, explicitly optimistic 200-workload what-if bundle."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


OUTPUT = Path(__file__).resolve().parent
SOURCE = OUTPUT / "estimated-results-200.csv"
STATUS = "optimistic-synthetic-what-if-not-measured"
ASSUMPTIONS = {
    "average_jct_incremental_gain_vs_duration_only": 0.06,
    "p95_jct_incremental_gain_vs_duration_only": 0.03,
    "makespan_incremental_gain_vs_duration_only": 0.02,
}
LABELS = {
    "kubernetes-default": "Kubernetes default",
    "duration-only": "Duration-only",
    "kubeml-optimistic": "KubeML-Aware",
    "reversed-ablation": "Reversed ablation",
}
COLORS = {
    "kubernetes-default": "#667085",
    "duration-only": "#7f56d9",
    "kubeml-optimistic": "#1570ef",
    "reversed-ablation": "#f04438",
}
POLICIES = tuple(LABELS)


def load_rows() -> list[dict[str, str]]:
    with SOURCE.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def percentile(values: list[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), quantile))


def summarize(rows: list[dict[str, object]]) -> dict[str, float]:
    jcts = [float(row["jct_seconds"]) for row in rows]
    starts = sorted(float(row["execution_start_seconds"]) for row in rows)
    return {
        "average_jct_seconds": statistics.mean(jcts),
        "median_jct_seconds": statistics.median(jcts),
        "p95_jct_seconds": percentile(jcts, 0.95),
        "maximum_jct_seconds": max(jcts),
        "makespan_seconds": max(float(row["completion_seconds"]) for row in rows),
        "average_ilt_seconds": statistics.mean(
            right - left for left, right in zip(starts, starts[1:])
        ),
    }


def optimistic_rows(duration_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    ordered = sorted(duration_rows, key=lambda row: float(row["jct_seconds"]))
    values = np.asarray([float(row["jct_seconds"]) for row in ordered], dtype=float)
    percentiles = np.linspace(0.0, 1.0, len(values))
    p95_index = int(round(0.95 * (len(values) - 1)))
    design = np.column_stack(
        [values, values * percentiles, values * percentiles**2]
    )
    constraints = np.asarray(
        [
            design.mean(axis=0),
            design[p95_index],
            design[-1],
        ]
    )
    targets = np.asarray(
        [
            values.mean()
            * (1.0 - ASSUMPTIONS["average_jct_incremental_gain_vs_duration_only"]),
            values[p95_index]
            * (1.0 - ASSUMPTIONS["p95_jct_incremental_gain_vs_duration_only"]),
            values[-1]
            * (1.0 - ASSUMPTIONS["makespan_incremental_gain_vs_duration_only"]),
        ]
    )
    coefficients = np.linalg.solve(constraints, targets)
    optimistic = design @ coefficients
    rows: list[dict[str, object]] = []
    for source, jct in zip(ordered, optimistic):
        original_jct = float(source["jct_seconds"])
        factor = float(jct / original_jct)
        rows.append(
            {
                "policy": "kubeml-optimistic",
                "workload_id": source["workload_id"],
                "execution_order": int(source["execution_order"]),
                "rank_used": float(source["rank_used"]),
                "execution_start_seconds": round(
                    float(source["execution_start_seconds"]) * factor, 6
                ),
                "completion_seconds": round(float(jct), 6),
                "jct_seconds": round(float(jct), 6),
                "assumed_job_time_factor": round(factor, 9),
                "evidence_status": STATUS,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def annotate_bars(axis, bars, values, improvements) -> None:
    for bar, value, improvement in zip(bars, values, improvements):
        delta = "baseline" if abs(improvement) < 1e-9 else f"{improvement:+.1f}%"
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}\n{delta}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def metric_figure(summary: dict[str, dict[str, float]]):
    panels = (
        ("average_jct_seconds", "Average JCT"),
        ("p95_jct_seconds", "p95 JCT"),
        ("makespan_seconds", "Makespan"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2))
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.15, top=0.93, wspace=0.25)
    baseline = summary["kubernetes-default"]
    for axis, (metric, title) in zip(axes, panels):
        values = [summary[policy][metric] for policy in POLICIES]
        improvements = [
            100.0 * (baseline[metric] - value) / baseline[metric] for value in values
        ]
        bars = axis.bar(
            range(len(POLICIES)), values, color=[COLORS[policy] for policy in POLICIES]
        )
        axis.set_title(title)
        axis.set_ylabel("Seconds (lower is better)")
        axis.set_xticks(
            range(len(POLICIES)),
            [LABELS[policy].replace(" ", "\n", 1) for policy in POLICIES],
        )
        axis.grid(axis="y", alpha=0.25)
        axis.set_ylim(0, max(values) * 1.20)
        annotate_bars(axis, bars, values, improvements)
    return fig


def distribution_figure(grouped, summary):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.14, top=0.93, wspace=0.25)
    baseline_avg = summary["kubernetes-default"]["average_jct_seconds"]
    for policy in POLICIES:
        jcts = sorted(float(row["jct_seconds"]) for row in grouped[policy])
        ecdf = np.arange(1, len(jcts) + 1) / len(jcts)
        improvement = 100.0 * (baseline_avg - statistics.mean(jcts)) / baseline_avg
        label = f"{LABELS[policy]} ({improvement:+.1f}% Avg JCT)"
        axes[0].plot(jcts, ecdf, label=label, color=COLORS[policy], linewidth=2)
        completions = sorted(
            float(row["completion_seconds"]) for row in grouped[policy]
        )
        axes[1].plot(
            range(1, len(completions) + 1),
            completions,
            label=label,
            color=COLORS[policy],
            linewidth=2,
        )
    axes[0].set_title("JCT distribution")
    axes[0].set_xlabel("JCT (seconds)")
    axes[0].set_ylabel("ECDF")
    axes[1].set_title("Completion curve")
    axes[1].set_xlabel("Completed workloads")
    axes[1].set_ylabel("Seconds since burst submission")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    return fig


def improvement_figure(summary):
    metrics = (
        ("average_jct_seconds", "Avg JCT"),
        ("p95_jct_seconds", "p95 JCT"),
        ("makespan_seconds", "Makespan"),
    )
    comparison_policies = ("duration-only", "kubeml-optimistic", "reversed-ablation")
    baseline = summary["kubernetes-default"]
    x = np.arange(len(metrics))
    width = 0.24
    fig, axis = plt.subplots(figsize=(11.5, 6.2))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.15, top=0.93)
    series_values = {
        policy: [
            100.0 * (baseline[metric] - summary[policy][metric]) / baseline[metric]
            for metric, _ in metrics
        ]
        for policy in comparison_policies
    }
    for index, policy in enumerate(comparison_policies):
        values = series_values[policy]
        bars = axis.bar(
            x + (index - 1) * width,
            values,
            width,
            label=LABELS[policy],
            color=COLORS[policy],
        )
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + (0.8 if value >= 0 else -1.6),
                f"{value:+.1f}%",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=9,
            )
    axis.axhline(0, color="#667085", linewidth=1)
    all_values = [value for values in series_values.values() for value in values]
    axis.set_ylim(min(all_values) - 6.0, max(all_values) + 6.0)
    axis.set_xticks(x, [label for _, label in metrics])
    axis.set_ylabel("Improvement over Kubernetes default (%)")
    axis.set_title("Estimated improvement percentages")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    return fig


def main() -> int:
    source_rows = load_rows()
    grouped_source = {
        policy: [row for row in source_rows if row["policy"] == policy]
        for policy in ("kubernetes-default", "duration-only", "reversed-ablation")
    }
    optimistic = optimistic_rows(grouped_source["duration-only"])
    grouped: dict[str, list[dict[str, object]]] = {
        "kubernetes-default": grouped_source["kubernetes-default"],
        "duration-only": grouped_source["duration-only"],
        "kubeml-optimistic": optimistic,
        "reversed-ablation": grouped_source["reversed-ablation"],
    }
    summary = {policy: summarize(rows) for policy, rows in grouped.items()}
    baseline = summary["kubernetes-default"]
    summary_rows = []
    for policy in POLICIES:
        values = summary[policy]
        summary_rows.append(
            {
                "policy": policy,
                **{key: round(value, 6) for key, value in values.items()},
                "average_jct_improvement_vs_default_percent": round(
                    100
                    * (baseline["average_jct_seconds"] - values["average_jct_seconds"])
                    / baseline["average_jct_seconds"],
                    6,
                ),
                "p95_jct_improvement_vs_default_percent": round(
                    100
                    * (baseline["p95_jct_seconds"] - values["p95_jct_seconds"])
                    / baseline["p95_jct_seconds"],
                    6,
                ),
                "makespan_improvement_vs_default_percent": round(
                    100
                    * (baseline["makespan_seconds"] - values["makespan_seconds"])
                    / baseline["makespan_seconds"],
                    6,
                ),
                "evidence_status": STATUS,
            }
        )

    all_rows = []
    for policy in POLICIES:
        for row in grouped[policy]:
            normalized = dict(row)
            normalized["policy"] = policy
            normalized["evidence_status"] = STATUS
            all_rows.append(normalized)
    write_csv(OUTPUT / "حدسی-خوشبینانه-نتایج-200-workload.csv", all_rows)
    write_csv(OUTPUT / "حدسی-خوشبینانه-خلاصه-200-workload.csv", summary_rows)
    (OUTPUT / "حدسی-خوشبینانه-نتایج-200-workload.json").write_text(
        json.dumps(
            {
                "evidence_status": STATUS,
                "assumptions": ASSUMPTIONS,
                "summary": summary_rows,
                "results": all_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    figures = [
        (
            metric_figure(summary),
            "حدسی-خوشبینانه-تمام-متریک‌ها-200-workload",
        ),
        (
            distribution_figure(grouped, summary),
            "حدسی-خوشبینانه-توزیع-و-تکمیل-200-workload",
        ),
        (
            improvement_figure(summary),
            "حدسی-خوشبینانه-درصد-بهبود-200-workload",
        ),
    ]
    with PdfPages(OUTPUT / "حدسی-خوشبینانه-تمام-نمودارها-200-workload.pdf") as pdf:
        for figure, name in figures:
            figure.savefig(OUTPUT / f"{name}.png", dpi=190)
            figure.savefig(OUTPUT / f"{name}.pdf")
            pdf.savefig(figure)
            plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
