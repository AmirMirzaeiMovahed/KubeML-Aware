#!/usr/bin/env python3
"""Generate a clearly labelled 200-workload KubeML forecast bundle."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np


def percentile(values: list[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), quantile))


def metrics(rows: list[dict[str, float]]) -> dict[str, float]:
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


def atomic_json(path: Path, document: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(project_root))

    import sim.simulate as simulator
    from scheduler.rank import (
        compute_duration_only_ranks,
        compute_ranks,
        sort_by_training_policy,
    )
    from sim.simulate import DEFAULT_WORK_MODEL, estimate_trainer_work, results_to_dicts
    from workload.generate_workload import category_for_job, generate_category_burst

    jobs = generate_category_burst(args.jobs, category="heavy", seed=args.seed)
    six_ranks = compute_ranks(jobs)
    duration_ranks = compute_duration_only_ranks(jobs)
    six_order = {
        job.job_id: index + 1
        for index, job in enumerate(
            sort_by_training_policy(jobs, policy="six_feature")
        )
    }
    duration_order = {
        job.job_id: index + 1
        for index, job in enumerate(
            sort_by_training_policy(jobs, policy="duration_only")
        )
    }

    workload_rows: list[dict[str, object]] = []
    for job in jobs:
        workload_rows.append(
            {
                "workload_id": job.job_id,
                "category": category_for_job(job),
                "T_estimated_training_seconds": round(float(job.T), 6),
                "R_loss_reduction_rate": round(float(job.R), 6),
                "M_matrix_dimension": int(job.M),
                "G_gradient_update_mib": round(float(job.G), 6),
                "C_checkpoint_interval_steps": int(job.C),
                "P_model_partitions": int(job.P),
                "six_feature_rank": round(six_ranks[job.job_id], 9),
                "duration_only_rank": round(duration_ranks[job.job_id], 9),
                "six_feature_position": six_order[job.job_id],
                "duration_only_position": duration_order[job.job_id],
                "proxy_work_seconds_uncalibrated": round(
                    estimate_trainer_work(job, DEFAULT_WORK_MODEL), 6
                ),
                "generation_seed": args.seed,
                "evidence_status": "synthetic-estimate-not-executed-on-kubernetes",
            }
        )

    common = {
        "n_cores": 4,
        "dt": 0.10,
        "alpha": 0.05,
        "max_time": 200000.0,
        "work_model": DEFAULT_WORK_MODEL,
    }
    policy_results = {
        "kubernetes-default": simulator.run_default(
            jobs, seed=args.seed, mean_ilt=1.0, **common
        ),
        "six-feature": simulator.run_paced(
            jobs, mode="fixed", inherent_gap=1.0, **common
        ),
        "kubeml-fastpath-tail": simulator.run_paced(
            jobs,
            mode="fixed",
            inherent_gap=1.0,
            balance_tail=True,
            tail_window=8,
            **common,
        ),
        "reversed-ablation": simulator.run_paced(
            jobs, mode="fixed", inherent_gap=1.0, reverse=True, **common
        ),
    }
    duration_sort = lambda values, reverse_order=False: sort_by_training_policy(
        values, policy="duration_only", reverse_order=reverse_order
    )
    with patch.object(simulator, "sort_by_rank", side_effect=duration_sort):
        policy_results["duration-only"] = simulator.run_paced(
            jobs, mode="fixed", inherent_gap=1.0, **common
        )

    # Calibrate only the global time unit to the archived five-run pilot mean.
    # Policy deltas remain outputs of the proxy simulator and are not fitted.
    reference_jobs = generate_category_burst(12, category="heavy", seed=args.seed)
    reference = simulator.run_default(
        reference_jobs, seed=args.seed, mean_ilt=1.0, **common
    )
    reference_avg_jct = statistics.mean(item.jct for item in reference)
    pilot_baseline_avg_jct = 31.046652591228487
    time_scale = pilot_baseline_avg_jct / reference_avg_jct

    result_rows: list[dict[str, object]] = []
    for policy, results in policy_results.items():
        ordered = sorted(results, key=lambda result: result.exec_start_t)
        for order, row in enumerate(results_to_dicts(ordered), start=1):
            job_id = str(row["job_id"])
            rank = (
                duration_ranks[job_id]
                if policy == "duration-only"
                else six_ranks[job_id]
            )
            result_rows.append(
                {
                    "policy": policy,
                    "workload_id": job_id,
                    "execution_order": order,
                    "rank_used": round(float(rank), 9),
                    "execution_start_seconds": round(
                        float(row["execution_start_time"]) * time_scale, 6
                    ),
                    "completion_seconds": round(
                        float(row["completion_time"]) * time_scale, 6
                    ),
                    "jct_seconds": round(float(row["jct_s"]) * time_scale, 6),
                    "calibration_scale": round(time_scale, 9),
                    "evidence_status": "synthetic-estimate-not-executed-on-kubernetes",
                }
            )

    grouped = {
        policy: [row for row in result_rows if row["policy"] == policy]
        for policy in policy_results
    }
    summary_rows: list[dict[str, object]] = []
    baseline_metrics = metrics(grouped["kubernetes-default"])
    for policy in policy_results:
        values = metrics(grouped[policy])
        summary_rows.append(
            {
                "policy": policy,
                **{key: round(value, 6) for key, value in values.items()},
                "average_jct_improvement_vs_default_percent": round(
                    100
                    * (
                        baseline_metrics["average_jct_seconds"]
                        - values["average_jct_seconds"]
                    )
                    / baseline_metrics["average_jct_seconds"],
                    6,
                ),
                "makespan_improvement_vs_default_percent": round(
                    100
                    * (
                        baseline_metrics["makespan_seconds"]
                        - values["makespan_seconds"]
                    )
                    / baseline_metrics["makespan_seconds"],
                    6,
                ),
                "evidence_status": "synthetic-estimate-not-executed-on-kubernetes",
            }
        )

    metadata = {
        "schema_version": "1.0",
        "title": "KubeML 200-workload synthetic forecast",
        "evidence_status": "SYNTHETIC ESTIMATE — NOT EXECUTED ON KUBERNETES",
        "jobs": args.jobs,
        "seed": args.seed,
        "load_profile": "heavy training-only ML proxy matching the archived pilot",
        "node_model": {
            "cpu_cores": 4,
            "simulator_dt_seconds": 0.10,
            "default_mean_inter_launch_proxy_seconds": 1.0,
            "ranked_inherent_gap_proxy_seconds": 1.0,
        },
        "calibration": {
            "source": "archived five-run pilot baseline mean JCT",
            "pilot_baseline_average_jct_seconds": pilot_baseline_avg_jct,
            "reference_proxy_average_jct_seconds": reference_avg_jct,
            "global_time_scale": time_scale,
            "policy_effects_fitted": False,
        },
        "category_counts": dict(Counter(str(row["category"]) for row in workload_rows)),
        "limitations": [
            "This is a deterministic proxy forecast, not a Kubernetes experiment.",
            "The global time unit is calibrated to five archived pilot runs.",
            "Policy differences are simulator outputs and must be replaced by live measurements.",
        ],
    }

    write_csv(output / "workloads-200.csv", workload_rows)
    write_csv(output / "estimated-results-200.csv", result_rows)
    write_csv(output / "estimated-summary-200.csv", summary_rows)
    atomic_json(
        output / "workloads-200.json",
        {"metadata": metadata, "workloads": workload_rows},
    )
    atomic_json(
        output / "estimated-results-200.json",
        {"metadata": metadata, "summary": summary_rows, "results": result_rows},
    )

    policies = list(policy_results)
    labels = {
        "kubernetes-default": "Kubernetes default",
        "duration-only": "Duration only",
        "six-feature": "Six feature",
        "kubeml-fastpath-tail": "KubeML + FastPath + tail",
        "reversed-ablation": "Reversed ablation",
    }
    colors = ["#667085", "#f79009", "#1570ef", "#12b76a", "#f04438"]
    summary_by_policy = {str(row["policy"]): row for row in summary_rows}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.7), constrained_layout=True)
    metrics_to_plot = (
        ("average_jct_seconds", "Estimated average JCT", "Seconds"),
        ("p95_jct_seconds", "Estimated p95 JCT", "Seconds"),
        ("makespan_seconds", "Estimated makespan", "Seconds"),
    )
    for axis, (metric, title, unit) in zip(axes, metrics_to_plot):
        values = [float(summary_by_policy[policy][metric]) for policy in policies]
        bars = axis.bar(range(len(policies)), values, color=colors)
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.set_xticks(range(len(policies)), [labels[p] for p in policies], rotation=24, ha="right")
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle("SYNTHETIC ESTIMATE — 200 heavy ML workloads (not a live Kubernetes run)")
    fig.savefig(output / "estimated-metrics-200.png", dpi=180)
    fig.savefig(output / "estimated-metrics-200.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    for policy, color in zip(policies, colors):
        jcts = sorted(float(row["jct_seconds"]) for row in grouped[policy])
        ecdf = np.arange(1, len(jcts) + 1) / len(jcts)
        axes[0].plot(jcts, ecdf, label=labels[policy], color=color, linewidth=2)
        completions = sorted(float(row["completion_seconds"]) for row in grouped[policy])
        axes[1].plot(
            range(1, len(completions) + 1),
            completions,
            label=labels[policy],
            color=color,
            linewidth=2,
        )
    axes[0].set_title("Estimated JCT distribution")
    axes[0].set_xlabel("JCT (seconds)")
    axes[0].set_ylabel("ECDF")
    axes[1].set_title("Estimated completion curve")
    axes[1].set_xlabel("Completed workloads")
    axes[1].set_ylabel("Seconds since burst submission")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("SYNTHETIC ESTIMATE — replace with live cluster measurements")
    fig.savefig(output / "estimated-distributions-200.png", dpi=180)
    fig.savefig(output / "estimated-distributions-200.pdf")
    plt.close(fig)

    feature_names = (
        ("T_estimated_training_seconds", "T: estimated duration"),
        ("R_loss_reduction_rate", "R: loss-reduction rate"),
        ("M_matrix_dimension", "M: matrix dimension"),
        ("G_gradient_update_mib", "G: gradient update (MiB)"),
        ("C_checkpoint_interval_steps", "C: checkpoint interval"),
        ("P_model_partitions", "P: model partitions"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, (field, title) in zip(axes.flat, feature_names):
        values = [float(row[field]) for row in workload_rows]
        axis.hist(values, bins=18, color="#1570ef", alpha=0.85)
        axis.set_title(title)
        axis.set_ylabel("Workloads")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Generated feature distributions — 200 synthetic ML workloads")
    fig.savefig(output / "workload-features-200.png", dpi=180)
    fig.savefig(output / "workload-features-200.pdf")
    plt.close(fig)

    readme = f"""# خروجی حدسی ۲۰۰ workload

**وضعیت شواهد:** حدسی و شبیه‌سازی‌شده؛ روی Kubernetes اجرا نشده است.

- تعداد workload: {args.jobs}
- seed: `{args.seed}`
- مدل گره: ۴ هسته CPU
- کالیبراسیون زمانی: میانگین JCT پایه در پنج اجرای واقعی قبلی
- تفاوت سیاست‌ها: خروجی شبیه‌ساز است و روی داده واقعی fit نشده است

فایل `workloads-200.csv` ورودی دقیق ۲۰۰ workload را نگه می‌دارد. فایل
`estimated-results-200.csv` شامل خروجی حدسی هر workload برای پنج سیاست است.
اعداد باید پس از اجرای کلاستر با نتایج واقعی جایگزین شوند.
"""
    (output / "README-FA.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": summary_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
