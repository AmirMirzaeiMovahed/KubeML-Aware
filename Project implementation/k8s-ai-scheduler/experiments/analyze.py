"""Strictly validate and analyze completed cluster result documents."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "sim"))
from experiments.run_cluster import (  # noqa: E402
    DEFAULT_PLAN,
    expand_plan,
    plan_document,
    validate_result_for_spec,
)
from experiments.schema import RESULT_SCHEMA_VERSION, validate_result_document  # noqa: E402
from experiments.statistics import (  # noqa: E402
    mean_ci95 as _mean_ci95,
    paired_improvement_table,
)
from plot_results import plot_pacing_sweep, plot_scenario  # noqa: E402


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_documents(directory: Path) -> Dict[str, Dict[str, Any]]:
    documents: Dict[str, Dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        validate_result_document(document, strict=True)
        run_id = document["run"]["run_id"]
        if run_id in documents:
            raise ValueError(f"duplicate run_id {run_id!r} in {directory}")
        documents[run_id] = document
    return documents


def validate_plan_coverage(
    documents: Mapping[str, Mapping[str, Any]],
    *,
    plan_path: Path,
    include_adaptive: bool,
    allow_partial: bool,
) -> List[str]:
    expanded_specs = expand_plan(plan_path, include_adaptive=include_adaptive)
    expected = {spec.run_id: spec for spec in expanded_specs}
    registered_plan = plan_document(
        expanded_specs,
        include_adaptive=include_adaptive,
        source_plan=plan_path,
    )
    actual_ids = set(documents)
    expected_ids = set(expected)
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    if (missing and not allow_partial) or extra:
        raise ValueError(f"result coverage mismatch: missing={missing}, extra={extra}")
    for run_id, document in documents.items():
        spec = expected[run_id]
        orchestration = (document.get("environment") or {}).get("orchestration") or {}
        target_node = orchestration.get("target_node")
        if not isinstance(target_node, str) or not target_node:
            raise ValueError(f"run {run_id} is missing target-node evidence")
        validate_result_for_spec(
            document,
            spec,
            plan_sha256=registered_plan["plan_sha256"],
            target_node=target_node,
            require_article_environment=True,
        )
        run = document["run"]
        mismatches = {
            "scenario": (run["scenario"], spec.scenario),
            "config": (run["config"], spec.config),
            "repetition": (run["repetition"], spec.repetition),
            "seed": (run["seed"], spec.seed),
            "expected_jobs": (run["expected_jobs"], spec.jobs),
        }
        invalid = {key: values for key, values in mismatches.items() if values[0] != values[1]}
        if invalid:
            raise ValueError(f"run {run_id} does not match registered plan: {invalid}")
    return missing


def flatten_documents(documents: Mapping[str, Mapping[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict[str, Any]] = []
    job_rows: List[Dict[str, Any]] = []
    for document in documents.values():
        run = document["run"]
        summary_rows.append({
            "schema_version": document["schema_version"],
            "source": document["source"],
            "run_id": run["run_id"],
            "scenario": run["scenario"],
            "config": run["config"],
            "rep": run["repetition"],
            "seed": run["seed"],
            "expected_jobs": run["expected_jobs"],
            **document["summary"],
        })
        for job in document["jobs"]:
            job_rows.append({
                "schema_version": document["schema_version"],
                "source": document["source"],
                "run_id": run["run_id"],
                "scenario": run["scenario"],
                "config": run["config"],
                "rep": run["repetition"],
                "seed": run["seed"],
                "job_id": job["job_id"],
                "category": job.get("category"),
                **{f"feature_{feature}": job["features"][feature] for feature in ("T", "R", "M", "G", "C", "P")},
                "rank": job["rank"],
                "submission_time": job["submission_time"],
                "execution_start_time": job["execution_start_time"],
                "completion_time": job["completion_time"],
                "jct": job["jct_s"],
                "status": job["status"],
                "node_name": job.get("node_name"),
                "error": job.get("error"),
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(job_rows)


def aggregate_run_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = ["avg_jct", "tail_jct_p95", "max_jct", "min_jct", "makespan", "avg_ilt"]
    rows: List[Dict[str, Any]] = []
    for (scenario, config), group in summary.groupby(["scenario", "config"], sort=True):
        row: Dict[str, Any] = {
            "scenario": scenario,
            "config": config,
            "repetitions": group.rep.nunique(),
        }
        for metric in metrics:
            mean, lower, upper = _mean_ci95(group[metric].to_list())
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_low"] = lower
            row[f"{metric}_ci95_high"] = upper
        rows.append(row)
    return pd.DataFrame(rows)


def improvement_table(aggregated: pd.DataFrame) -> List[Dict[str, Any]]:
    """Legacy ratio-of-means view retained for CSV compatibility."""
    comparisons: List[Dict[str, Any]] = []
    for scenario in sorted(set(aggregated.scenario) - {"48-half-pacing"}):
        subset = aggregated[aggregated.scenario == scenario].set_index("config")
        if "default" not in subset.index:
            continue
        baseline = subset.loc["default"]
        for config in ("custom-baseline", "custom-adaptive", "reversed"):
            if config not in subset.index:
                continue
            candidate = subset.loc[config]
            comparisons.append({
                "scenario": scenario,
                "config": config,
                "avg_jct_vs_default_pct": 100 * (
                    baseline.avg_jct_mean - candidate.avg_jct_mean
                ) / baseline.avg_jct_mean,
                "makespan_vs_default_pct": 100 * (
                    baseline.makespan_mean - candidate.makespan_mean
                ) / baseline.makespan_mean,
            })
    return comparisons


def analyze(
    *,
    runs_dir: Path,
    output_dir: Path,
    plan_path: Path = DEFAULT_PLAN,
    include_adaptive: bool = False,
    allow_partial: bool = False,
    make_plots: bool = True,
) -> Dict[str, Any]:
    documents = load_documents(runs_dir)
    if not documents:
        raise ValueError(f"no run JSON documents found in {runs_dir}")
    missing = validate_plan_coverage(
        documents,
        plan_path=plan_path,
        include_adaptive=include_adaptive,
        allow_partial=allow_partial,
    )
    summary, jobs = flatten_documents(documents)
    aggregate = aggregate_run_metrics(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "all_runs.csv", index=False)
    jobs.to_csv(output_dir / "all_jobs.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    paired_improvements = paired_improvement_table(summary)
    paired_improvements.to_csv(output_dir / "paired_improvements.csv", index=False)

    pacing = summary[summary.scenario == "48-half-pacing"]
    main = summary[summary.scenario != "48-half-pacing"]
    pacing.to_csv(output_dir / "pacing_sweep.csv", index=False)
    main.to_csv(output_dir / "main_comparison.csv", index=False)
    jobs[jobs.scenario != "48-half-pacing"].to_csv(output_dir / "raw_jobs.csv", index=False)

    improvements = improvement_table(aggregate)
    report = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "kind": "ml-scheduler-cluster-analysis",
        "run_count": len(documents),
        "expected_run_count": 90 if include_adaptive else 70,
        "plan_sha256": plan_document(
            expand_plan(plan_path, include_adaptive=include_adaptive),
            include_adaptive=include_adaptive,
            source_plan=plan_path,
        )["plan_sha256"],
        "missing_run_ids": missing,
        "strictly_complete": not missing,
        "improvements": improvements,
        "paired_improvements": paired_improvements.to_dict(orient="records"),
        "pairing_keys": ["scenario", "repetition", "seed"],
        "confidence_interval": (
            "Two-sided Student-t 95% CI over run-level metrics; scheduler "
            "effects use within-scenario repetition/seed paired differences."
        ),
        "ecdf_method": "Per-run ECDFs evaluated on a common grid; mean curve and 25th–75th percentile band across runs.",
    }
    _atomic_json(output_dir / "analysis.json", report)

    if make_plots:
        main_jobs = jobs[jobs.scenario != "48-half-pacing"]
        for scenario in sorted(main.scenario.unique()):
            plot_scenario(scenario, main, main_jobs, output_dir / f"scenario_{scenario}.png")
        plot_pacing_sweep(pacing, output_dir / "pacing_sweep.png")
    return report


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "results" / "cluster" / "runs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "cluster" / "analysis")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--include-adaptive", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = analyze(
        runs_dir=args.runs_dir,
        output_dir=args.output_dir,
        plan_path=args.plan,
        include_adaptive=args.include_adaptive,
        allow_partial=args.allow_partial,
        make_plots=not args.no_plots,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
