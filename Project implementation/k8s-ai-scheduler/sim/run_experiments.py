"""Run reproducible proxy experiments using the shared result schema.

The generated run JSON files have the same contract as real Kubernetes
collections.  CSVs are derived views for plotting; JSON remains authoritative.
All contention and trainer-cost constants are CLI parameters so sensitivity
analysis can distinguish conclusions from simulator assumptions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(ROOT / "workload"))
sys.path.append(str(ROOT))
from generate_workload import category_for_job, generate_burst  # noqa: E402

from experiments.schema import (  # noqa: E402
    RESULT_SCHEMA_VERSION,
    make_result_document,
    summarize_jobs,
    validate_result_document,
)
from experiments.statistics import paired_improvement_table  # noqa: E402
from sim.calibration import load_calibrated_model  # noqa: E402
from sim.simulate import (  # noqa: E402
    SimResult,
    TrainerWorkModel,
    results_to_dicts,
    run_default,
    run_paced,
)

N_CORES = 4
INHERENT_GAP = 1.0
N_REPEATS = 5
RESULTS_DIR = ROOT / "results"
SCENARIOS = [
    ("12-normal", 12, "normal"),
    ("48-normal", 48, "normal"),
    ("48-half", 48, "half"),
]
PACING_CONFIGS = [
    ("custom-baseline", 0.0),
    ("custom-delay-1s", 1.0),
    ("custom-delay-2s", 2.0),
    ("custom-delay-5s", 5.0),
]


@dataclass(frozen=True)
class SimulationSettings:
    n_cores: int = N_CORES
    dt: float = 0.05
    alpha: float = 0.05
    inherent_gap: float = INHERENT_GAP
    default_mean_ilt: float = 0.67
    cpu_threshold: float = 0.85
    max_wait: float = 6.0
    max_time: float = 100000.0
    work_model: TrainerWorkModel = TrainerWorkModel()

    def validate(self) -> None:
        if self.n_cores <= 0:
            raise ValueError("n_cores must be > 0")
        for name in ("dt", "max_time"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0")
        for name in ("inherent_gap", "default_mean_ilt"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0")
        if not math.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be finite and >= 0")
        if not 0 < self.cpu_threshold <= 1:
            raise ValueError("cpu_threshold must be in (0, 1]")
        if not math.isfinite(self.max_wait) or self.max_wait < 0:
            raise ValueError("max_wait must be finite and >= 0")
        self.work_model.validate()


def metrics_from_results(results: Sequence[SimResult]) -> Dict[str, Any]:
    return summarize_jobs(results_to_dicts(list(results)))


def run_scenario(n_jobs: int, load: str, seed: int):
    return generate_burst(n_jobs, seed=seed, load=load)


def _simulate_config(
    jobs,
    config_name: str,
    *,
    seed: int,
    settings: SimulationSettings,
) -> List[SimResult]:
    common = {
        "n_cores": settings.n_cores,
        "dt": settings.dt,
        "alpha": settings.alpha,
        "max_time": settings.max_time,
        "work_model": settings.work_model,
    }
    if config_name == "default":
        return run_default(
            jobs,
            seed=seed,
            mean_ilt=settings.default_mean_ilt,
            **common,
        )
    if config_name == "custom-baseline":
        return run_paced(jobs, mode="fixed", inherent_gap=settings.inherent_gap, **common)
    if config_name == "reversed":
        return run_paced(
            jobs, mode="fixed", inherent_gap=settings.inherent_gap, reverse=True, **common,
        )
    if config_name == "custom-adaptive":
        return run_paced(
            jobs,
            mode="adaptive",
            inherent_gap=settings.inherent_gap,
            cpu_threshold=settings.cpu_threshold,
            max_wait=settings.max_wait,
            **common,
        )
    for known_name, delay in PACING_CONFIGS:
        if config_name == known_name:
            return run_paced(
                jobs,
                mode="fixed",
                inherent_gap=settings.inherent_gap,
                extra_delay=delay,
                **common,
            )
    raise ValueError(f"unknown simulation config {config_name!r}")


def _result_document(
    results: Sequence[SimResult],
    *,
    jobs,
    run_id: str,
    scenario: str,
    config_name: str,
    repetition: int,
    seed: int,
    settings: SimulationSettings,
) -> Dict[str, Any]:
    job_by_id = {job.job_id: job for job in jobs}
    rows = results_to_dicts(list(results))
    for row in rows:
        row["category"] = category_for_job(job_by_id[row["job_id"]])
    document = make_result_document(
        run_id=run_id,
        scenario=scenario,
        config=config_name,
        repetition=repetition,
        seed=seed,
        expected_jobs=len(jobs),
        jobs=rows,
        environment={
            "simulation": {
                **{key: value for key, value in asdict(settings).items() if key != "work_model"},
                "work_model": asdict(settings.work_model),
                "calibration_status": (
                    "hardware-calibrated"
                    if settings.work_model.calibrated
                    else "uncalibrated-assumptions"
                ),
                "claim_eligibility": "exploratory-only",
                "warning": "Proxy model; final claims require real Kubernetes measurements.",
            }
        },
        source="simulation",
    )
    validate_result_document(document, strict=True)
    return document


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _flatten_document(document: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    run = document["run"]
    summary = document["summary"]
    summary_row = {
        "schema_version": document["schema_version"],
        "run_id": run["run_id"],
        "scenario": run["scenario"],
        "config": run["config"],
        "rep": run["repetition"],
        "seed": run["seed"],
        "expected_jobs": run["expected_jobs"],
        **summary,
    }
    raw_rows: List[Dict[str, Any]] = []
    for job in document["jobs"]:
        raw_rows.append({
            "schema_version": document["schema_version"],
            "run_id": run["run_id"],
            "scenario": run["scenario"],
            "config": run["config"],
            "rep": run["repetition"],
            "seed": run["seed"],
            "job_id": job["job_id"],
            "category": job["category"],
            **{f"feature_{name}": job["features"][name] for name in ("T", "R", "M", "G", "C", "P")},
            "rank": job["rank"],
            "submission_time": job["submission_time"],
            "execution_start_time": job["execution_start_time"],
            "completion_time": job["completion_time"],
            "jct": job["jct_s"],
            "status": job["status"],
            "node_name": job["node_name"],
            "error": job["error"],
        })
    return summary_row, raw_rows


def _execute_run(
    *,
    scenario: str,
    n_jobs: int,
    load: str,
    config_name: str,
    repetition: int,
    seed: int,
    settings: SimulationSettings,
    run_dir: Path,
) -> Dict[str, Any]:
    jobs = run_scenario(n_jobs, load, seed)
    run_id = f"sim-{scenario}-{config_name}-r{repetition}"
    results = _simulate_config(jobs, config_name, seed=seed, settings=settings)
    document = _result_document(
        results,
        jobs=jobs,
        run_id=run_id,
        scenario=scenario,
        config_name=config_name,
        repetition=repetition,
        seed=seed,
        settings=settings,
    )
    _atomic_json(run_dir / f"{run_id}.json", document)
    return document


def pacing_sweep(
    *,
    settings: Optional[SimulationSettings] = None,
    results_dir: Path | str = RESULTS_DIR,
    include_adaptive: bool = True,
    repeats: int = N_REPEATS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    settings = settings or SimulationSettings()
    settings.validate()
    output = Path(results_dir)
    configs = ["default", *[name for name, _ in PACING_CONFIGS]]
    if include_adaptive:
        configs.append("custom-adaptive")
    summary_rows: List[Dict[str, Any]] = []
    run_dir = output / "simulation_runs" / "pacing"
    for repetition in range(repeats):
        seed = 1000 + repetition
        for config_name in configs:
            document = _execute_run(
                scenario="48-half-pacing",
                n_jobs=48,
                load="half",
                config_name=config_name,
                repetition=repetition,
                seed=seed,
                settings=settings,
                run_dir=run_dir,
            )
            summary_rows.append(_flatten_document(document)[0])
    dataframe = pd.DataFrame(summary_rows)
    output.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output / "pacing_sweep.csv", index=False)
    summary = dataframe.groupby("config")[["avg_ilt", "avg_jct", "makespan"]].mean()
    return dataframe, summary


def main_comparison(
    *,
    settings: Optional[SimulationSettings] = None,
    results_dir: Path | str = RESULTS_DIR,
    include_adaptive: bool = True,
    repeats: int = N_REPEATS,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    settings = settings or SimulationSettings()
    settings.validate()
    output = Path(results_dir)
    configs = ["default", "custom-baseline", "reversed"]
    if include_adaptive:
        configs.insert(2, "custom-adaptive")
    summary_rows: List[Dict[str, Any]] = []
    raw_rows: List[Dict[str, Any]] = []
    documents: Dict[str, Any] = {}
    for scenario_index, (scenario, n_jobs, load) in enumerate(SCENARIOS):
        for repetition in range(repeats):
            seed = 2000 + scenario_index * 100 + repetition
            for config_name in configs:
                document = _execute_run(
                    scenario=scenario,
                    n_jobs=n_jobs,
                    load=load,
                    config_name=config_name,
                    repetition=repetition,
                    seed=seed,
                    settings=settings,
                    run_dir=output / "simulation_runs" / "main",
                )
                summary_row, job_rows = _flatten_document(document)
                summary_rows.append(summary_row)
                raw_rows.extend(job_rows)
                documents[document["run"]["run_id"]] = document

    dataframe = pd.DataFrame(summary_rows)
    raw_dataframe = pd.DataFrame(raw_rows)
    output.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output / "main_comparison.csv", index=False)
    raw_dataframe.to_csv(output / "raw_jobs.csv", index=False)
    summary = dataframe.groupby(["scenario", "config"])[
        ["avg_jct", "tail_jct_p95", "max_jct", "min_jct", "makespan", "avg_ilt"]
    ].mean()

    paired = paired_improvement_table(dataframe)
    paired.to_csv(output / "paired_improvements.csv", index=False)
    improvements: Dict[str, Any] = {
        "method": "within-scenario repetition/seed paired differences with Student-t 95% CI",
        "comparisons": paired.to_dict(orient="records"),
    }
    summary_document = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "kind": "ml-scheduler-experiment-summary",
        "source": "simulation",
        "settings": {**asdict(settings), "work_model": asdict(settings.work_model)},
        "claim_eligibility": "exploratory-only",
        "improvements": improvements,
    }
    _atomic_json(output / "summary_metrics.json", summary_document)
    return dataframe, summary, documents, improvements


def sensitivity_analysis(
    *,
    settings: SimulationSettings,
    alphas: Sequence[float],
    time_steps: Sequence[float],
    results_dir: Path | str,
) -> pd.DataFrame:
    """Run a compact 48-half baseline grid over model-sensitive parameters."""
    rows: List[Dict[str, Any]] = []
    for alpha in alphas:
        for dt in time_steps:
            candidate = SimulationSettings(**{
                **asdict(settings),
                "alpha": alpha,
                "dt": dt,
                "work_model": settings.work_model,
            })
            candidate.validate()
            jobs = generate_burst(48, seed=9000, load="half")
            for config_name in ("default", "custom-baseline"):
                result = _simulate_config(jobs, config_name, seed=9000, settings=candidate)
                rows.append({"alpha": alpha, "dt": dt, "config": config_name, **metrics_from_results(result)})
    dataframe = pd.DataFrame(rows)
    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output / "sensitivity.csv", index=False)
    return dataframe


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--without-adaptive", action="store_true")
    parser.add_argument("--n-cores", type=int, default=N_CORES)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--inherent-gap", type=float, default=INHERENT_GAP)
    parser.add_argument("--default-mean-ilt", type=float, default=0.67)
    parser.add_argument("--cpu-threshold", type=float, default=0.85)
    parser.add_argument("--max-wait", type=float, default=6.0)
    parser.add_argument("--max-time", type=float, default=100000.0)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--require-calibration", action="store_true")
    parser.add_argument("--matrix-reference", type=float)
    parser.add_argument("--matmul-seconds-at-reference", type=float)
    parser.add_argument("--gradient-scale", type=float)
    parser.add_argument("--synchronization-scale", type=float)
    parser.add_argument("--checkpoint-scale", type=float)
    parser.add_argument("--estimated-time-weight", type=float)
    parser.add_argument("--sensitivity-alpha", type=float, nargs="*")
    parser.add_argument("--sensitivity-dt", type=float, nargs="*")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    if args.repeats <= 0:
        raise ValueError("--repeats must be > 0")
    manual_model_values = {
        "matrix_reference": args.matrix_reference,
        "matmul_seconds_at_reference": args.matmul_seconds_at_reference,
        "gradient_scale": args.gradient_scale,
        "synchronization_scale": args.synchronization_scale,
        "checkpoint_scale": args.checkpoint_scale,
        "estimated_time_weight": args.estimated_time_weight,
    }
    if args.calibration:
        supplied_overrides = [
            name for name, value in manual_model_values.items() if value is not None
        ]
        if supplied_overrides:
            raise ValueError(
                "--calibration cannot be combined with manual model overrides: "
                + ", ".join(supplied_overrides)
            )
        work_model = load_calibrated_model(args.calibration)
    else:
        if args.require_calibration:
            raise ValueError("--require-calibration requires --calibration")
        defaults = TrainerWorkModel()
        work_model = TrainerWorkModel(**{
            name: getattr(defaults, name) if value is None else value
            for name, value in manual_model_values.items()
        })
    settings = SimulationSettings(
        n_cores=args.n_cores,
        dt=args.dt,
        alpha=args.alpha,
        inherent_gap=args.inherent_gap,
        default_mean_ilt=args.default_mean_ilt,
        cpu_threshold=args.cpu_threshold,
        max_wait=args.max_wait,
        max_time=args.max_time,
        work_model=work_model,
    )
    settings.validate()
    include_adaptive = not args.without_adaptive
    pacing_df, pacing_summary = pacing_sweep(
        settings=settings,
        results_dir=args.results_dir,
        include_adaptive=include_adaptive,
        repeats=args.repeats,
    )
    main_df, main_summary, _, improvements = main_comparison(
        settings=settings,
        results_dir=args.results_dir,
        include_adaptive=include_adaptive,
        repeats=args.repeats,
    )
    print("\n=== Pacing sweep ===")
    print(pacing_summary.round(3).to_string())
    print("\n=== Main comparison ===")
    print(main_summary.round(3).to_string())
    if improvements:
        print("\n=== Paired improvement estimates ===")
        print(json.dumps(improvements, indent=2))
    if args.sensitivity_alpha or args.sensitivity_dt:
        sensitivity_analysis(
            settings=settings,
            alphas=args.sensitivity_alpha or [settings.alpha],
            time_steps=args.sensitivity_dt or [settings.dt],
            results_dir=args.results_dir,
        )
    print(f"\nResults saved under {args.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
