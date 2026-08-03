"""Run-level confidence intervals and registered-pair scheduler effects."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Any

import pandas as pd

_T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def mean_ci95(values: Sequence[float]) -> tuple[float, float, float]:
    """Two-sided Student-t 95% CI across independent run-level values."""

    numeric = [float(value) for value in values]
    if not numeric or any(not math.isfinite(value) for value in numeric):
        raise ValueError("confidence interval values must be non-empty and finite")
    mean = statistics.fmean(numeric)
    if len(numeric) < 2:
        return mean, mean, mean
    degrees_of_freedom = len(numeric) - 1
    critical = _T_CRITICAL_975.get(degrees_of_freedom, 1.96)
    margin = critical * statistics.stdev(numeric) / math.sqrt(len(numeric))
    return mean, mean - margin, mean + margin


def paired_improvement_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Compute positive-is-better baseline-minus-candidate effects by seed pair."""

    required = {
        "scenario",
        "config",
        "rep",
        "seed",
        "avg_jct",
        "makespan",
    }
    missing_columns = sorted(required - set(summary.columns))
    if missing_columns:
        raise ValueError(f"paired analysis is missing columns: {missing_columns}")

    rows: list[dict[str, Any]] = []
    main = summary[summary.scenario != "48-half-pacing"]
    for scenario in sorted(main.scenario.unique()):
        scenario_rows = main[main.scenario == scenario]
        baseline = scenario_rows[scenario_rows.config == "default"]
        if baseline.empty:
            continue
        if baseline.duplicated(["rep", "seed"]).any():
            raise ValueError(f"scenario {scenario!r} has duplicate default pairs")
        candidates = sorted(set(scenario_rows.config) - {"default"})
        for config in candidates:
            candidate = scenario_rows[scenario_rows.config == config]
            if candidate.duplicated(["rep", "seed"]).any():
                raise ValueError(
                    f"scenario {scenario!r} config {config!r} has duplicate pairs"
                )
            paired = baseline.merge(
                candidate,
                on=["scenario", "rep", "seed"],
                how="inner",
                suffixes=("_default", "_candidate"),
                validate="one_to_one",
            )
            if paired.empty:
                continue
            row: dict[str, Any] = {
                "scenario": scenario,
                "config": config,
                "baseline_config": "default",
                "pairs": len(paired),
                "baseline_runs": len(baseline),
                "candidate_runs": len(candidate),
            }
            for metric in ("avg_jct", "makespan"):
                baseline_values = paired[f"{metric}_default"]
                if (baseline_values <= 0).any():
                    raise ValueError(
                        f"paired {metric} baseline must be > 0 for {scenario}/{config}"
                    )
                differences = (
                    baseline_values - paired[f"{metric}_candidate"]
                ).to_list()
                percentages = (
                    100.0
                    * (baseline_values - paired[f"{metric}_candidate"])
                    / baseline_values
                ).to_list()
                mean, lower, upper = mean_ci95(differences)
                row[f"{metric}_difference_mean"] = mean
                row[f"{metric}_difference_ci95_low"] = lower
                row[f"{metric}_difference_ci95_high"] = upper
                mean, lower, upper = mean_ci95(percentages)
                row[f"{metric}_improvement_pct_mean"] = mean
                row[f"{metric}_improvement_pct_ci95_low"] = lower
                row[f"{metric}_improvement_pct_ci95_high"] = upper
            rows.append(row)
    return pd.DataFrame(rows)
