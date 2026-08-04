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


def paired_effect_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Compute the article's paired scheduler and reversed-order effects.

    Custom schedulers are improvements relative to ``default`` (reference
    minus comparison for lower-is-better metrics).  The reversed ablation is a
    degradation relative to the intended ``custom-baseline`` order (comparison
    minus reference), matching the wording and denominator in the article.
    Every effect is positive when it supports the corresponding claim.
    """

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
        configurations = set(scenario_rows.config)
        comparisons = []
        for config in sorted(configurations - {"default"}):
            if config == "reversed":
                comparisons.append(
                    (config, "custom-baseline", "degradation_vs_intended", 1.0)
                )
            else:
                comparisons.append((config, "default", "improvement_vs_default", -1.0))
        for config, reference_config, effect_kind, direction in comparisons:
            if reference_config not in configurations:
                raise ValueError(
                    f"scenario {scenario!r} requires reference {reference_config!r} "
                    f"for comparison {config!r}"
                )
            reference = scenario_rows[scenario_rows.config == reference_config]
            comparison = scenario_rows[scenario_rows.config == config]
            pair_keys = ["rep", "seed"]
            if reference.duplicated(pair_keys).any():
                raise ValueError(
                    f"scenario {scenario!r} has duplicate {reference_config!r} pairs"
                )
            if comparison.duplicated(pair_keys).any():
                raise ValueError(
                    f"scenario {scenario!r} config {config!r} has duplicate pairs"
                )
            reference_keys = set(map(tuple, reference[pair_keys].to_numpy().tolist()))
            comparison_keys = set(map(tuple, comparison[pair_keys].to_numpy().tolist()))
            if reference_keys != comparison_keys:
                raise ValueError(
                    f"scenario {scenario!r} comparison {config!r} has unpaired runs: "
                    f"reference_only={sorted(reference_keys - comparison_keys)}, "
                    f"comparison_only={sorted(comparison_keys - reference_keys)}"
                )
            paired = reference.merge(
                comparison,
                on=["scenario", "rep", "seed"],
                how="inner",
                suffixes=("_reference", "_comparison"),
                validate="one_to_one",
            )
            if paired.empty:
                continue
            row: dict[str, Any] = {
                "scenario": scenario,
                "config": config,
                "reference_config": reference_config,
                "effect_kind": effect_kind,
                "pairs": len(paired),
                "reference_runs": len(reference),
                "comparison_runs": len(comparison),
            }
            for metric in ("avg_jct", "makespan"):
                reference_values = paired[f"{metric}_reference"]
                comparison_values = paired[f"{metric}_comparison"]
                if (reference_values <= 0).any():
                    raise ValueError(
                        f"paired {metric} reference must be > 0 for {scenario}/{config}"
                    )
                effects = (direction * (comparison_values - reference_values)).to_list()
                percentages = (
                    100.0 * direction * (comparison_values - reference_values) / reference_values
                ).to_list()
                mean, lower, upper = mean_ci95(effects)
                row[f"{metric}_effect_mean"] = mean
                row[f"{metric}_effect_ci95_low"] = lower
                row[f"{metric}_effect_ci95_high"] = upper
                mean, lower, upper = mean_ci95(percentages)
                row[f"{metric}_effect_pct_mean"] = mean
                row[f"{metric}_effect_pct_ci95_low"] = lower
                row[f"{metric}_effect_pct_ci95_high"] = upper
            rows.append(row)
    return pd.DataFrame(rows)
