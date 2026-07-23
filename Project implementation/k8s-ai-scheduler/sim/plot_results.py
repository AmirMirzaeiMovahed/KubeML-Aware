"""Plot run-level mean ECDFs with inter-run IQR uncertainty bands."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
CONFIGS = ["default", "custom-baseline", "custom-adaptive", "reversed"]
COLORS = {
    "default": "#e67e22",
    "custom-baseline": "#3498db",
    "custom-adaptive": "#2ecc71",
    "reversed": "#c0392b",
}
LABELS = {
    "default": "Default Scheduler",
    "custom-baseline": "Custom (article baseline, delay=0s)",
    "custom-adaptive": "Custom + Adaptive Pacing",
    "reversed": "Reversed order (ablation)",
}


def ecdf(values: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    array = np.sort(np.asarray(list(values), dtype=float))
    if array.size == 0:
        raise ValueError("ECDF requires at least one value")
    if not np.all(np.isfinite(array)):
        raise ValueError("ECDF values must be finite")
    return array, np.arange(1, array.size + 1, dtype=float) / array.size


def mean_ecdf_iqr(
    frame: pd.DataFrame,
    *,
    value_column: str = "jct",
    repetition_column: str = "rep",
    grid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate each run's ECDF on one grid, then aggregate across runs."""
    required = {value_column, repetition_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing ECDF columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("mean ECDF requires at least one row")
    values = frame[value_column].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("ECDF values must be finite")
    if grid is None:
        grid = np.unique(np.sort(values))
    else:
        grid = np.asarray(grid, dtype=float)
        if grid.ndim != 1 or grid.size == 0 or not np.all(np.isfinite(grid)):
            raise ValueError("grid must be a non-empty finite one-dimensional array")

    run_cdfs = []
    for _, group in frame.groupby(repetition_column, sort=True):
        run_values = np.sort(group[value_column].to_numpy(dtype=float))
        run_cdfs.append(np.searchsorted(run_values, grid, side="right") / run_values.size)
    matrix = np.asarray(run_cdfs)
    return (
        grid,
        np.mean(matrix, axis=0),
        np.quantile(matrix, 0.25, axis=0, method="linear"),
        np.quantile(matrix, 0.75, axis=0, method="linear"),
    )


def plot_scenario(
    scenario_name: str,
    main_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    out_path: str | os.PathLike[str],
) -> None:
    fig, (ax_ecdf, ax_bar) = plt.subplots(1, 2, figsize=(13, 5))
    summary = main_df[main_df.scenario == scenario_name]
    raw = raw_df[(raw_df.scenario == scenario_name) & (raw_df.status == "completed")]
    available_configs = [config for config in CONFIGS if config in set(raw.config)]
    for config in available_configs:
        config_rows = raw[raw.config == config]
        grid, mean, lower, upper = mean_ecdf_iqr(config_rows)
        ax_ecdf.step(grid, mean, where="post", label=LABELS[config], color=COLORS[config], linewidth=2)
        ax_ecdf.fill_between(grid, lower, upper, step="post", color=COLORS[config], alpha=0.16)
    ax_ecdf.set_xlabel("Job Completion Time (s)")
    ax_ecdf.set_ylabel("Mean CDF")
    ax_ecdf.set_ylim(0, 1.02)
    ax_ecdf.set_title(f"Run-level mean ECDF with IQR — {scenario_name}")
    ax_ecdf.legend(fontsize=8)
    ax_ecdf.grid(alpha=0.3)

    metrics = ["avg_jct", "tail_jct_p95", "max_jct", "min_jct", "makespan"]
    means = summary.groupby("config")[metrics].mean().reindex(available_configs)
    x = np.arange(len(metrics))
    width = 0.8 / max(1, len(available_configs))
    for index, config in enumerate(available_configs):
        ax_bar.bar(
            x + index * width,
            means.loc[config, metrics].values,
            width,
            label=LABELS[config],
            color=COLORS[config],
        )
    ax_bar.set_xticks(x + (len(available_configs) - 1) * width / 2)
    ax_bar.set_xticklabels(["Avg JCT", "P95 JCT", "Max JCT", "Min JCT", "Makespan"], fontsize=8)
    ax_bar.set_ylabel("Seconds")
    ax_bar.set_title(f"Mean summary metrics — {scenario_name}")
    ax_bar.legend(fontsize=7)
    ax_bar.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Scenario: {scenario_name} ({summary.rep.nunique()} repetitions)")
    fig.tight_layout()
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def plot_pacing_sweep(pacing_df: pd.DataFrame, out_path: str | os.PathLike[str]) -> None:
    order = [
        "default",
        "custom-baseline",
        "custom-baseline",
        "custom-delay-1s",
        "custom-delay-2s",
        "custom-delay-5s",
        "custom-adaptive",
    ]
    order = [config for config in order if config in set(pacing_df.config)]
    means = pacing_df.groupby("config")[["avg_ilt", "avg_jct", "makespan"]].mean().reindex(order)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(order))
    colors = ["#e67e22" if name == "default" else "#2ecc71" if name == "custom-adaptive" else "#3498db" for name in order]
    for axis, metric, title in (
        (axes[0], "avg_jct", "Effect of pacing on average JCT"),
        (axes[1], "makespan", "Effect of pacing on makespan"),
    ):
        axis.bar(x, means[metric], color=colors)
        axis.set_xticks(x)
        axis.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
        axis.set_ylabel("Seconds")
        axis.set_title(title)
        axis.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(list(argv) if argv is not None else None)
    main_df = pd.read_csv(args.results_dir / "main_comparison.csv")
    raw_df = pd.read_csv(args.results_dir / "raw_jobs.csv")
    pacing_df = pd.read_csv(args.results_dir / "pacing_sweep.csv")
    for scenario in main_df.scenario.unique():
        output = args.results_dir / f"scenario_{scenario}.png"
        plot_scenario(scenario, main_df, raw_df, output)
        print(f"saved {output}")
    output = args.results_dir / "pacing_sweep.png"
    plot_pacing_sweep(pacing_df, output)
    print(f"saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
