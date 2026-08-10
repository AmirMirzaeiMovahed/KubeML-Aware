import json

import pytest

from scripts import analyze_reviewer_matrix as analysis


def _complete_report(*, claimable: bool = True, repetitions: int = 30):
    runs = []
    for repetition in range(repetitions):
        for arm_index, arm in enumerate(analysis.EXPECTED_ARMS):
            base = 100.0 + repetition
            value = base * (1.0 - 0.01 * arm_index)
            runs.append(
                {
                    "repetition": repetition,
                    "arm": arm,
                    "metrics": {metric: value for metric in analysis.METRICS},
                }
            )
    return {
        "eligible_for_article_claim": claimable,
        "reason": "unit-test fixture",
        "repetitions": repetitions,
        "runs": runs,
    }


def test_complete_claimable_matrix_generates_paired_effects():
    summary = analysis.summarize(_complete_report(), bootstrap_resamples=200)

    assert summary["publication_ready"] is True
    assert summary["repetitions"] == 30
    assert len(summary["effects"]) == len(analysis.COMPARISONS) * len(analysis.METRICS)
    primary = next(
        row
        for row in summary["effects"]
        if row["comparison"] == "end-to-end-vs-native"
        and row["metric"] == "avg_jct_seconds"
    )
    assert primary["n"] == 30
    assert primary["mean_improvement_pct"] > 0


def test_fifty_block_extension_is_analyzed_without_downsampling():
    summary = analysis.summarize(
        _complete_report(repetitions=50), bootstrap_resamples=200
    )

    assert summary["repetitions"] == 50
    assert all(row["n"] == 50 for row in summary["effects"])


def test_incomplete_matrix_cannot_generate_an_analysis():
    report = _complete_report()
    report["runs"].pop()

    with pytest.raises(analysis.MatrixValidationError, match="not a complete 30 x 7"):
        analysis.summarize(report, bootstrap_resamples=200)


def test_shared_node_output_requires_an_explicit_exploratory_override():
    report = _complete_report(claimable=False)

    with pytest.raises(analysis.MatrixValidationError, match="non-claimable"):
        analysis.summarize(report, bootstrap_resamples=200)
    assert (
        analysis.summarize(
            report, allow_nonclaimable=True, bootstrap_resamples=200
        )["publication_ready"]
        is False
    )


def test_output_bundle_contains_json_csv_and_latex(tmp_path):
    summary = analysis.summarize(_complete_report(), bootstrap_resamples=200)
    analysis.write_outputs(summary, tmp_path)

    parsed = json.loads((tmp_path / "reviewer_matrix_summary.json").read_text())
    assert parsed["publication_ready"] is True
    assert (tmp_path / "reviewer_matrix_effects.csv").is_file()
    assert "Evidence status: publishable" in (
        tmp_path / "reviewer_matrix_table.tex"
    ).read_text()
