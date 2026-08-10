import json

import pytest

from scripts.analyze_pilot_structure import (
    _kendall_tau_b,
    _rankdata,
    _spearman,
    analyze_pilot,
)


def test_rank_statistics_handle_order_and_ties():
    assert _rankdata([10, 20, 20, 30]) == [1.0, 2.5, 2.5, 4.0]
    assert _spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert _spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert _kendall_tau_b([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_real_archive_extractor_writes_sanitized_evidence(tmp_path):
    pilot = tmp_path / "pilot"
    output = tmp_path / "out"
    pilot.mkdir()
    annotations = {
        "ml.scheduler/estimated-training-time": "10",
        "ml.scheduler/loss-reduction-rate": "0.1",
        "ml.scheduler/matrix-size": "128",
        "ml.scheduler/gradient-update-size": "5",
        "ml.scheduler/checkpoint-interval": "20",
        "ml.scheduler/model-partitions": "1",
    }
    manifest = {
        "items": [
            {
                "metadata": {
                    "labels": {"pilot.kubeml/job-index": str(index)},
                    "annotations": {
                        **annotations,
                        "ml.scheduler/estimated-training-time": str(10 + index),
                        "ml.scheduler/matrix-size": str(128 + index),
                    },
                }
            }
            for index in range(3)
        ]
    }
    (pilot / "tfp-k-r0-test-manifest.json").write_text(json.dumps(manifest))
    record = {
        "events": [
            {
                "event": "fast_path_decision",
                "selected": True,
                "reason": "ranked_queue_prefill",
                "cpu_threshold": 0.85,
                "current_cpu_utilization": 0.3,
                "burst_cpu_demand_cores": 3.0,
                "allocatable_cpu_cores": 2.0,
                "projected_cpu_utilization": 1.8,
                "headroom_release_count": 1,
                "initial_release_count": 3,
            },
            {
                "event": "training_tail_balance",
                "selected": True,
                "parallelism": 2,
                "protected_prefix": 1,
                "predicted_makespan_before": 20,
                "predicted_makespan_after": 19,
            },
        ]
    }
    (pilot / "gate-tfp-k-r0-test.json").write_text(json.dumps(record))

    summary = analyze_pilot(pilot, output)

    assert summary["fast_path"]["reasons"] == {"ranked_queue_prefill": 1}
    assert (output / "pilot_feature_matrix.csv").is_file()
    assert (output / "rank_policy_diagnostics.csv").is_file()
    assert (output / "fastpath_branch_evidence.csv").is_file()
    serialized = (output / "structural_diagnostics.json").read_text()
    assert "namespace" not in serialized
    assert "pod_uid" not in serialized
