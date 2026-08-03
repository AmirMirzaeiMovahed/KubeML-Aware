import json

import pytest

from k8s.work_model import WORK_MODEL_VERSION
from sim.calibration import (
    CALIBRATION_KIND,
    CALIBRATION_SCHEMA_VERSION,
    _canonical_sha256,
    calibrated_model,
    load_calibrated_model,
    validate_calibration_document,
)


def _document():
    evidence = {
        "work_model_version": WORK_MODEL_VERSION,
        "parameters": {
            "matrix_reference": 128.0,
            "matmul_seconds_at_reference": 0.001,
            "gradient_scale": 1.2,
            "synchronization_scale": 1.3,
            "checkpoint_scale": 1.4,
            "estimated_time_weight": 0.0,
            "convergence_threshold": 0.02,
        },
        "benchmark": {"repeats": 7},
        "environment": {"machine": "test"},
    }
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "kind": CALIBRATION_KIND,
        "calibration_id": f"sha256:{_canonical_sha256(evidence)}",
        **evidence,
    }


def test_calibration_is_content_addressed_and_builds_model(tmp_path):
    document = _document()
    validate_calibration_document(document)
    model = calibrated_model(document)
    assert model.calibrated is True
    assert model.calibration_id == document["calibration_id"]
    assert model.gradient_scale == pytest.approx(1.2)

    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert load_calibrated_model(path) == model


def test_tampered_or_stale_calibration_fails_closed():
    document = _document()
    document["parameters"]["gradient_scale"] = 99
    with pytest.raises(ValueError, match="does not match"):
        validate_calibration_document(document)

    stale = _document()
    stale["work_model_version"] = "stale"
    with pytest.raises(ValueError, match="stale"):
        validate_calibration_document(stale)
