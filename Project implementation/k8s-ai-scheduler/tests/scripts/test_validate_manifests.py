from pathlib import Path

import pytest

from scripts.validate_manifests import validate_paths


def test_strict_manifest_validation_accepts_valid_pod(tmp_path: Path):
    manifest = tmp_path / "pod.yaml"
    manifest.write_text(
        """apiVersion: v1
kind: Pod
metadata:
  name: test
spec:
  containers:
    - name: app
      image: example/app:1.0.0
""",
        encoding="utf-8",
    )
    assert validate_paths([manifest], "1.36.0") == 1


def test_strict_manifest_validation_rejects_unknown_fields(tmp_path: Path):
    manifest = tmp_path / "invalid.yaml"
    manifest.write_text(
        """apiVersion: v1
kind: Pod
metadata:
  name: test
spec:
  unknownField: true
  containers:
    - name: app
      image: example/app:1.0.0
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest validation failed"):
        validate_paths([manifest], "1.36.0")
