"""Strict offline validation for rendered Kubernetes YAML documents."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Iterator

import yaml
from kubernetes_validate import validate


def yaml_paths(inputs: Iterable[Path]) -> Iterator[Path]:
    for source in inputs:
        if source.is_dir():
            yield from sorted(source.rglob("*.yaml"))
            yield from sorted(source.rglob("*.yml"))
        elif source.is_file():
            yield source
        else:
            raise FileNotFoundError(f"manifest input does not exist: {source}")


def validate_paths(inputs: Iterable[Path], kubernetes_version: str) -> int:
    count = 0
    paths = list(yaml_paths(inputs))
    if not paths:
        raise ValueError("no YAML manifest files were found")
    for path in paths:
        try:
            documents = yaml.safe_load_all(path.read_text(encoding="utf-8"))
            for index, document in enumerate(documents, start=1):
                if document is None:
                    continue
                if not isinstance(document, dict):
                    raise ValueError(f"{path} document {index} is not an object")
                validate(document, kubernetes_version, strict=True)
                count += 1
        except Exception as exc:
            raise ValueError(f"manifest validation failed for {path}: {exc}") from exc
    if count == 0:
        raise ValueError("YAML inputs contained no Kubernetes objects")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--kubernetes-version", default="1.36.0")
    args = parser.parse_args()
    count = validate_paths(args.paths, args.kubernetes_version)
    print(
        f"strict Kubernetes {args.kubernetes_version} validation passed: "
        f"{count} objects"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
