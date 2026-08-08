"""CLI for producing scheduler annotations from JSONL inference samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .model import InferenceSample, build_inference_profile


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSONL sample file")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--latency-slo-ms", type=float, required=True)
    parser.add_argument("--priority", type=float, default=1.0)
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--output", type=Path, help="write JSON here instead of stdout")
    return parser


def _load_samples(path: Path) -> list[InferenceSample]:
    samples = []
    # utf-8-sig accepts ordinary UTF-8 and the BOM emitted by Windows
    # PowerShell's legacy Set-Content implementation.
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("sample must be a JSON object")
                samples.append(InferenceSample.from_mapping(payload))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"invalid sample at line {line_number}: {exc}") from exc
    return samples


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        profile = build_inference_profile(
            _load_samples(args.input),
            job_id=args.job_id,
            latency_slo_ms=args.latency_slo_ms,
            priority=args.priority,
            percentile=args.percentile,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    document = json.dumps(profile.to_dict(), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(document, encoding="utf-8")
    else:
        sys.stdout.write(document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
