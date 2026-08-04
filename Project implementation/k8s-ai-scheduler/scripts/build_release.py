"""Build release archives and emit checksums outside the source tree."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_release(output_dir: Path) -> list[Path]:
    output_dir = output_dir.resolve()
    if output_dir == PROJECT_ROOT or PROJECT_ROOT not in output_dir.parents:
        raise ValueError("release output must be a child of the project root")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(output_dir)],
        cwd=PROJECT_ROOT,
        check=True,
    )
    artifacts = sorted(
        path for path in output_dir.iterdir() if path.suffix == ".whl" or path.name.endswith(".tar.gz")
    )
    if len(artifacts) != 2:
        raise RuntimeError(f"expected one wheel and one sdist, found: {artifacts}")
    checksum_file = output_dir / "SHA256SUMS.txt"
    checksum_file.write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in artifacts),
        encoding="utf-8",
        newline="\n",
    )
    return [*artifacts, checksum_file]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "dist")
    args = parser.parse_args(argv)
    for path in build_release(args.output_dir):
        print(f"{path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
