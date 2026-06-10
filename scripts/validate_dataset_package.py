#!/usr/bin/env python3
"""Validate a lightweight dataset delivery package."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path("/Users/skkac/Work/SCAN")
    parser.add_argument("--package-dir", type=Path, default=root / "dataset_delivery_0000_0999")
    parser.add_argument("--tgz", type=Path, default=root / "dataset_delivery_0000_0999.tgz")
    parser.add_argument("--output", type=Path, default=root / "dataset_delivery_0000_0999_validation.json")
    args = parser.parse_args()

    errors = []
    package_manifest = args.package_dir / "package_manifest.json"
    large_files = args.package_dir / "large_files.json"
    readme = args.package_dir / "README.md"
    for path in [package_manifest, large_files, readme]:
        if not path.exists() or path.stat().st_size <= 0:
            errors.append(f"missing or empty package file: {path}")

    manifest = {}
    if package_manifest.exists():
        manifest = json.loads(package_manifest.read_text(encoding="utf-8"))
        if not manifest.get("passed"):
            errors.append("package_manifest passed is false")
        for row in manifest.get("files", []):
            if row.get("packaged"):
                packaged_path = args.package_dir / row.get("package_path", "")
                if not packaged_path.exists() or packaged_path.stat().st_size <= 0:
                    errors.append(f"missing packaged artifact: {packaged_path}")
            elif row.get("required"):
                source = Path(row.get("path", ""))
                if not source.exists():
                    errors.append(f"missing referenced required artifact: {source}")

    if not args.tgz.exists() or args.tgz.stat().st_size <= 0:
        errors.append(f"missing tgz: {args.tgz}")
    else:
        try:
            with tarfile.open(args.tgz, "r:gz") as tf:
                names = set(tf.getnames())
            if f"{args.package_dir.name}/package_manifest.json" not in names:
                errors.append("tgz missing package_manifest.json")
            if f"{args.package_dir.name}/README.md" not in names:
                errors.append("tgz missing README.md")
        except tarfile.TarError as exc:
            errors.append(f"invalid tgz: {exc}")

    result = {
        "package_dir": str(args.package_dir),
        "tgz": str(args.tgz),
        "passed": not errors,
        "errors": errors,
        "packaged_file_count": sum(1 for row in manifest.get("files", []) if row.get("packaged")),
        "large_file_count": len(manifest.get("large_files", [])),
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
