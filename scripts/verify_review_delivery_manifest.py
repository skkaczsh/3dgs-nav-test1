#!/usr/bin/env python3
"""Verify a packaged cross-candidate review delivery manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_dir(delivery_dir: Path) -> dict:
    manifest_path = delivery_dir / "manifest_sha256.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    for row in manifest.get("files", []):
        rel = row["path"]
        path = delivery_dir / rel
        if not path.exists():
            errors.append({"path": rel, "error": "missing"})
            continue
        size = path.stat().st_size
        digest = sha256(path)
        if size != int(row["bytes"]):
            errors.append({"path": rel, "error": "size_mismatch", "expected": row["bytes"], "actual": size})
        if digest != row["sha256"]:
            errors.append({"path": rel, "error": "sha256_mismatch", "expected": row["sha256"], "actual": digest})
    return {
        "delivery_dir": str(delivery_dir),
        "manifest": str(manifest_path),
        "expected_file_count": int(manifest.get("file_count", 0)),
        "checked_file_count": len(manifest.get("files", [])),
        "errors": errors,
        "passed": not errors and int(manifest.get("file_count", 0)) == len(manifest.get("files", [])),
    }


def verify_zip(zip_path: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="review_delivery_verify_") as tmp:
        out = Path(tmp)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out)
        report = verify_dir(out)
        report["zip"] = str(zip_path)
        return report


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--delivery-dir", type=Path)
    group.add_argument("--zip-path", type=Path)
    parser.add_argument("--output-report", type=Path, default=None)
    args = parser.parse_args()

    report = verify_zip(args.zip_path) if args.zip_path else verify_dir(args.delivery_dir)
    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
