#!/usr/bin/env python3
"""Package cross-candidate review artifacts for transfer or manual review."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


CORE_FILES = [
    "README_review.md",
    "cross_candidate_review_items.jsonl",
    "cross_candidate_review_items.csv",
    "cross_candidate_review_pack_report.json",
    "review_html/index.html",
    "review_html/manual_merge_decisions.csv",
    "review_html/review_html_report.json",
    "stage_summary/cross_candidate_review_stage_summary.md",
    "stage_summary/cross_candidate_review_stage_summary.json",
    "manual_workflow_pending/manual_merge_workflow_report.json",
    "manual_workflow_pending/qa_reviewed_merge_report.json",
    "manual_workflow_pending/applied/review_merge_report.json",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_file(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def collect_files(pack_dir: Path, output_dir: Path, include_assets: bool = False) -> tuple[list[Path], list[str]]:
    copied = []
    missing = []
    for rel in CORE_FILES:
        src = pack_dir / rel
        dst = output_dir / rel
        if copy_file(src, dst):
            copied.append(dst)
        else:
            missing.append(rel)
    for src in sorted((pack_dir / "contact_sheets").glob("*")):
        if src.is_file() and copy_file(src, output_dir / "contact_sheets" / src.name):
            copied.append(output_dir / "contact_sheets" / src.name)
    if include_assets:
        for src in sorted((pack_dir / "assets").glob("proposal_*/*")):
            if src.is_file():
                dst = output_dir / "assets" / src.parent.name / src.name
                if copy_file(src, dst):
                    copied.append(dst)
    return copied, missing


def build_manifest(output_dir: Path, copied: list[Path], missing: list[str], source_pack: Path) -> dict:
    files = []
    for path in sorted(copied):
        rel = path.relative_to(output_dir)
        files.append({"path": str(rel), "bytes": path.stat().st_size, "sha256": sha256(path)})
    return {
        "source_pack": str(source_pack),
        "output_dir": str(output_dir),
        "file_count": len(files),
        "missing": missing,
        "files": files,
    }


def write_zip(output_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file() and path != zip_path:
                zf.write(path, path.relative_to(output_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path, default=None)
    parser.add_argument("--include-assets", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    copied, missing = collect_files(args.pack_dir, args.output_dir, include_assets=args.include_assets)
    manifest = build_manifest(args.output_dir, copied, missing, args.pack_dir)
    manifest_path = args.output_dir / "manifest_sha256.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.zip_path:
        write_zip(args.output_dir, args.zip_path)
    print(json.dumps({"output_dir": str(args.output_dir), "zip": str(args.zip_path or ""), "file_count": manifest["file_count"], "missing": missing}, indent=2))


if __name__ == "__main__":
    main()
