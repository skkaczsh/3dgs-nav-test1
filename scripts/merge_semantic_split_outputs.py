#!/usr/bin/env python3
"""Merge split semantic-eval outputs into one directory via image-dir symlinks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def load_items(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("items", [])


def same_link(path: Path, target: Path) -> bool:
    return path.is_symlink() and Path(os.readlink(path)) == target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", action="append", nargs=3, metavar=("NAME", "MANIFEST", "OUTPUT_DIR"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replace", action="store_true", help="Replace existing destination image dirs or links.")
    parser.add_argument("--require-combo", action="append", default=[])
    args = parser.parse_args()

    out_images = args.output_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    merged_items: list[dict] = []
    seen: dict[str, str] = {}
    merged_ids: set[str] = set()
    rows = []
    duplicate_ids = []
    missing_source_dirs = []
    missing_required = {combo: [] for combo in args.require_combo}
    linked = 0
    skipped_existing = 0

    for name, manifest_text, output_text in args.split:
        manifest = Path(manifest_text)
        output_dir = Path(output_text)
        items = load_items(manifest)
        split_linked = 0
        split_missing = 0
        for item in items:
            image_id = item["image_id"]
            if image_id in seen:
                duplicate_ids.append({"image_id": image_id, "first_split": seen[image_id], "split": name})
                continue
            seen[image_id] = name
            src = output_dir / "images" / image_id
            dst = out_images / image_id
            if not src.exists():
                missing_source_dirs.append({"split": name, "image_id": image_id, "path": str(src)})
                split_missing += 1
                continue
            for combo in args.require_combo:
                if not (src / combo / "semantic.png").exists():
                    if len(missing_required[combo]) < 50:
                        missing_required[combo].append({"split": name, "image_id": image_id})
            if dst.exists() or dst.is_symlink():
                if same_link(dst, src):
                    skipped_existing += 1
                    merged_items.append(item)
                    merged_ids.add(image_id)
                    continue
                if not args.replace:
                    raise FileExistsError(f"{dst} exists; pass --replace to overwrite")
                if dst.is_symlink() or dst.is_file():
                    dst.unlink()
                else:
                    shutil.rmtree(dst)
            dst.symlink_to(src)
            linked += 1
            split_linked += 1
            merged_items.append(item)
            merged_ids.add(image_id)
        rows.append(
            {
                "split": name,
                "manifest": str(manifest),
                "output_dir": str(output_dir),
                "items": len(items),
                "linked": split_linked,
                "missing_source_dirs": split_missing,
            }
        )

    manifest_out = args.output_dir / "manifest.json"
    manifest_out.write_text(json.dumps({"items": merged_items}, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "output_dir": str(args.output_dir),
        "total_items": len(merged_items),
        "expected_unique_image_ids": len(seen),
        "merged_unique_image_ids": len(merged_ids),
        "linked": linked,
        "skipped_existing": skipped_existing,
        "duplicate_ids": duplicate_ids[:50],
        "duplicate_count": len(duplicate_ids),
        "missing_source_dirs": missing_source_dirs[:50],
        "missing_source_count": len(missing_source_dirs),
        "missing_required": missing_required,
        "splits": rows,
    }
    report_path = args.output_dir / "merge_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
