#!/usr/bin/env python3
"""Link SAM2 mask artifacts for manifest image_ids from several source dirs.

This avoids copying large SAM2 JSON/PNG files while still giving run_eval.py a
single --sam-masks-dir containing the expected filenames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SUFFIXES = ("_sam_masks.json", "_sam_masks.png", "_numbered.png", "_sam_done.flag")


def link_or_replace(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and Path(dst.readlink()) == src:
            return False
        dst.unlink()
    dst.symlink_to(src)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, action="append", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--require-json", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    items = json.loads(args.manifest.read_text(encoding="utf-8")).get("items", [])
    selected = items[args.start_index:args.end_index]
    wanted = [item["image_id"] for item in selected]

    linked = 0
    found_json = 0
    missing_json = []
    source_hits = {str(src): 0 for src in args.source_dir}

    for image_id in wanted:
        chosen_source = None
        for src_dir in args.source_dir:
            if (src_dir / f"{image_id}_sam_masks.json").exists():
                chosen_source = src_dir
                break
        if chosen_source is None:
            if len(missing_json) < 100:
                missing_json.append(image_id)
            continue
        found_json += 1
        source_hits[str(chosen_source)] += 1
        for suffix in SUFFIXES:
            src = chosen_source / f"{image_id}{suffix}"
            if src.exists():
                linked += int(link_or_replace(src, args.output_dir / src.name))

    report = {
        "manifest": str(args.manifest),
        "output_dir": str(args.output_dir),
        "selected_items": len(wanted),
        "found_json": found_json,
        "missing_json": len(wanted) - found_json,
        "missing_json_samples": missing_json,
        "linked_or_replaced": linked,
        "source_hits": source_hits,
    }
    if args.require_json and report["missing_json"]:
        report["status"] = "incomplete"
    else:
        report["status"] = "ok"
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text)
    if args.require_json and report["missing_json"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
