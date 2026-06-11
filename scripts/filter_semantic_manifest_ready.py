#!/usr/bin/env python3
"""Filter a semantic manifest to items that have required cached artifacts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def sky_exists(item: dict) -> bool:
    path = item.get("sky_mask_path") or ""
    return bool(path) and Path(path).exists()


def sam_ready(path: Path, validate_json: bool, min_age_seconds: float) -> tuple[bool, str | None]:
    if not path.exists():
        return False, "sam"
    if min_age_seconds > 0:
        age = time.time() - path.stat().st_mtime
        if age < min_age_seconds:
            return False, "sam_unstable"
    if validate_json:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False, "sam_invalid_json"
    return True, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sam-masks-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-sky", action="store_true")
    parser.add_argument("--validate-sam-json", action="store_true")
    parser.add_argument("--min-sam-age-seconds", type=float, default=0.0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = data.get("items", [])[args.start_index:args.end_index]
    ready = []
    missing = {"sky": 0, "sam": 0, "sam_unstable": 0, "sam_invalid_json": 0}
    for item in items:
        image_id = item["image_id"]
        if args.require_sky and not sky_exists(item):
            missing["sky"] += 1
            continue
        ok, reason = sam_ready(
            args.sam_masks_dir / f"{image_id}_sam_masks.json",
            args.validate_sam_json,
            args.min_sam_age_seconds,
        )
        if not ok:
            missing[reason or "sam"] += 1
            continue
        ready.append(item)
        if args.limit and len(ready) >= args.limit:
            break

    out = {
        **{k: v for k, v in data.items() if k != "items"},
        "items": ready,
        "filter_report": {
            "source_manifest": str(args.manifest),
            "sam_masks_dir": str(args.sam_masks_dir),
            "require_sky": args.require_sky,
            "validate_sam_json": args.validate_sam_json,
            "min_sam_age_seconds": args.min_sam_age_seconds,
            "selected": len(ready),
            "missing_seen": missing,
            "start_index": args.start_index,
            "end_index": args.end_index,
            "limit": args.limit,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out["filter_report"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
