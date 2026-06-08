#!/usr/bin/env python3
"""Filter a semantic manifest to items that have required cached artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def sky_exists(item: dict) -> bool:
    path = item.get("sky_mask_path") or ""
    return bool(path) and Path(path).exists()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sam-masks-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-sky", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = data.get("items", [])[args.start_index:args.end_index]
    ready = []
    missing = {"sky": 0, "sam": 0}
    for item in items:
        image_id = item["image_id"]
        if args.require_sky and not sky_exists(item):
            missing["sky"] += 1
            continue
        if not (args.sam_masks_dir / f"{image_id}_sam_masks.json").exists():
            missing["sam"] += 1
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
