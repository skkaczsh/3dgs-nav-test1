#!/usr/bin/env python3
"""Filter a semantic manifest to image_ids missing a target combo artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--combo", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-source-combo", default=None)
    args = parser.parse_args()

    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    missing = []
    skipped_done = 0
    skipped_missing_source = 0
    for item in data.get("items", []):
        image_id = item["image_id"]
        combo_dir = args.output_dir / "images" / image_id / args.combo
        if (combo_dir / "semantic.png").exists():
            skipped_done += 1
            continue
        if args.require_source_combo:
            src = args.output_dir / "images" / image_id / args.require_source_combo / "semantic.png"
            if not src.exists():
                skipped_missing_source += 1
                continue
        missing.append(item)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"items": missing}, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "source_manifest": str(args.manifest),
        "output_dir": str(args.output_dir),
        "combo": args.combo,
        "require_source_combo": args.require_source_combo,
        "missing": len(missing),
        "skipped_done": skipped_done,
        "skipped_missing_source": skipped_missing_source,
        "output": str(args.output),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
