#!/usr/bin/env python3
"""Validate a ConceptSeg fine-object runlist package."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


BROAD_PROMPTS = {"floor", "wall", "building facade"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runlist", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-local-assets", action="store_true")
    args = parser.parse_args()

    data = read_json(args.runlist)
    items = data.get("items", [])
    errors: list[str] = []
    concept_counts = Counter()
    source_label_counts = Counter()

    for idx, item in enumerate(items):
        prefix = f"items[{idx}]"
        for key in ("image_id", "concept", "image_path", "metadata"):
            if key not in item:
                errors.append(f"{prefix}: missing {key}")
        concept = str(item.get("concept", ""))
        concept_counts[concept] += 1
        if concept.strip().lower() in BROAD_PROMPTS:
            errors.append(f"{prefix}: broad prompt is not allowed: {concept}")
        meta = item.get("metadata") or {}
        source_label_counts[str(meta.get("source_label", "unknown"))] += 1
        rep = meta.get("representative") or {}
        if not rep.get("target_id"):
            errors.append(f"{prefix}: missing target_id")
        if args.check_local_assets:
            local_assets = meta.get("local_assets") or {}
            for asset_name in ("image", "overlay", "instance", "semantic", "labels"):
                path = local_assets.get(asset_name)
                if not path or not Path(path).exists():
                    errors.append(f"{prefix}: missing local asset {asset_name}: {path}")

    report = {
        "passed": not errors,
        "runlist": str(args.runlist),
        "items": len(items),
        "concept_counts": dict(concept_counts),
        "source_label_counts": dict(source_label_counts),
        "errors": errors[:200],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
