#!/usr/bin/env python3
"""Summarize semantic-eval progress across split output directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_COMBOS = [
    "sam2_qwen",
    "sam2_sky_label_merge_qwen_review",
    "sam2_prompt_v3_sky_label_merge",
    "sam2_prompt_v3_sky_label_merge_completion",
]


def load_manifest(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [item["image_id"] for item in data.get("items", [])]


def count_combo(output_dir: Path, image_ids: list[str], combo: str) -> tuple[int, list[str]]:
    missing = []
    count = 0
    for image_id in image_ids:
        path = output_dir / "images" / image_id / combo / "semantic.png"
        if path.exists():
            count += 1
        elif len(missing) < 20:
            missing.append(image_id)
    return count, missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", action="append", nargs=3, metavar=("NAME", "MANIFEST", "OUTPUT_DIR"), required=True)
    parser.add_argument("--combo", action="append", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    combos = args.combo or DEFAULT_COMBOS
    split_rows = {}
    all_expected: list[str] = []
    all_completed_by_combo = {combo: set() for combo in combos}

    for name, manifest_text, output_text in args.split:
        manifest = Path(manifest_text)
        output_dir = Path(output_text)
        image_ids = load_manifest(manifest)
        all_expected.extend(image_ids)
        combo_rows = {}
        for combo in combos:
            count, missing = count_combo(output_dir, image_ids, combo)
            completed_ids = {
                image_id
                for image_id in image_ids
                if (output_dir / "images" / image_id / combo / "semantic.png").exists()
            }
            all_completed_by_combo[combo].update(completed_ids)
            combo_rows[combo] = {
                "count": count,
                "total": len(image_ids),
                "ratio": count / max(len(image_ids), 1),
                "missing_samples": missing,
            }
        split_rows[name] = {
            "manifest": str(manifest),
            "output_dir": str(output_dir),
            "total": len(image_ids),
            "combos": combo_rows,
        }

    expected_set = set(all_expected)
    duplicate_expected = len(all_expected) - len(expected_set)
    totals = {}
    for combo, completed in all_completed_by_combo.items():
        missing = sorted(expected_set - completed)
        extra = sorted(completed - expected_set)
        totals[combo] = {
            "completed": len(completed & expected_set),
            "expected": len(expected_set),
            "ratio": len(completed & expected_set) / max(len(expected_set), 1),
            "missing_samples": missing[:30],
            "extra_samples": extra[:30],
        }

    result = {
        "splits": split_rows,
        "summary": {
            "expected_unique_images": len(expected_set),
            "duplicate_expected_images": duplicate_expected,
            "combos": totals,
        },
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
