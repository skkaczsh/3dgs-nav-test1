#!/usr/bin/env python3
"""Summarize a SAM2 TensorRT downstream semantic smoke output directory."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


COMBOS = (
    "sam2_qwen",
    "sam2_sky_label_merge_qwen_review",
    "sam2_prompt_v3_sky_label_merge",
    "sam2_prompt_v3_sky_label_merge_completion",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def label_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = read_json(path)
    values = data.values() if isinstance(data, dict) else data
    return [row for row in values if isinstance(row, dict)]


def parse_stage_logs(log_dir: Path, stage: str) -> dict[str, Any]:
    false_ids: list[str] = []
    true_count = 0
    total = 0
    for path in sorted(log_dir.glob(f"{stage}_*.log")):
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            match = re.search(r"(cam\d+_\d+).*parse=(True|False)", line)
            if not match:
                continue
            total += 1
            if match.group(2) == "True":
                true_count += 1
            else:
                false_ids.append(match.group(1))
    return {
        "total_parse_lines": total,
        "parse_true": true_count,
        "parse_false_count": len(false_ids),
        "parse_false": false_ids,
    }


def summarize(output_dir: Path, combo: str) -> dict[str, Any]:
    counts = {}
    for name in COMBOS:
        counts[name] = len(list((output_dir / "images").glob(f"*/{name}/semantic.png")))

    label_counts: Counter[str] = Counter()
    record_counts: list[int] = []
    missing_records: list[str] = []
    for combo_dir in sorted((output_dir / "images").glob(f"*/{combo}")):
        image_id = combo_dir.parent.name
        records = label_records(combo_dir / "label_records.json")
        if not records:
            missing_records.append(image_id)
        record_counts.append(len(records))
        for row in records:
            label_counts[str(row.get("label", "unknown"))] += 1

    log_dir = output_dir / "_sharded_work" / "logs"
    parse = {
        "sam2_qwen": parse_stage_logs(log_dir, "sam2_qwen"),
        "review": parse_stage_logs(log_dir, "review"),
        "completion": parse_stage_logs(log_dir, "completion"),
    }

    return {
        "output_dir": str(output_dir),
        "combo": combo,
        "combo_semantic_png_counts": counts,
        "label_record_files": len(record_counts),
        "label_records": sum(record_counts),
        "mean_records_per_frame": mean(record_counts) if record_counts else 0,
        "max_records_per_frame": max(record_counts) if record_counts else 0,
        "missing_label_records": missing_records,
        "top_labels": label_counts.most_common(30),
        "parse_from_logs": parse,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--combo", default="sam2_prompt_v3_sky_label_merge_completion")
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    report = summarize(args.output_dir, args.combo)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
