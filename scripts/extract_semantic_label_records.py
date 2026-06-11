#!/usr/bin/env python3
"""Extract structured VLM label records from semantic_eval summaries.

The semantic_eval artifacts keep labels.json as a legacy mask_id -> label map.
This script writes label_records.json next to it, preserving optional identity
fields such as description, identity_hint, and attributes for Target/Object
fusion without changing the old labels.json contract.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


LABEL_ALIASES = {
    "天空": "sky",
    "天": "sky",
    "地面": "floor",
    "屋顶": "floor",
    "楼面": "floor",
    "地板": "floor",
    "道路": "road",
    "墙": "wall",
    "墙面": "wall",
    "建筑": "building",
    "建筑物": "building",
    "栏杆": "railing",
    "护栏": "railing",
    "围栏": "railing",
    "设备": "equipment",
    "空调": "equipment",
    "空调外机": "equipment",
    "管道": "pipe",
    "管线": "pipe",
    "树": "tree",
    "草": "grass",
    "车": "car",
    "人": "person",
    "其他": "other",
    "忽略": "ignore",
    "无效": "ignore",
}

ALLOWED = {
    "sky",
    "floor",
    "road",
    "wall",
    "building",
    "railing",
    "equipment",
    "pipe",
    "tree",
    "grass",
    "car",
    "person",
    "other",
    "ignore",
    "unknown",
}


def normalize_label(value: Any) -> str:
    label = str(value or "other").strip().lower()
    label = LABEL_ALIASES.get(label, label)
    return label if label in ALLOWED else "other"


def normalize_record(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        attrs = value.get("attributes") or {}
        if not isinstance(attrs, dict):
            attrs = {}
        try:
            confidence = float(value.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        return {
            "label": normalize_label(value.get("label", "other")),
            "confidence": max(0.0, min(1.0, confidence)),
            "description": str(value.get("description", "")).strip(),
            "identity_hint": str(value.get("identity_hint", "")).strip(),
            "attributes": {str(k): str(v).strip() for k, v in attrs.items() if str(v).strip()},
        }
    return {
        "label": normalize_label(value),
        "confidence": 1.0,
        "description": "",
        "identity_hint": "",
        "attributes": {},
    }


def parse_raw_items(raw: str) -> dict[int, dict[str, Any]]:
    if not raw:
        return {}
    first = raw.find("{")
    last = raw.rfind("}")
    if first < 0 or last <= first:
        return {}
    try:
        data = json.loads(raw[first : last + 1])
    except Exception:
        return {}
    rows: dict[int, dict[str, Any]] = {}
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        for item in data["items"]:
            if not isinstance(item, dict):
                continue
            match = re.search(r"(\d+)", str(item.get("mask_id", item.get("id", ""))))
            if match:
                rows[int(match.group(1))] = normalize_record(item)
    elif isinstance(data, dict):
        for key, value in data.items():
            match = re.search(r"(\d+)", str(key))
            if match:
                rows[int(match.group(1))] = normalize_record(value)
    return rows


def extract_from_summary(summary: dict[str, Any], labels: dict[str, Any]) -> dict[int, dict[str, Any]]:
    records = {int(k): normalize_record(v) for k, v in labels.items() if str(k).isdigit()}
    chunks = summary.get("vlm", {}).get("chunks", [])
    source_count = int(summary.get("source_mask_count", 0) or 0)
    is_completion = "completion" in str(summary.get("combo", ""))
    for chunk in chunks:
        raw_records = parse_raw_items(str(chunk.get("raw", "")))
        for local_id, record in raw_records.items():
            mask_id = source_count + local_id if is_completion and source_count else local_id
            if str(mask_id) in labels:
                records[mask_id] = {**records.get(mask_id, normalize_record(labels[str(mask_id)])), **record}
    return records


def process_combo(combo_dir: Path, overwrite: bool) -> dict[str, Any]:
    labels_path = combo_dir / "labels.json"
    summary_path = combo_dir / "summary.json"
    out_path = combo_dir / "label_records.json"
    if not labels_path.exists() or not summary_path.exists():
        return {"combo_dir": str(combo_dir), "status": "missing_inputs"}
    if out_path.exists() and not overwrite:
        return {"combo_dir": str(combo_dir), "status": "exists"}
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(labels, dict):
        return {"combo_dir": str(combo_dir), "status": "bad_labels"}
    records = extract_from_summary(summary, labels)
    source_combo = summary.get("source_combo")
    source_records_path = combo_dir.parent / str(source_combo) / "label_records.json" if source_combo else None
    if source_records_path and source_records_path.exists():
        try:
            source_records = json.loads(source_records_path.read_text(encoding="utf-8"))
        except Exception:
            source_records = {}
        if isinstance(source_records, dict):
            for key, value in source_records.items():
                if str(key).isdigit() and str(key) in labels:
                    records[int(key)] = normalize_record(value)
            records = extract_from_summary(summary, {str(k): v for k, v in records.items()})
    out = {str(k): v for k, v in sorted(records.items())}
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    enriched = sum(1 for row in out.values() if row.get("description") or row.get("identity_hint") or row.get("attributes"))
    return {"combo_dir": str(combo_dir), "status": "written", "records": len(out), "enriched": enriched}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-eval-dir", type=Path, required=True)
    parser.add_argument("--combo", default="sam2_prompt_v3_sky_label_merge_completion")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for combo_dir in sorted((args.semantic_eval_dir / "images").glob(f"cam*_*/{args.combo}")):
        rows.append(process_combo(combo_dir, args.overwrite))
    report = {
        "semantic_eval_dir": str(args.semantic_eval_dir),
        "combo": args.combo,
        "combo_dirs": len(rows),
        "written": sum(1 for row in rows if row.get("status") == "written"),
        "enriched_records": sum(int(row.get("enriched", 0)) for row in rows),
        "rows": rows[:100],
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
