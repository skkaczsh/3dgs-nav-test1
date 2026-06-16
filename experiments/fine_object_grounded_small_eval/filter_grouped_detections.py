#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_review_mapping(path: Path) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            image_id = row.get("image_id", "")
            parts = image_id.split("_")
            if len(parts) < 3:
                continue
            review_id = parts[0] + "_" + parts[1]
            slot = parts[2]
            key = f"{review_id}__{slot}"
            mapping.setdefault(
                key,
                {
                    "frame": int(row["frame"]),
                    "cam": int(row["cam"]),
                    "target_id": row["target_id"],
                    "source_label": row.get("source_label", "unknown"),
                },
            )
    return mapping


def accept_detection(det: dict) -> tuple[bool, str]:
    focus = det["focus"]
    score = float(det["grounding_score"])
    area_ratio = float(det["mask_area_ratio"])
    box_area_ratio = float(det["box_area_ratio"])
    aspect = float(det["box_aspect_ratio"])
    phrase = det["phrase"].lower()

    if focus == "railing":
        if score < 0.20:
            return False, "low_score"
        if area_ratio < 0.002:
            return False, "tiny_mask"
        if area_ratio > 0.18:
            return False, "oversized_mask"
        if aspect < 2.2:
            return False, "not_elongated"
        if not any(word in phrase for word in ("railing", "guardrail", "handrail", "fence")):
            return False, "phrase_mismatch"
        return True, "accepted"

    if focus == "pipe":
        if score < 0.20:
            return False, "low_score"
        if area_ratio < 0.001:
            return False, "tiny_mask"
        if area_ratio > 0.08:
            return False, "oversized_mask"
        if aspect < 1.8:
            return False, "not_elongated"
        if not any(word in phrase for word in ("pipe", "cable")):
            return False, "phrase_mismatch"
        return True, "accepted"

    if focus in {"equipment", "hvac"}:
        if score < 0.35:
            return False, "low_score"
        if area_ratio < 0.001:
            return False, "tiny_mask"
        if area_ratio > 0.08:
            return False, "oversized_mask"
        if box_area_ratio > 0.14:
            return False, "oversized_box"
        if not any(word in phrase for word in ("hvac", "outdoor", "air", "unit", "conditioning")):
            return False, "phrase_mismatch"
        return True, "accepted"

    return False, "unsupported_focus"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--review-mapping", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    summary_path = Path(args.summary)
    mapping_path = Path(args.review_mapping)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads(summary_path.read_text())
    review_mapping = load_review_mapping(mapping_path)

    accepted_rows = []
    rejected_rows = []
    for sample in summary["samples"]:
        sample_id = sample["id"]
        sample_mapping = review_mapping.get(sample_id)
        if not sample_mapping:
            continue
        best_by_focus = {}
        for det in sample["detections"]:
            ok, reason = accept_detection(det)
            row = {
                "sample_id": sample_id,
                "frame": sample_mapping["frame"],
                "cam": sample_mapping["cam"],
                "target_id": sample_mapping["target_id"],
                "source_label": sample_mapping["source_label"],
                "focus": det["focus"],
                "phrase": det["phrase"],
                "grounding_score": det["grounding_score"],
                "sam_score": det["sam_score"],
                "mask_area": det["mask_area"],
                "mask_area_ratio": det["mask_area_ratio"],
                "box_area_ratio": det["box_area_ratio"],
                "box_aspect_ratio": det["box_aspect_ratio"],
                "mask_path": det["mask_path"],
                "box_xyxy": det["box_xyxy"],
                "status": reason,
            }
            if ok:
                prev = best_by_focus.get(det["focus"])
                if prev is None or (row["grounding_score"], row["sam_score"]) > (
                    prev["grounding_score"],
                    prev["sam_score"],
                ):
                    best_by_focus[det["focus"]] = row
            else:
                rejected_rows.append(row)
        accepted_rows.extend(best_by_focus.values())

    summary_out = {
        "input_summary": str(summary_path),
        "review_mapping": str(mapping_path),
        "accepted_count": len(accepted_rows),
        "rejected_count": len(rejected_rows),
        "accepted_by_focus": {},
        "rejected_by_reason": {},
    }
    for row in accepted_rows:
        summary_out["accepted_by_focus"][row["focus"]] = summary_out["accepted_by_focus"].get(row["focus"], 0) + 1
    for row in rejected_rows:
        summary_out["rejected_by_reason"][row["status"]] = summary_out["rejected_by_reason"].get(row["status"], 0) + 1

    (output_dir / "accepted_detections.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in accepted_rows) + ("\n" if accepted_rows else "")
    )
    (output_dir / "rejected_detections.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rejected_rows) + ("\n" if rejected_rows else "")
    )
    (output_dir / "filter_summary.json").write_text(json.dumps(summary_out, ensure_ascii=False, indent=2) + "\n")
    print(output_dir / "filter_summary.json")


if __name__ == "__main__":
    main()
