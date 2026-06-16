#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def normalize_phrase(text: str) -> str:
    return " ".join(text.lower().split())


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


def accept_equipment_detection(det: dict, *, mode: str) -> tuple[bool, str]:
    score = float(det["grounding_score"])
    area_ratio = float(det["mask_area_ratio"])
    box_area_ratio = float(det["box_area_ratio"])
    aspect = float(det["box_aspect_ratio"])
    normalized = normalize_phrase(det["phrase"])

    if score < 0.35:
        return False, "low_score"
    if area_ratio < 0.002:
        return False, "tiny_mask"
    if aspect > 2.2:
        return False, "too_elongated"

    weak_phrases = {
        "air",
        "unit",
        "unit unit",
        "outdoor unit",
        "outdoor unit unit",
        "outdoor unit rooftop",
        "unit conditioning unit",
        "h outdoor unit unit",
    }
    if normalized in weak_phrases:
        return False, "phrase_too_weak"

    has_air_conditioning = (
        "air conditioning" in normalized
        or ("conditioning" in normalized and "unit" in normalized)
    )
    has_hvac = "hvac" in normalized
    has_outdoor_unit = "outdoor unit" in normalized
    has_rooftop_box = "rooftop equipment box" in normalized

    if mode == "strict_precision":
        if not (has_air_conditioning or has_hvac):
            return False, "phrase_too_broad"
        if has_rooftop_box and not (has_air_conditioning or has_hvac):
            return False, "phrase_too_broad"
        if has_outdoor_unit and not has_hvac and not has_air_conditioning:
            return False, "phrase_too_broad"
        if area_ratio > 0.03:
            return False, "oversized_mask"
        if box_area_ratio > 0.06:
            return False, "oversized_box"
        return True, "accepted"

    if area_ratio > 0.05:
        return False, "oversized_mask"
    if box_area_ratio > 0.08:
        return False, "oversized_box"
    strong_phrase = (
        has_air_conditioning
        or has_hvac
        or has_outdoor_unit
        or has_rooftop_box
    )
    if not strong_phrase:
        return False, "phrase_mismatch"
    return True, "accepted"


def accept_detection(det: dict, *, equipment_filter_mode: str) -> tuple[bool, str]:
    focus = det["focus"]
    score = float(det["grounding_score"])
    area_ratio = float(det["mask_area_ratio"])
    box_area_ratio = float(det["box_area_ratio"])
    aspect = float(det["box_aspect_ratio"])
    fill_ratio = float(det.get("mask_bbox_fill_ratio", 0.0))
    largest_component_ratio = float(det.get("largest_component_ratio", 0.0))
    minrect_aspect = float(det.get("minrect_aspect_ratio", 0.0))
    phrase = det["phrase"].lower()

    if focus == "railing":
        if score < 0.20:
            return False, "low_score"
        if area_ratio < 0.002:
            return False, "tiny_mask"
        if area_ratio > 0.12:
            return False, "oversized_mask"
        if aspect < 2.2 and minrect_aspect < 3.0:
            return False, "not_elongated"
        if fill_ratio > 0.38 and minrect_aspect < 4.0:
            return False, "too_filled_for_railing"
        if largest_component_ratio < 0.45:
            return False, "too_fragmented"
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
        return accept_equipment_detection(det, mode=equipment_filter_mode)

    return False, "unsupported_focus"


def candidate_rank(det: dict) -> tuple[float, ...]:
    focus = det["focus"]
    if focus == "railing":
        return (
            float(det.get("minrect_aspect_ratio", 0.0)),
            float(det.get("largest_component_ratio", 0.0)),
            -float(det.get("mask_bbox_fill_ratio", 0.0)),
            -float(det.get("mask_area_ratio", 0.0)),
            float(det.get("grounding_score", 0.0)),
            float(det.get("sam_score", 0.0)),
        )
    return (
        float(det.get("grounding_score", 0.0)),
        float(det.get("sam_score", 0.0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--review-mapping", required=False, default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--equipment-filter-mode",
        choices=["default", "strict_precision"],
        default="default",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    mapping_path = Path(args.review_mapping) if args.review_mapping else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads(summary_path.read_text())
    review_mapping = load_review_mapping(mapping_path) if mapping_path else {}

    accepted_rows = []
    rejected_rows = []
    for sample in summary["samples"]:
        sample_id = sample["id"]
        sample_mapping = review_mapping.get(sample_id)
        if not sample_mapping:
            sample_mapping = {
                "frame": int(sample.get("frame", -1)),
                "cam": int(sample.get("cam", -1)),
                "target_id": sample.get("id", ""),
                "source_label": sample.get("focus", ["unknown"])[0] if sample.get("focus") else "unknown",
            }
        best_by_focus = {}
        for det in sample["detections"]:
            ok, reason = accept_detection(
                det,
                equipment_filter_mode=args.equipment_filter_mode,
            )
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
                "mask_bbox_fill_ratio": det.get("mask_bbox_fill_ratio", 0.0),
                "largest_component_ratio": det.get("largest_component_ratio", 0.0),
                "minrect_aspect_ratio": det.get("minrect_aspect_ratio", 0.0),
                "component_count": det.get("component_count", 0),
                "mask_path": det["mask_path"],
                "box_xyxy": det["box_xyxy"],
                "status": reason,
            }
            if ok:
                prev = best_by_focus.get(det["focus"])
                if prev is None or candidate_rank(row) > candidate_rank(prev):
                    best_by_focus[det["focus"]] = row
            else:
                rejected_rows.append(row)
        accepted_rows.extend(best_by_focus.values())

    summary_out = {
        "input_summary": str(summary_path),
        "review_mapping": str(mapping_path) if mapping_path else "",
        "equipment_filter_mode": args.equipment_filter_mode,
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
