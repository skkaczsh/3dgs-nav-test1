#!/usr/bin/env python3
"""Build QA summaries and image evidence packs for frame-local fused objects."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from project_priority_masks_to_lx import PRIORITY_COLORS, PRIORITY_NAMES


FINE_LABELS = {"car", "railing"}
SURFACE_LABELS = {"ground", "wall", "grass"}
PRIORITY_BY_NAME = {name: idx for idx, name in PRIORITY_NAMES.items()}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def bbox_extent(bbox: dict[str, Any]) -> tuple[float, float, float]:
    lo = bbox.get("min") or [0, 0, 0]
    hi = bbox.get("max") or [0, 0, 0]
    return tuple(float(hi[i]) - float(lo[i]) for i in range(3))


def dominant_ratio(votes: dict[str, Any]) -> float:
    if not votes:
        return 0.0
    values = [float(v) for v in votes.values()]
    return float(max(values) / max(sum(values), 1.0))


def object_risk(obj: dict[str, Any]) -> tuple[float, list[str]]:
    label = str(obj.get("semantic_label") or "unknown")
    status = str(obj.get("status") or "")
    votes = obj.get("label_vote_weights") or obj.get("label_votes") or {}
    target_count = int(obj.get("target_count") or 0)
    point_count = int(obj.get("point_count") or 0)
    dx, dy, dz = bbox_extent(obj.get("bbox_3d") or {})
    horizontal = math.hypot(dx, dy)
    normal = obj.get("normal") or [0, 0, 0]
    normal_z = abs(float(normal[2])) if len(normal) >= 3 else 0.0
    planarity = float((obj.get("geometry_stats") or {}).get("planarity_mean") or 0.0)
    linearity = float((obj.get("geometry_stats") or {}).get("linearity_mean") or 0.0)
    ratio = dominant_ratio(votes)

    reasons: list[str] = []
    score = 0.0
    if label == "ambiguous" or status == "ambiguous_object" or ratio < 0.8:
        score += 100.0
        reasons.append("label_vote_conflict")
    if target_count <= 1 and point_count >= 500:
        score += 55.0
        reasons.append("large_single_target_object")
    if label in FINE_LABELS and point_count < 80:
        score += 35.0
        reasons.append("fine_object_low_points")
    if label == "railing":
        if linearity < 0.45:
            score += 45.0
            reasons.append("railing_not_linear")
        if dz > 1.8 or horizontal > 8.0:
            score += 35.0
            reasons.append("railing_extent_too_large")
    if label == "car":
        if dz < 0.25 or dz > 3.0 or horizontal > 10.0:
            score += 35.0
            reasons.append("car_extent_suspicious")
        if planarity > 0.65 and linearity < 0.2:
            score += 30.0
            reasons.append("car_surface_like")
    if label == "ground":
        if dz > 0.8:
            score += 40.0
            reasons.append("ground_has_large_height_span")
        if normal_z < 0.55 and planarity > 0.35:
            score += 30.0
            reasons.append("ground_normal_not_up")
    if label == "wall":
        if dz < 0.4 and horizontal > 1.5:
            score += 35.0
            reasons.append("wall_too_flat_low_height")
        if normal_z > 0.75 and planarity > 0.35:
            score += 35.0
            reasons.append("wall_normal_too_up")
    if label == "grass" and dz > 2.0:
        score += 25.0
        reasons.append("grass_large_height_span")
    if len(votes) > 1:
        score += 10.0 * (len(votes) - 1)
        reasons.append("multiple_label_votes")
    return score, reasons


def select_candidates(objects: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    candidates = []
    for obj in objects:
        score, reasons = object_risk(obj)
        if score <= 0:
            continue
        row = {
            "object_id": obj.get("object_id"),
            "semantic_label": obj.get("semantic_label"),
            "status": obj.get("status"),
            "risk_score": round(float(score), 3),
            "risk_reasons": reasons,
            "target_count": obj.get("target_count"),
            "point_count": obj.get("point_count"),
            "frames": obj.get("frames"),
            "label_votes": obj.get("label_votes"),
            "label_vote_weights": obj.get("label_vote_weights"),
            "bbox_3d": obj.get("bbox_3d"),
            "centroid": obj.get("centroid"),
            "mean_color": obj.get("mean_color"),
            "geometry_stats": obj.get("geometry_stats"),
            "color_stats": obj.get("color_stats"),
            "targets": obj.get("targets") or [],
        }
        candidates.append(row)
    candidates.sort(key=lambda r: (-float(r["risk_score"]), -int(r.get("point_count") or 0), str(r["object_id"])))
    return candidates[:limit]


def target_map(targets: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["target_id"]): row for row in targets}


def pick_evidence_targets(candidate: dict[str, Any], targets_by_id: dict[str, dict[str, Any]], per_object: int) -> list[dict[str, Any]]:
    rows = [targets_by_id[t] for t in candidate.get("targets", []) if t in targets_by_id]
    rows.sort(key=lambda r: (-int(r.get("cluster_size") or 0), int(r.get("frame_id") or 0), int(r.get("cam_id") or 0)))
    return rows[:per_object]


def resolve_path(workdir: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else workdir / p


def source_priority_id(target: dict[str, Any]) -> int | None:
    raw_label = str(target.get("raw_label") or target.get("refined_from_label") or "").strip()
    if raw_label in PRIORITY_BY_NAME:
        return int(PRIORITY_BY_NAME[raw_label])
    for key in ("source_priority_label_id", "priority_label_id", "mask_id"):
        value = target.get(key)
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def apply_source_mask_overlay(
    workdir: Path,
    target: dict[str, Any],
    crop: Image.Image,
    crop_box: tuple[int, int, int, int],
    alpha: float,
) -> tuple[Image.Image, str]:
    if alpha <= 0:
        return crop, "disabled"
    mask_path = resolve_path(workdir, str(target.get("mask_path") or ""))
    label_id = source_priority_id(target)
    if label_id is None or not mask_path.exists():
        return crop, "missing"
    with Image.open(mask_path) as mask_im:
        mask_im = mask_im.convert("L")
        mask_crop = mask_im.crop(crop_box)
    mask_data = mask_crop.point(lambda p: 255 if p == label_id else 0)
    if not mask_data.getbbox():
        return crop, "empty"
    color = PRIORITY_COLORS.get(label_id, (255, 40, 40))
    alpha_value = int(max(0.0, min(1.0, alpha)) * 255)
    overlay = Image.new("RGBA", crop.size, (*color, 0))
    overlay.putalpha(mask_data.point(lambda p: alpha_value if p else 0))
    base = crop.convert("RGBA")
    base.alpha_composite(overlay)
    return base.convert("RGB"), f"source_priority_{label_id}"


def crop_target_image(
    workdir: Path,
    candidate: dict[str, Any],
    target: dict[str, Any],
    output: Path,
    margin: int,
    mask_overlay_alpha: float,
) -> tuple[Path | None, str]:
    image_path = resolve_path(workdir, str(target.get("image_path") or ""))
    if not image_path.exists():
        return None, "missing_image"
    bbox = ((target.get("bbox_2d") or {}).get("xyxy") or [0, 0, 0, 0])
    x0, y0, x1, y1 = [int(round(float(x))) for x in bbox]
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        cx0, cy0 = max(0, x0 - margin), max(0, y0 - margin)
        cx1, cy1 = min(w - 1, x1 + margin), min(h - 1, y1 + margin)
        crop_box = (cx0, cy0, cx1 + 1, cy1 + 1)
        crop = im.crop(crop_box)
        crop, overlay_status = apply_source_mask_overlay(workdir, target, crop, crop_box, mask_overlay_alpha)
        draw = ImageDraw.Draw(crop)
        draw.rectangle((x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0), outline=(255, 30, 30), width=3)
        title = (
            f"{candidate['object_id']} {candidate['semantic_label']} "
            f"risk={candidate['risk_score']} f{target.get('frame_id')} c{target.get('cam_id')} "
            f"{target.get('label')} pts={target.get('cluster_size')}"
        )
        draw.rectangle((0, 0, crop.width, 34), fill=(0, 0, 0))
        draw.text((6, 8), title[:140], fill=(255, 255, 255))
        output.parent.mkdir(parents=True, exist_ok=True)
        crop.save(output, quality=92)
    return output, overlay_status


def make_contact_sheet(image_paths: list[Path], output: Path, thumb: tuple[int, int]) -> None:
    if not image_paths:
        return
    thumbs = []
    for path in image_paths:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail(thumb)
            canvas = Image.new("RGB", thumb, (18, 18, 18))
            canvas.paste(im, ((thumb[0] - im.width) // 2, (thumb[1] - im.height) // 2))
            thumbs.append(canvas)
    cols = min(4, len(thumbs))
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * thumb[0], rows * thumb[1]), (12, 12, 12))
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % cols) * thumb[0], (i // cols) * thumb[1]))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def summarize(objects: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(str(o.get("semantic_label") or "unknown") for o in objects)
    statuses = Counter(str(o.get("status") or "unknown") for o in objects)
    risk_reasons = Counter(reason for c in candidates for reason in c.get("risk_reasons", []))
    by_label = Counter(str(c.get("semantic_label") or "unknown") for c in candidates)
    return {
        "objects": len(objects),
        "semantic_label_counts": dict(labels),
        "status_counts": dict(statuses),
        "candidate_count": len(candidates),
        "candidate_label_counts": dict(by_label),
        "risk_reason_counts": dict(risk_reasons),
        "top_candidates": [
            {k: c.get(k) for k in ("object_id", "semantic_label", "risk_score", "risk_reasons", "target_count", "point_count", "label_votes")}
            for c in candidates[:20]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets-jsonl", type=Path, required=True)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-limit", type=int, default=120)
    parser.add_argument("--evidence-per-object", type=int, default=3)
    parser.add_argument("--crop-margin", type=int, default=48)
    parser.add_argument("--mask-overlay-alpha", type=float, default=0.35)
    parser.add_argument("--thumb-width", type=int, default=360)
    parser.add_argument("--thumb-height", type=int, default=260)
    args = parser.parse_args()

    objects = read_jsonl(args.objects_jsonl)
    targets = read_jsonl(args.targets_jsonl)
    targets_by_id = target_map(targets)
    candidates = select_candidates(objects, args.candidate_limit)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = args.output_dir / "frame_local_object_qa_candidates.jsonl"
    evidence_rows_path = args.output_dir / "frame_local_object_qa_evidence.jsonl"
    crop_paths: list[Path] = []
    with candidates_path.open("w", encoding="utf-8") as cf, evidence_rows_path.open("w", encoding="utf-8") as ef:
        for candidate in candidates:
            evidence_targets = pick_evidence_targets(candidate, targets_by_id, args.evidence_per_object)
            candidate["evidence_target_count"] = len(evidence_targets)
            cf.write(json.dumps(candidate, ensure_ascii=False) + "\n")
            for target in evidence_targets:
                crop_name = f"{candidate['object_id']}_{target['target_id']}.jpg"
                crop_path = args.output_dir / "crops" / crop_name
                made, overlay_status = crop_target_image(args.workdir, candidate, target, crop_path, args.crop_margin, args.mask_overlay_alpha)
                row = {
                    "object_id": candidate["object_id"],
                    "semantic_label": candidate["semantic_label"],
                    "risk_score": candidate["risk_score"],
                    "risk_reasons": candidate["risk_reasons"],
                    "target_id": target.get("target_id"),
                    "target_label": target.get("label"),
                    "frame_id": target.get("frame_id"),
                    "cam_id": target.get("cam_id"),
                    "cluster_size": target.get("cluster_size"),
                    "bbox_2d": target.get("bbox_2d"),
                    "image_path": target.get("image_path"),
                    "mask_path": target.get("mask_path"),
                    "mask_overlay": overlay_status,
                    "mask_overlay_note": "overlay shows the source priority class mask, not the exact 3D-refined child target",
                    "crop_path": str(made) if made else "",
                }
                if made:
                    crop_paths.append(made)
                ef.write(json.dumps(row, ensure_ascii=False) + "\n")

    contact = args.output_dir / "frame_local_object_qa_contact.jpg"
    make_contact_sheet(crop_paths[: min(len(crop_paths), 80)], contact, (args.thumb_width, args.thumb_height))
    report = summarize(objects, candidates)
    report.update({
        "targets": len(targets),
        "candidate_limit": args.candidate_limit,
        "evidence_images": len(crop_paths),
        "candidates_jsonl": str(candidates_path),
        "evidence_jsonl": str(evidence_rows_path),
        "contact_sheet": str(contact) if contact.exists() else "",
    })
    report_path = args.output_dir / "frame_local_object_qa_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
