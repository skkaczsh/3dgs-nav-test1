#!/usr/bin/env python3
"""Build a reusable QA pack for fine-object labels.

The full-scene route now separates:

- final fine labels (`car`, `railing`)
- visually confirmed but geometry-rejected candidates
- unconfirmed fine candidates

This script ranks those objects by review risk, writes JSONL/CSV summaries, and
optionally builds contact sheets when the evidence crop/overlay images are
available locally.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return []
    return out


def geometry_features(obj: dict[str, Any]) -> dict[str, float]:
    extent_raw = float_list(obj.get("extent"))
    extent = sorted(extent_raw, reverse=True)
    if len(extent) != 3:
        extent = [0.0, 0.0, 0.0]
    eig = float_list(obj.get("pca_eigenvalues"))
    if len(eig) == 3 and eig[0] > 1e-12:
        linearity = (eig[0] - eig[1]) / eig[0]
        planarity_pca = (eig[1] - eig[2]) / eig[0]
        scattering = eig[2] / eig[0]
    else:
        linearity = 0.0
        planarity_pca = float(obj.get("planarity") or 0.0)
        scattering = 0.0
    normal = float_list(obj.get("pca_normal"))
    normal_z_abs = abs(normal[2]) if len(normal) == 3 else 0.0
    point_count = float(obj.get("point_count") or 0.0)
    volume = max(extent[0] * extent[1] * extent[2], 1e-6)
    return {
        "extent_x": extent_raw[0] if len(extent_raw) > 0 else 0.0,
        "extent_y": extent_raw[1] if len(extent_raw) > 1 else 0.0,
        "extent_z": extent_raw[2] if len(extent_raw) > 2 else 0.0,
        "extent_max": extent[0],
        "extent_mid": extent[1],
        "extent_min": extent[2],
        "extent_mid_ratio": extent[1] / extent[0] if extent[0] > 0 else 0.0,
        "extent_min_ratio": extent[2] / extent[0] if extent[0] > 0 else 0.0,
        "linearity": linearity,
        "planarity_pca": planarity_pca,
        "scattering": scattering,
        "normal_z_abs": normal_z_abs,
        "thickness_rms": float(obj.get("thickness_rms") or 0.0),
        "point_count": point_count,
        "bbox_density": point_count / volume,
    }


def risk_for_object(obj: dict[str, Any], features: dict[str, float]) -> tuple[float, list[str]]:
    label = str(obj.get("semantic_label") or "unknown")
    candidate_label = str(obj.get("candidate_label") or "")
    status = str(obj.get("status") or "")
    score = float(obj.get("visual_review_best_score") or 0.0)
    reasons: list[str] = []
    risk = 0.0

    if label in {"car", "railing"}:
        risk += 1.0
        if score < 0.45:
            risk += 2.0
            reasons.append("low_visual_score")
        if label == "car":
            if features["extent_max"] > 7.0:
                risk += 1.5
                reasons.append("large_car_extent")
            if features["extent_min"] < 0.18 and features["extent_max"] > 1.2:
                risk += 2.0
                reasons.append("thin_car_shape")
            if features["normal_z_abs"] > 0.9 and features["extent_min"] < 0.45:
                risk += 2.0
                reasons.append("horizontal_car_fragment")
            if features["point_count"] > 18000 and features["planarity_pca"] > 0.35:
                risk += 1.0
                reasons.append("large_planar_car_component")
        if label == "railing":
            if features["extent_max"] > 6.0:
                risk += 1.0
                reasons.append("long_railing_component")
            if features["thickness_rms"] > 0.18:
                risk += 1.5
                reasons.append("thick_railing_component")
            if features["normal_z_abs"] > 0.55:
                risk += 2.0
                reasons.append("railing_horizontal_or_slanted_surface")
            if features["extent_mid_ratio"] > 0.75 and features["planarity_pca"] > 0.35:
                risk += 1.5
                reasons.append("railing_plane_like")
            if features["point_count"] > 6000:
                risk += 1.0
                reasons.append("large_railing_point_count")
    elif label == "fine_candidate":
        risk += 0.5
        if candidate_label in {"car", "railing"}:
            risk += 1.0
            reasons.append("unresolved_priority_candidate")
        if status.startswith("geometry_rejected_visual_confirmed"):
            risk += 3.0
            reasons.append("visual_confirmed_but_geometry_rejected")
        elif status.startswith("unconfirmed_"):
            risk += 1.0
            reasons.append("no_visual_confirmation")
    if not reasons:
        reasons.append("low_risk")
    return risk, reasons


def resolve_evidence_path(raw_path: str, root: Path) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return root / p


def make_contact_sheet(rows: list[dict[str, Any]], evidence_root: Path, output_path: Path, image_key: str) -> bool:
    try:
        import cv2
        import numpy as np
    except Exception:
        return make_contact_sheet_pil(rows, evidence_root, output_path, image_key)

    thumbs = []
    for row in rows:
        raw_path = row.get(image_key)
        if not raw_path:
            continue
        path = resolve_evidence_path(str(raw_path), evidence_root)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        h, w = image.shape[:2]
        thumb_size = 220
        scale = min(thumb_size / max(w, 1), thumb_size / max(h, 1))
        resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))))
        canvas = np.zeros((thumb_size + 62, thumb_size, 3), dtype=np.uint8)
        y = (thumb_size - resized.shape[0]) // 2
        x = (thumb_size - resized.shape[1]) // 2
        canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        lines = [
            f"{row['object_id']} {row['semantic_label']} risk {row['risk_score']:.1f}",
            f"cand {row.get('candidate_label') or '-'} score {row.get('visual_score', 0):.2f}",
            ",".join(row.get("risk_reasons", []))[:34],
        ]
        for i, text in enumerate(lines):
            cv2.putText(canvas, text, (4, thumb_size + 17 + i * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (230, 230, 230), 1, cv2.LINE_AA)
        thumbs.append(canvas)
    if not thumbs:
        return False
    cols = 5
    rows_count = math.ceil(len(thumbs) / cols)
    sheet = np.zeros((rows_count * (220 + 62), cols * 220, 3), dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet[r * 282:(r + 1) * 282, c * 220:(c + 1) * 220] = thumb
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)
    return True


def make_contact_sheet_pil(rows: list[dict[str, Any]], evidence_root: Path, output_path: Path, image_key: str) -> bool:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return False

    thumb_size = 220
    caption_h = 62
    cols = 5
    thumbs = []
    for row in rows:
        raw_path = row.get(image_key)
        if not raw_path:
            continue
        path = resolve_evidence_path(str(raw_path), evidence_root)
        if not path.exists():
            continue
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            continue
        image.thumbnail((thumb_size, thumb_size))
        canvas = Image.new("RGB", (thumb_size, thumb_size + caption_h), (0, 0, 0))
        x = (thumb_size - image.width) // 2
        y = (thumb_size - image.height) // 2
        canvas.paste(image, (x, y))
        draw = ImageDraw.Draw(canvas)
        lines = [
            f"{row['object_id']} {row['semantic_label']} risk {row['risk_score']:.1f}",
            f"cand {row.get('candidate_label') or '-'} score {row.get('visual_score', 0):.2f}",
            ",".join(row.get("risk_reasons", []))[:34],
        ]
        for i, text in enumerate(lines):
            draw.text((4, thumb_size + 4 + i * 17), text, fill=(235, 235, 235))
        thumbs.append(canvas)
    if not thumbs:
        return False
    rows_count = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * thumb_size, rows_count * (thumb_size + caption_h)), (0, 0, 0))
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet.paste(thumb, (c * thumb_size, r * (thumb_size + caption_h)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--evidence-jsonl", type=Path, required=True)
    parser.add_argument("--visual-review-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, default=Path("."))
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    objects = read_jsonl(args.objects_jsonl)
    visual_by_id = {int(row["object_id"]): row for row in read_jsonl(args.visual_review_jsonl)}
    evidence_by_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.evidence_jsonl):
        evidence_by_id[int(row["object_id"])].append(row)
    for rows in evidence_by_id.values():
        rows.sort(key=lambda r: int(r.get("rank", 999)))

    qa_rows: list[dict[str, Any]] = []
    evidence_paths: list[str] = []
    for obj in objects:
        label = str(obj.get("semantic_label") or "unknown")
        candidate_label = str(obj.get("candidate_label") or "")
        if label not in {"car", "railing", "fine_candidate"} and candidate_label not in {"car", "railing"}:
            continue
        object_id = int(obj["object_id"])
        features = geometry_features(obj)
        risk, reasons = risk_for_object(obj, features)
        visual = visual_by_id.get(object_id, {})
        evidence_rows = evidence_by_id.get(object_id, [])
        best_evidence = evidence_rows[0] if evidence_rows else {}
        out = {
            "object_id": object_id,
            "semantic_label": label,
            "candidate_label": candidate_label,
            "status": obj.get("status", ""),
            "candidate_status": obj.get("candidate_status", ""),
            "risk_score": round(risk, 3),
            "risk_reasons": reasons,
            "point_count": int(obj.get("point_count") or 0),
            "visual_status": visual.get("visual_status") or obj.get("visual_review_status", ""),
            "visual_score": float(visual.get("best_score") or obj.get("visual_review_best_score") or 0.0),
            "best_phrase": visual.get("best_phrase") or obj.get("visual_review_best_phrase", ""),
            "evidence_count": len(evidence_rows),
            "crop_path": best_evidence.get("crop_path", ""),
            "overlay_path": best_evidence.get("overlay_path", ""),
            "frame_id": best_evidence.get("frame_id"),
            "cam_id": best_evidence.get("cam_id"),
            **{k: round(v, 6) for k, v in features.items()},
        }
        qa_rows.append(out)
        for ev in evidence_rows[:3]:
            for key in ("crop_path", "overlay_path"):
                if ev.get(key):
                    evidence_paths.append(str(ev[key]))

    qa_rows.sort(key=lambda row: (-float(row["risk_score"]), -int(row["point_count"]), int(row["object_id"])))
    write_jsonl(args.output_dir / "fine_object_qa_ranked.jsonl", qa_rows)
    fieldnames = list(qa_rows[0].keys()) if qa_rows else []
    with (args.output_dir / "fine_object_qa_ranked.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(qa_rows)
    (args.output_dir / "evidence_files.txt").write_text("\n".join(sorted(set(evidence_paths))) + "\n", encoding="utf-8")

    top_rows = qa_rows[:args.top_k]
    write_jsonl(args.output_dir / "fine_object_qa_top.jsonl", top_rows)
    made_crop_sheet = make_contact_sheet(top_rows, args.evidence_root, args.output_dir / "fine_object_qa_top_crops.jpg", "crop_path")
    made_overlay_sheet = make_contact_sheet(top_rows, args.evidence_root, args.output_dir / "fine_object_qa_top_overlays.jpg", "overlay_path")

    summary = {
        "objects_jsonl": str(args.objects_jsonl),
        "evidence_jsonl": str(args.evidence_jsonl),
        "visual_review_jsonl": str(args.visual_review_jsonl),
        "output_dir": str(args.output_dir),
        "qa_object_count": len(qa_rows),
        "label_counts": dict(Counter(row["semantic_label"] for row in qa_rows)),
        "candidate_label_counts": dict(Counter(row["candidate_label"] or "none" for row in qa_rows)),
        "status_counts": dict(Counter(row["status"] for row in qa_rows)),
        "top_k": args.top_k,
        "contact_sheets": {
            "crops": made_crop_sheet,
            "overlays": made_overlay_sheet,
        },
        "top_risk_objects": [
            {
                "object_id": row["object_id"],
                "semantic_label": row["semantic_label"],
                "candidate_label": row["candidate_label"],
                "risk_score": row["risk_score"],
                "risk_reasons": row["risk_reasons"],
                "point_count": row["point_count"],
                "visual_score": row["visual_score"],
            }
            for row in top_rows[:20]
        ],
    }
    (args.output_dir / "fine_object_qa_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
