#!/usr/bin/env python3
"""Run GroundingDINO on object image evidence crops.

This is a crop-level visual confirmation stage for priority fine-object
candidates. It consumes:

- candidate JSONL, usually `needs_visual_review.jsonl`
- image evidence JSONL from `build_object_image_evidence.py`

and writes per-object detections, annotated images, a contact sheet, and a
summary report. It intentionally reviews candidates rather than rewriting point
clouds; the output can later be consumed by a label-refinement script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROMPTS = {
    "car": ["car", "vehicle", "parked car", "van", "truck"],
    "railing": ["railing", "guardrail", "handrail", "metal fence", "fence"],
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_path(raw_path: str, workdir: Path, evidence_dir: Path) -> Path:
    p = Path(raw_path)
    if p.is_absolute() and p.exists():
        return p
    for base in (workdir, evidence_dir, evidence_dir.parent):
        q = base / p
        if q.exists():
            return q
    return workdir / p


def load_model(config_path: Path, weights_path: Path, device: str):
    patch_transformers_bert_head_mask()
    from groundingdino.util.inference import load_model

    return load_model(str(config_path), str(weights_path), device=device)


def patch_transformers_bert_head_mask() -> None:
    """GroundingDINO 0.4 expects an older BertModel.get_head_mask helper.

    Newer transformers builds no longer expose it on BertModel/PreTrainedModel.
    The detector only needs the standard attention-head mask expansion, so keep a
    local compatibility patch instead of pinning or downgrading the environment.
    """

    import torch
    from transformers import BertModel

    original_get_extended_attention_mask = BertModel.get_extended_attention_mask

    def get_extended_attention_mask(self, attention_mask, input_shape, device_or_dtype=None):
        device = device_or_dtype if isinstance(device_or_dtype, torch.device) else None
        dtype = None if device is not None else device_or_dtype
        mask = original_get_extended_attention_mask(self, attention_mask, input_shape, dtype=dtype)
        return mask.to(device=device) if device is not None else mask

    BertModel.get_extended_attention_mask = get_extended_attention_mask

    if hasattr(BertModel, "get_head_mask"):
        return

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked: bool = False):
        if head_mask is None:
            return [None] * num_hidden_layers
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        if is_attention_chunked:
            head_mask = head_mask.unsqueeze(-1)
        dtype = next(self.parameters()).dtype if any(True for _ in self.parameters()) else torch.float32
        return head_mask.to(dtype=dtype)

    BertModel.get_head_mask = get_head_mask


def run_predict(model, image_path: Path, caption: str, box_threshold: float, text_threshold: float, device: str):
    from groundingdino.util.inference import load_image, predict
    from torchvision.ops import box_convert

    image_source, image = load_image(str(image_path))
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=caption,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )
    h, w = image_source.shape[:2]
    detections = []
    if len(boxes):
        xyxy = box_convert(boxes=boxes * boxes.new_tensor([w, h, w, h]), in_fmt="cxcywh", out_fmt="xyxy").numpy()
        for box, score, phrase in zip(xyxy, logits.numpy(), phrases):
            detections.append({
                "bbox_xyxy": [float(x) for x in box.tolist()],
                "score": float(score),
                "phrase": str(phrase),
            })
    return image_source, detections


def detection_matches_label(detection: dict[str, Any], label: str) -> bool:
    phrase = str(detection.get("phrase", "")).lower()
    terms = PROMPTS.get(label, [label])
    return any(term in phrase or phrase in term for term in terms)


def annotate(image_rgb: np.ndarray, detections: list[dict[str, Any]], output_path: Path) -> None:
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for det in detections:
        x0, y0, x1, y1 = [int(round(v)) for v in det["bbox_xyxy"]]
        score = float(det["score"])
        phrase = str(det.get("phrase", ""))
        color = (50, 220, 90) if score >= 0.35 else (0, 180, 255)
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
        cv2.putText(image, f"{phrase} {score:.2f}", (max(0, x0), max(16, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def make_contact_sheet(rows: list[dict[str, Any]], output_path: Path, thumb_size: int = 180, cols: int = 6) -> None:
    thumbs = []
    for row in rows:
        if int(row.get("rank", 999)) != 1:
            continue
        image = cv2.imread(row["annotated_path"], cv2.IMREAD_COLOR)
        if image is None:
            continue
        h, w = image.shape[:2]
        scale = min(thumb_size / max(w, 1), thumb_size / max(h, 1))
        resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))))
        canvas = np.zeros((thumb_size + 48, thumb_size, 3), dtype=np.uint8)
        y = (thumb_size - resized.shape[0]) // 2
        x = (thumb_size - resized.shape[1]) // 2
        canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        cv2.putText(canvas, f"{row['object_id']} {row['semantic_label']}", (4, thumb_size + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{row['visual_status']} {row['best_score']:.2f}", (4, thumb_size + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 220, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"f{row.get('frame_id')} c{row.get('cam_id')}", (4, thumb_size + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (170, 170, 170), 1, cv2.LINE_AA)
        thumbs.append(canvas)
    if not thumbs:
        return
    rows_count = int(np.ceil(len(thumbs) / cols))
    sheet = np.zeros((rows_count * (thumb_size + 48), cols * thumb_size, 3), dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet[r * (thumb_size + 48):(r + 1) * (thumb_size + 48), c * thumb_size:(c + 1) * thumb_size] = thumb
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates-jsonl", type=Path, required=True)
    parser.add_argument("--evidence-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("/root/epfs/vlm_seg_project/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"))
    parser.add_argument("--weights", type=Path, default=Path("/root/epfs/vlm_seg_project/weights/groundingdino_swint_ogc.pth"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--box-threshold", type=float, default=0.28)
    parser.add_argument("--text-threshold", type=float, default=0.22)
    parser.add_argument("--confirm-threshold", type=float, default=0.34)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--limit-objects", type=int, default=0)
    parser.add_argument("--workdir", type=Path, default=Path("."))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = read_jsonl(args.candidates_jsonl)
    if args.limit_objects:
        candidates = candidates[:args.limit_objects]
    candidate_by_id = {int(row["object_id"]): row for row in candidates}
    evidence_rows = read_jsonl(args.evidence_jsonl)
    evidence_by_object: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in evidence_rows:
        oid = int(row["object_id"])
        if oid in candidate_by_id:
            evidence_by_object[oid].append(row)
    for rows in evidence_by_object.values():
        rows.sort(key=lambda r: int(r.get("rank", 999)))

    model = load_model(args.config, args.weights, args.device)
    review_rows = []
    object_rows = []
    evidence_dir = args.evidence_jsonl.parent

    for oid, obj in sorted(candidate_by_id.items()):
        label = str(obj.get("semantic_label") or obj.get("semantic_label_original") or "")
        prompts = PROMPTS.get(label, [label])
        caption = ". ".join(prompts)
        best_score = 0.0
        best_phrase = ""
        best_matched = False
        rows = evidence_by_object.get(oid, [])[:args.top_k]
        for row in rows:
            image_path = resolve_path(str(row.get("crop_path", "")), args.workdir, evidence_dir)
            if not image_path.exists():
                review_rows.append({
                    "object_id": oid,
                    "semantic_label": label,
                    "rank": row.get("rank"),
                    "visual_status": "missing_crop",
                    "crop_path": str(image_path),
                    "detections": [],
                    "best_score": 0.0,
                })
                continue
            image_rgb, detections = run_predict(model, image_path, caption, args.box_threshold, args.text_threshold, args.device)
            matched = [d for d in detections if detection_matches_label(d, label)]
            local_best = max([float(d["score"]) for d in matched], default=0.0)
            if local_best > best_score:
                best_score = local_best
                best_phrase = max(matched, key=lambda d: float(d["score"])).get("phrase", "") if matched else ""
                best_matched = bool(matched)
            visual_status = "visual_confirmed" if local_best >= args.confirm_threshold else ("visual_detected_weak" if local_best > 0 else "visual_not_detected")
            annotated_path = args.output_dir / "objects" / str(oid) / (Path(str(row.get("crop_path", f"rank{row.get('rank', 0)}.jpg"))).stem + "_gdino.jpg")
            annotate(image_rgb, detections, annotated_path)
            out = {
                "object_id": oid,
                "semantic_label": label,
                "rank": int(row.get("rank", 999)),
                "frame_id": row.get("frame_id"),
                "cam_id": row.get("cam_id"),
                "crop_path": str(image_path),
                "annotated_path": str(annotated_path),
                "caption": caption,
                "detections": detections,
                "matched_detection_count": len(matched),
                "best_score": float(local_best),
                "visual_status": visual_status,
            }
            review_rows.append(out)

        if best_matched and best_score >= args.confirm_threshold:
            object_status = "visual_confirmed"
        elif best_score > 0:
            object_status = "visual_weak"
        else:
            object_status = "visual_not_detected"
        object_rows.append({
            "object_id": oid,
            "semantic_label": label,
            "priority_guard_status": obj.get("priority_guard_status", ""),
            "priority_guard_reasons": obj.get("priority_guard_reasons", []),
            "visual_status": object_status,
            "best_score": float(best_score),
            "best_phrase": best_phrase,
            "evidence_count": len(rows),
        })

    (args.output_dir / "groundingdino_evidence_rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in review_rows),
        encoding="utf-8",
    )
    (args.output_dir / "groundingdino_object_review.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in object_rows),
        encoding="utf-8",
    )
    make_contact_sheet(review_rows, args.output_dir / "groundingdino_review_contact.jpg")
    report = {
        "candidates_jsonl": str(args.candidates_jsonl),
        "evidence_jsonl": str(args.evidence_jsonl),
        "output_dir": str(args.output_dir),
        "candidate_count": len(candidates),
        "reviewed_object_count": len(object_rows),
        "evidence_rows": len(review_rows),
        "visual_status_counts": dict(Counter(row["visual_status"] for row in object_rows)),
        "label_status_counts": dict(Counter(f"{row['semantic_label']}:{row['visual_status']}" for row in object_rows)),
        "params": {
            "config": str(args.config),
            "weights": str(args.weights),
            "device": args.device,
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "confirm_threshold": args.confirm_threshold,
            "top_k": args.top_k,
        },
    }
    (args.output_dir / "groundingdino_review_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
