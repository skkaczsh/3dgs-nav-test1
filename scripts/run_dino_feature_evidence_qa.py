#!/usr/bin/env python3
"""Use DINO-style dense features to score fine-object evidence crops.

This is not a detector. It checks whether the projected object bounding box is
visually coherent and distinct from surrounding context. That addresses the
current failure mode where GroundingDINO confirms a label somewhere in a loose
crop, while the projected 3D object may actually be wall/floor/stair structure.

The script works with DINOv3 when access is available. It can also run DINOv2 as
a public feature-backbone sanity check with the same interface.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


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


def resolve_path(raw_path: str, workdir: Path, evidence_dir: Path) -> Path:
    p = Path(raw_path)
    if p.is_absolute() and p.exists():
        return p
    for base in (workdir, evidence_dir, evidence_dir.parent):
        q = base / p
        if q.exists():
            return q
    return workdir / p


@dataclass
class OnnxFeatureModel:
    session: Any
    input_name: str
    patch_size: int


def model_runtime_name(model: Any) -> str:
    if isinstance(model, OnnxFeatureModel):
        return "onnxruntime:" + ",".join(model.session.get_providers())
    return "torch"


def load_feature_model(model_id: str, device: str):
    from transformers import AutoImageProcessor, AutoModel

    model_path = Path(model_id)
    onnx_path = model_path / "onnx" / "model_quantized.onnx"
    if model_path.exists() and onnx_path.exists():
        import onnxruntime as ort

        processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
        available = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if device.startswith("cuda") and "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        input_name = session.get_inputs()[0].name
        patch_size = int(getattr(getattr(processor, "size", None), "get", lambda _k, d=None: d)("patch_size", 16) or 16)
        config_path = model_path / "config.json"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                patch_size = int(json.load(f).get("patch_size") or patch_size)
        return processor, OnnxFeatureModel(session=session, input_name=input_name, patch_size=patch_size)

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval().to(device)
    return processor, model


def infer_patch_grid(processor, model, image: Image.Image, device: str) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    return infer_patch_grids(processor, model, [image], device)[0]


def infer_patch_grids(processor, model, images: list[Image.Image], device: str) -> list[tuple[np.ndarray, tuple[int, int], tuple[int, int]]]:
    if not images:
        return []
    if isinstance(model, OnnxFeatureModel):
        inputs = processor(images=images, return_tensors="np")
        pixel_values = inputs["pixel_values"].astype(np.float32)
        input_h, input_w = int(pixel_values.shape[-2]), int(pixel_values.shape[-1])
        outputs = model.session.run(None, {model.input_name: pixel_values})
        tokens = np.asarray(outputs[0], dtype=np.float32)
        patch_tokens = tokens[:, 1:, :]
        n = int(patch_tokens.shape[1])
        patch_size = model.patch_size
        gh = max(1, input_h // patch_size)
        gw = max(1, input_w // patch_size)
        if gh * gw != n:
            # Some DINOv3 ONNX exports include register tokens after CLS.
            register_count = n - gh * gw
            if register_count > 0:
                patch_tokens = patch_tokens[:, register_count:, :]
                n = int(patch_tokens.shape[1])
            if gh * gw != n:
                side = int(round(math.sqrt(n)))
                if side * side == n:
                    gh = gw = side
                else:
                    gh, gw = 1, n
        results: list[tuple[np.ndarray, tuple[int, int], tuple[int, int]]] = []
        for i, image in enumerate(images):
            feats = patch_tokens[i]
            feats = feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-6)
            results.append((feats.reshape(gh, gw, -1), (input_w, input_h), image.size))
        return results

    inputs = processor(images=images, return_tensors="pt").to(device)
    pixel_values = inputs["pixel_values"]
    input_h, input_w = int(pixel_values.shape[-2]), int(pixel_values.shape[-1])
    with torch.no_grad():
        output = model(**inputs)
    tokens = output.last_hidden_state.detach()
    patch_tokens = tokens[:, 1:, :].float()
    n = int(patch_tokens.shape[1])

    patch_size = int(getattr(model.config, "patch_size", 14) or 14)
    gh = max(1, input_h // patch_size)
    gw = max(1, input_w // patch_size)
    if gh * gw != n:
        side = int(round(math.sqrt(n)))
        if side * side == n:
            gh = gw = side
        else:
            # Fall back to a single-row layout; metrics will be conservative.
            gh, gw = 1, n
    results = []
    for i, image in enumerate(images):
        feats = patch_tokens[i].cpu().numpy()
        feats = feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-6)
        results.append((feats.reshape(gh, gw, -1), (input_w, input_h), image.size))
    return results


def roi_patch_mask(
    bbox_xyxy: list[float],
    crop_bbox_xyxy: list[float],
    original_size: tuple[int, int],
    processed_size: tuple[int, int],
    grid_shape: tuple[int, int],
) -> np.ndarray:
    ox0, oy0, _ox1, _oy1 = crop_bbox_xyxy
    x0, y0, x1, y1 = bbox_xyxy
    rel = np.array([x0 - ox0, y0 - oy0, x1 - ox0, y1 - oy0], dtype=np.float32)
    orig_w, orig_h = original_size
    proc_w, proc_h = processed_size
    scale = np.array([
        proc_w / max(orig_w, 1),
        proc_h / max(orig_h, 1),
        proc_w / max(orig_w, 1),
        proc_h / max(orig_h, 1),
    ], dtype=np.float32)
    rel *= scale
    gh, gw = grid_shape
    xs = (np.arange(gw) + 0.5) * proc_w / gw
    ys = (np.arange(gh) + 0.5) * proc_h / gh
    xx, yy = np.meshgrid(xs, ys)
    mask = (xx >= rel[0]) & (xx <= rel[2]) & (yy >= rel[1]) & (yy <= rel[3])
    return mask


def point_patch_mask(
    uv_samples: list[list[float]],
    crop_bbox_xyxy: list[float],
    original_size: tuple[int, int],
    processed_size: tuple[int, int],
    grid_shape: tuple[int, int],
    dilation: int,
) -> np.ndarray:
    mask = np.zeros(grid_shape, dtype=bool)
    if not uv_samples:
        return mask
    ox0, oy0, _ox1, _oy1 = crop_bbox_xyxy
    orig_w, orig_h = original_size
    proc_w, proc_h = processed_size
    gh, gw = grid_shape
    scale_x = proc_w / max(orig_w, 1)
    scale_y = proc_h / max(orig_h, 1)
    radius = max(0, int(dilation))
    for sample in uv_samples:
        if len(sample) < 2:
            continue
        px = (float(sample[0]) - ox0) * scale_x
        py = (float(sample[1]) - oy0) * scale_y
        cx = int(np.floor(px * gw / max(proc_w, 1)))
        cy = int(np.floor(py * gh / max(proc_h, 1)))
        if cx < 0 or cx >= gw or cy < 0 or cy >= gh:
            continue
        y0 = max(0, cy - radius)
        y1 = min(gh, cy + radius + 1)
        x0 = max(0, cx - radius)
        x1 = min(gw, cx + radius + 1)
        mask[y0:y1, x0:x1] = True
    return mask


def feature_metrics(feats: np.ndarray, roi_mask: np.ndarray) -> dict[str, float]:
    flat = feats.reshape(-1, feats.shape[-1])
    roi = roi_mask.reshape(-1)
    if int(roi.sum()) < 2:
        return {
            "roi_patch_count": float(roi.sum()),
            "roi_coherence": 0.0,
            "context_separation": 0.0,
            "context_similarity": 1.0,
            "feature_risk": 1.0,
        }
    roi_feats = flat[roi]
    ctx_feats = flat[~roi]
    roi_mean = roi_feats.mean(axis=0)
    roi_mean /= max(float(np.linalg.norm(roi_mean)), 1e-6)
    roi_sims = roi_feats @ roi_mean
    roi_coherence = float(np.mean(roi_sims))
    if len(ctx_feats) > 0:
        ctx_mean = ctx_feats.mean(axis=0)
        ctx_mean /= max(float(np.linalg.norm(ctx_mean)), 1e-6)
        context_similarity = float(roi_mean @ ctx_mean)
    else:
        context_similarity = 0.0
    context_separation = float(roi_coherence - context_similarity)
    feature_risk = 0.0
    if roi_coherence < 0.55:
        feature_risk += 0.5
    if context_separation < 0.06:
        feature_risk += 0.5
    return {
        "roi_patch_count": float(roi.sum()),
        "roi_coherence": roi_coherence,
        "context_similarity": context_similarity,
        "context_separation": context_separation,
        "feature_risk": feature_risk,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-jsonl", type=Path, required=True)
    parser.add_argument("--visual-review-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, default=Path("."))
    parser.add_argument("--model-id", default="facebook/dinov2-small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--roi-source", choices=["auto", "bbox", "points"], default="auto")
    parser.add_argument("--point-dilation", type=int, default=1)
    parser.add_argument("--min-point-patches", type=int, default=2)
    parser.add_argument("--limit-objects", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_rows = read_jsonl(args.evidence_jsonl)
    visual_rows = read_jsonl(args.visual_review_jsonl)
    visual_by_id = {int(row["object_id"]): row for row in visual_rows}
    evidence_by_object: dict[int, list[dict[str, Any]]] = {}
    for row in evidence_rows:
        evidence_by_object.setdefault(int(row["object_id"]), []).append(row)
    for rows in evidence_by_object.values():
        rows.sort(key=lambda r: int(r.get("rank", 999)))
    object_ids = sorted(evidence_by_object)
    if args.limit_objects:
        object_ids = object_ids[: args.limit_objects]

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    try:
        processor, model = load_feature_model(args.model_id, device)
    except Exception as exc:
        report = {
            "model_id": args.model_id,
            "status": "model_load_failed",
            "error": repr(exc),
        }
        (args.output_dir / "dino_feature_qa_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    evidence_dir = args.evidence_jsonl.parent
    work_items: list[tuple[int, dict[str, Any], dict[str, Any], Path]] = []
    for object_id in object_ids:
        visual = visual_by_id.get(object_id, {})
        for row in evidence_by_object[object_id][: args.top_k]:
            crop_path = resolve_path(str(row.get("crop_path", "")), args.workdir, evidence_dir)
            if not crop_path.exists():
                continue
            work_items.append((object_id, visual, row, crop_path))

    out_rows: list[dict[str, Any]] = []
    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(work_items), batch_size):
        batch = work_items[start:start + batch_size]
        images: list[Image.Image] = []
        valid_items: list[tuple[int, dict[str, Any], dict[str, Any], Path]] = []
        for item in batch:
            _object_id, _visual, _row, crop_path = item
            try:
                images.append(Image.open(crop_path).convert("RGB"))
                valid_items.append(item)
            except Exception:
                continue
        grids = infer_patch_grids(processor, model, images, device)
        for (object_id, visual, row, crop_path), (feats, processed_size, original_size) in zip(valid_items, grids):
            bbox_mask = roi_patch_mask(
                row["bbox_xyxy"],
                row["crop_bbox_xyxy"],
                original_size,
                processed_size,
                feats.shape[:2],
            )
            point_mask = point_patch_mask(
                row.get("projected_uv_samples") or [],
                row["crop_bbox_xyxy"],
                original_size,
                processed_size,
                feats.shape[:2],
                args.point_dilation,
            )
            if args.roi_source == "points":
                roi_mask = point_mask
                roi_source = "points"
            elif args.roi_source == "bbox":
                roi_mask = bbox_mask
                roi_source = "bbox"
            elif int(point_mask.sum()) >= args.min_point_patches:
                roi_mask = point_mask
                roi_source = "points"
            else:
                roi_mask = bbox_mask
                roi_source = "bbox_fallback"
            metrics = feature_metrics(feats, roi_mask)
            out_rows.append({
                "object_id": object_id,
                "semantic_label": row.get("semantic_label", ""),
                "visual_status": visual.get("visual_status", ""),
                "visual_score": float(visual.get("best_score") or 0.0),
                "best_phrase": visual.get("best_phrase", ""),
                "rank": int(row.get("rank", 999)),
                "frame_id": row.get("frame_id"),
                "cam_id": row.get("cam_id"),
                "crop_path": str(crop_path),
                "bbox_xyxy": row.get("bbox_xyxy"),
                "crop_bbox_xyxy": row.get("crop_bbox_xyxy"),
                "bbox_area_ratio": row.get("bbox_area_ratio"),
                "bbox_inlier_ratio": row.get("bbox_inlier_ratio"),
                "roi_source": roi_source,
                "point_patch_count": int(point_mask.sum()),
                **{k: round(float(v), 6) for k, v in metrics.items()},
            })

    write_jsonl(args.output_dir / "dino_feature_evidence_qa.jsonl", out_rows)
    risky = [row for row in out_rows if float(row.get("feature_risk", 0.0)) >= 0.5]
    report = {
        "model_id": args.model_id,
        "device": device,
        "runtime": model_runtime_name(model),
        "batch_size": batch_size,
        "roi_source": args.roi_source,
        "point_dilation": args.point_dilation,
        "min_point_patches": args.min_point_patches,
        "evidence_jsonl": str(args.evidence_jsonl),
        "visual_review_jsonl": str(args.visual_review_jsonl),
        "output_dir": str(args.output_dir),
        "object_count": len(object_ids),
        "row_count": len(out_rows),
        "feature_risky_rows": len(risky),
        "visual_status_counts": dict(Counter(row.get("visual_status", "") for row in out_rows)),
        "label_counts": dict(Counter(row.get("semantic_label", "") for row in out_rows)),
        "top_feature_risk": sorted(
            [
                {
                    "object_id": row["object_id"],
                    "semantic_label": row["semantic_label"],
                    "visual_status": row["visual_status"],
                    "visual_score": row["visual_score"],
                    "roi_coherence": row["roi_coherence"],
                    "context_separation": row["context_separation"],
                    "feature_risk": row["feature_risk"],
                }
                for row in out_rows
            ],
            key=lambda r: (-(float(r["feature_risk"])), float(r["context_separation"])),
        )[:30],
    }
    (args.output_dir / "dino_feature_qa_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
