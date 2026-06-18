#!/usr/bin/env python3
"""Build DINO patch-similarity maps from projected 3D point seeds.

This is a small-sample diagnostic for the fine-object route.  It does not
classify objects.  It asks a narrower question:

Given a projected 3D object seed inside an image crop, do DINO patch features
expand over a coherent visual region, or do they bleed into wall/floor context?

That makes it useful for railing/car candidates where global object labels are
already noisy and bbox-level crops are too loose.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_dino_feature_evidence_qa import (  # noqa: E402
    infer_patch_grids,
    load_feature_model,
    model_runtime_name,
    point_patch_mask,
    read_jsonl,
    resolve_path,
)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize01(values: np.ndarray) -> np.ndarray:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def feature_prototype(feats: np.ndarray, seed_mask: np.ndarray) -> np.ndarray | None:
    flat = feats.reshape(-1, feats.shape[-1])
    seed = seed_mask.reshape(-1)
    if int(seed.sum()) < 1:
        return None
    proto = flat[seed].mean(axis=0)
    norm = float(np.linalg.norm(proto))
    if norm < 1e-6:
        return None
    return (proto / norm).astype(np.float32)


def connected_component_stats(mask: np.ndarray, seed_mask: np.ndarray) -> dict[str, Any]:
    mask8 = mask.astype(np.uint8)
    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask8, connectivity=8)
    if n <= 1:
        return {
            "component_count": 0,
            "largest_component_patches": 0,
            "seed_component_patch_ratio": 0.0,
            "seed_component_retained": False,
        }

    seed_labels = labels[seed_mask]
    seed_labels = seed_labels[seed_labels > 0]
    if len(seed_labels):
        seed_component = int(Counter(seed_labels.tolist()).most_common(1)[0][0])
    else:
        seed_component = 0
    largest = int(stats[1:, cv2.CC_STAT_AREA].max())
    seed_area = int(stats[seed_component, cv2.CC_STAT_AREA]) if seed_component > 0 else 0
    total = int(mask8.sum())
    return {
        "component_count": int(n - 1),
        "largest_component_patches": largest,
        "seed_component_patch_ratio": round(seed_area / max(total, 1), 6),
        "seed_component_retained": bool(seed_component > 0),
    }


def patch_seed_mask(
    row: dict[str, Any],
    original_size: tuple[int, int],
    processed_size: tuple[int, int],
    grid_shape: tuple[int, int],
    dilation: int,
) -> np.ndarray:
    return point_patch_mask(
        row.get("projected_uv_samples") or [],
        row.get("crop_bbox_xyxy") or [0, 0, original_size[0], original_size[1]],
        original_size,
        processed_size,
        grid_shape,
        dilation,
    )


def upsample_patch_map(patch_values: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    return cv2.resize(patch_values.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)


def draw_overlay(
    image: Image.Image,
    similarity: np.ndarray,
    seed_mask: np.ndarray,
    foreground_mask: np.ndarray,
    title: str,
) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    heat = (normalize01(upsample_patch_map(similarity, image.size)) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(rgb, 0.58, heat_color, 0.42, 0)

    seed_up = upsample_patch_map(seed_mask.astype(np.float32), image.size) > 0.5
    fg_up = upsample_patch_map(foreground_mask.astype(np.float32), image.size) > 0.5
    overlay[fg_up] = (0.70 * overlay[fg_up] + 0.30 * np.array([255, 255, 255])).astype(np.uint8)
    overlay[seed_up] = (0.35 * overlay[seed_up] + 0.65 * np.array([0, 255, 80])).astype(np.uint8)

    canvas = overlay.copy()
    cv2.putText(canvas, title[:110], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, title[:110], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def make_contact_sheet(image_paths: list[Path], output_path: Path, thumb_w: int = 320, cols: int = 3) -> None:
    if not image_paths:
        return
    thumbs = []
    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = thumb_w / max(w, 1)
        thumb_h = max(1, int(round(h * scale)))
        thumbs.append(cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA))
    if not thumbs:
        return
    max_h = max(t.shape[0] for t in thumbs)
    rows = []
    for i in range(0, len(thumbs), cols):
        row = thumbs[i:i + cols]
        padded = []
        for thumb in row:
            if thumb.shape[0] < max_h:
                pad = np.zeros((max_h - thumb.shape[0], thumb.shape[1], 3), dtype=np.uint8)
                thumb = np.vstack([thumb, pad])
            padded.append(thumb)
        while len(padded) < cols:
            padded.append(np.zeros((max_h, thumb_w, 3), dtype=np.uint8))
        rows.append(np.hstack(padded))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.vstack(rows))


def row_label(row: dict[str, Any]) -> str:
    return str(row.get("dino_prompt_group") or row.get("candidate_label") or row.get("semantic_label") or "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="facebook/dinov2-small")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workdir", type=Path, default=Path("."))
    parser.add_argument("--labels", nargs="*", default=["railing", "car"])
    parser.add_argument("--max-rows", type=int, default=30)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed-dilation", type=int, default=1)
    parser.add_argument("--similarity-percentile", type=float, default=82.0)
    parser.add_argument("--min-threshold", type=float, default=0.48)
    args = parser.parse_args()

    rows = read_jsonl(args.evidence_jsonl)
    labels = set(args.labels)
    selected = [
        row for row in rows
        if int(row.get("rank", 999)) == args.rank and (not labels or row_label(row) in labels)
    ][: args.max_rows]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    processor, model = load_feature_model(args.model_id, args.device)
    evidence_dir = args.evidence_jsonl.parent

    out_rows: list[dict[str, Any]] = []
    overlay_paths: list[Path] = []
    for offset in range(0, len(selected), max(1, args.batch_size)):
        batch = selected[offset: offset + max(1, args.batch_size)]
        images: list[Image.Image] = []
        valid_rows: list[tuple[dict[str, Any], Path]] = []
        for row in batch:
            crop_path = resolve_path(str(row.get("crop_path", "")), args.workdir, evidence_dir)
            if not crop_path.exists():
                continue
            try:
                images.append(Image.open(crop_path).convert("RGB"))
                valid_rows.append((row, crop_path))
            except Exception:
                continue
        inferred = infer_patch_grids(processor, model, images, args.device)
        for (row, crop_path), image, (feats, processed_size, original_size) in zip(valid_rows, images, inferred):
            grid_shape = feats.shape[:2]
            seed_mask = patch_seed_mask(row, original_size, processed_size, grid_shape, args.seed_dilation)
            proto = feature_prototype(feats, seed_mask)
            if proto is None:
                result = {
                    "object_id": row.get("object_id"),
                    "semantic_label": row.get("semantic_label"),
                    "dino_prompt_group": row.get("dino_prompt_group"),
                    "rank": row.get("rank"),
                    "crop_path": str(crop_path),
                    "status": "no_seed_patches",
                }
                out_rows.append(result)
                continue

            sims = feats.reshape(-1, feats.shape[-1]) @ proto
            sim_grid = sims.reshape(grid_shape)
            seed_values = sim_grid[seed_mask]
            context_values = sim_grid[~seed_mask]
            dynamic_threshold = float(np.percentile(sim_grid, args.similarity_percentile))
            threshold = max(float(args.min_threshold), dynamic_threshold)
            fg_mask = sim_grid >= threshold
            cc = connected_component_stats(fg_mask, seed_mask)
            context_p95 = float(np.percentile(context_values, 95)) if len(context_values) else 0.0
            seed_mean = float(seed_values.mean()) if len(seed_values) else 0.0
            bleed_risk = bool(context_p95 >= seed_mean - 0.02 or cc["seed_component_patch_ratio"] < 0.45)

            title = (
                f"obj {row.get('object_id')} {row_label(row)} "
                f"seed={int(seed_mask.sum())} fg={int(fg_mask.sum())} "
                f"ctx95={context_p95:.3f} risk={int(bleed_risk)}"
            )
            overlay = draw_overlay(image, sim_grid, seed_mask, fg_mask, title)
            rel_dir = args.output_dir / "objects" / str(row.get("object_id"))
            rel_dir.mkdir(parents=True, exist_ok=True)
            overlay_path = rel_dir / f"obj{row.get('object_id')}_rank{row.get('rank')}_dino_seed_similarity.jpg"
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            overlay_paths.append(overlay_path)

            result = {
                "object_id": row.get("object_id"),
                "frame_id": row.get("frame_id"),
                "cam_id": row.get("cam_id"),
                "rank": row.get("rank"),
                "semantic_label": row.get("semantic_label"),
                "dino_prompt_group": row.get("dino_prompt_group"),
                "crop_path": str(crop_path),
                "overlay_path": str(overlay_path),
                "seed_patch_count": int(seed_mask.sum()),
                "foreground_patch_count": int(fg_mask.sum()),
                "similarity_threshold": round(threshold, 6),
                "seed_similarity_mean": round(seed_mean, 6),
                "seed_similarity_min": round(float(seed_values.min()) if len(seed_values) else 0.0, 6),
                "context_similarity_p95": round(context_p95, 6),
                "bleed_risk": bleed_risk,
                **cc,
            }
            out_rows.append(result)

    write_jsonl(args.output_dir / "dino_seed_similarity_maps.jsonl", out_rows)
    make_contact_sheet(overlay_paths, args.output_dir / "dino_seed_similarity_contact.jpg")
    report = {
        "evidence_jsonl": str(args.evidence_jsonl),
        "model_id": args.model_id,
        "runtime": model_runtime_name(model),
        "device": args.device,
        "selected_rows": len(selected),
        "output_rows": len(out_rows),
        "status_counts": dict(Counter(str(row.get("status", "ok")) for row in out_rows)),
        "label_counts": dict(Counter(str(row.get("dino_prompt_group") or row.get("semantic_label")) for row in out_rows)),
        "bleed_risk_rows": sum(1 for row in out_rows if row.get("bleed_risk")),
        "contact_sheet": str(args.output_dir / "dino_seed_similarity_contact.jpg") if overlay_paths else None,
    }
    (args.output_dir / "dino_seed_similarity_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
