#!/usr/bin/env python3
"""Diagnose SAM2 Python baseline vs TensorRT/C++ candidate mask drift.

This complements compare_sam_mask_dirs.py. The compare gate tells whether a
candidate is promotable; this script explains why it is not by measuring union
extra/missing pixels and unmatched-mask size distributions.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scripts.sam_rle import decode_rle


@dataclass
class UnmatchedMaskSummary:
    image_id: str
    side: str
    mask_index: int
    area: int
    area_ratio: float
    bbox_area_ratio: float
    max_iou_with_other_side: float
    touches_image_edge: bool


@dataclass
class ImageParitySummary:
    image_id: str
    baseline_masks: int
    candidate_masks: int
    baseline_coverage: float
    candidate_coverage: float
    coverage_delta: float
    extra_pixel_ratio: float
    missing_pixel_ratio: float
    union_iou: float
    matched_masks: int
    unmatched_baseline_masks: int
    unmatched_candidate_masks: int
    unmatched_baseline_area_ratio: float
    unmatched_candidate_area_ratio: float


def load_manifest_ids(path: Path | None) -> list[str]:
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items", data if isinstance(data, list) else [])
    ids = []
    for item in items:
        if isinstance(item, str):
            ids.append(item)
        elif isinstance(item, dict) and item.get("image_id"):
            ids.append(str(item["image_id"]))
    return ids


def infer_ids(baseline_dir: Path, candidate_dir: Path) -> list[str]:
    baseline = {p.name.removesuffix("_sam_masks.json") for p in baseline_dir.glob("*_sam_masks.json")}
    candidate = {p.name.removesuffix("_sam_masks.json") for p in candidate_dir.glob("*_sam_masks.json")}
    return sorted(baseline & candidate)


def decode_segmentation(segmentation: object) -> np.ndarray:
    if isinstance(segmentation, dict) and "counts" in segmentation and "size" in segmentation:
        return decode_rle(segmentation)
    arr = np.asarray(segmentation, dtype=bool)
    if arr.ndim != 2:
        raise ValueError(f"segmentation must be 2D, got shape={arr.shape}")
    return arr


def load_masks(path: Path, min_area: int) -> list[np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("masks", data if isinstance(data, list) else [])
    masks = []
    for item in items:
        area = int(item.get("area", 0))
        if area < min_area:
            continue
        masks.append(decode_segmentation(item["segmentation"]))
    return masks


def union_mask(masks: list[np.ndarray]) -> np.ndarray:
    if not masks:
        return np.zeros((0, 0), dtype=bool)
    out = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        out |= mask
    return out


def pairwise_iou(a: list[np.ndarray], b: list[np.ndarray]) -> np.ndarray:
    matrix = np.zeros((len(a), len(b)), dtype=np.float32)
    b_areas = [int(mask.sum()) for mask in b]
    for i, amask in enumerate(a):
        a_area = int(amask.sum())
        for j, bmask in enumerate(b):
            inter = int(np.logical_and(amask, bmask).sum())
            union = a_area + b_areas[j] - inter
            matrix[i, j] = float(inter / union) if union else 0.0
    return matrix


def greedy_pairs(ious: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    work = ious.copy()
    pairs: list[tuple[int, int, float]] = []
    while work.size:
        flat_idx = int(work.argmax())
        score = float(work.flat[flat_idx])
        if score < threshold:
            break
        i, j = np.unravel_index(flat_idx, work.shape)
        pairs.append((int(i), int(j), score))
        work[i, :] = -1.0
        work[:, j] = -1.0
    return pairs


def bbox_area_ratio(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0.0
    bbox_area = (int(xs.max()) - int(xs.min()) + 1) * (int(ys.max()) - int(ys.min()) + 1)
    return float(bbox_area / max(mask.size, 1))


def touches_edge(mask: np.ndarray) -> bool:
    if mask.size == 0:
        return False
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def summarize_unmatched(
    image_id: str,
    side: str,
    masks: list[np.ndarray],
    other: list[np.ndarray],
    unmatched: set[int],
    ious: np.ndarray,
) -> list[UnmatchedMaskSummary]:
    out = []
    image_area = masks[0].size if masks else 1
    for idx in sorted(unmatched):
        mask = masks[idx]
        if side == "baseline":
            max_iou = float(ious[idx, :].max()) if ious.shape[1] else 0.0
        else:
            max_iou = float(ious[:, idx].max()) if ious.shape[0] else 0.0
        out.append(
            UnmatchedMaskSummary(
                image_id=image_id,
                side=side,
                mask_index=idx,
                area=int(mask.sum()),
                area_ratio=float(mask.sum() / max(image_area, 1)),
                bbox_area_ratio=bbox_area_ratio(mask),
                max_iou_with_other_side=max_iou,
                touches_image_edge=touches_edge(mask),
            )
        )
    return out


def compare_image(
    image_id: str,
    baseline_dir: Path,
    candidate_dir: Path,
    min_area: int,
    match_iou: float,
) -> tuple[ImageParitySummary, list[UnmatchedMaskSummary]]:
    baseline = load_masks(baseline_dir / f"{image_id}_sam_masks.json", min_area)
    candidate = load_masks(candidate_dir / f"{image_id}_sam_masks.json", min_area)
    base_union = union_mask(baseline)
    cand_union = union_mask(candidate)
    if base_union.size == 0 and cand_union.size != 0:
        base_union = np.zeros_like(cand_union, dtype=bool)
    if cand_union.size == 0 and base_union.size != 0:
        cand_union = np.zeros_like(base_union, dtype=bool)
    image_area = max(int(base_union.size or cand_union.size), 1)

    inter = int(np.logical_and(base_union, cand_union).sum()) if image_area else 0
    union = int(np.logical_or(base_union, cand_union).sum()) if image_area else 0
    extra = int(np.logical_and(cand_union, ~base_union).sum()) if image_area else 0
    missing = int(np.logical_and(base_union, ~cand_union).sum()) if image_area else 0

    ious = pairwise_iou(baseline, candidate)
    pairs = greedy_pairs(ious, match_iou)
    matched_base = {i for i, _, _ in pairs}
    matched_cand = {j for _, j, _ in pairs}
    unmatched_base = set(range(len(baseline))) - matched_base
    unmatched_cand = set(range(len(candidate))) - matched_cand

    unmatched_base_area = sum(int(baseline[i].sum()) for i in unmatched_base)
    unmatched_cand_area = sum(int(candidate[i].sum()) for i in unmatched_cand)
    row = ImageParitySummary(
        image_id=image_id,
        baseline_masks=len(baseline),
        candidate_masks=len(candidate),
        baseline_coverage=float(base_union.sum() / image_area),
        candidate_coverage=float(cand_union.sum() / image_area),
        coverage_delta=float((cand_union.sum() - base_union.sum()) / image_area),
        extra_pixel_ratio=float(extra / image_area),
        missing_pixel_ratio=float(missing / image_area),
        union_iou=float(inter / union) if union else 0.0,
        matched_masks=len(pairs),
        unmatched_baseline_masks=len(unmatched_base),
        unmatched_candidate_masks=len(unmatched_cand),
        unmatched_baseline_area_ratio=float(unmatched_base_area / image_area),
        unmatched_candidate_area_ratio=float(unmatched_cand_area / image_area),
    )
    details = []
    details.extend(summarize_unmatched(image_id, "baseline", baseline, candidate, unmatched_base, ious))
    details.extend(summarize_unmatched(image_id, "candidate", candidate, baseline, unmatched_cand, ious))
    return row, details


def mean(rows: list[ImageParitySummary], attr: str) -> float:
    return float(np.mean([getattr(row, attr) for row in rows])) if rows else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--image-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-area", type=int, default=500)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--unmatched-csv-output", type=Path, default=None)
    args = parser.parse_args()

    image_ids = list(args.image_id)
    image_ids.extend(load_manifest_ids(args.manifest))
    if not image_ids:
        image_ids = infer_ids(args.baseline_dir, args.candidate_dir)
    image_ids = sorted(dict.fromkeys(image_ids))
    if args.limit:
        image_ids = image_ids[: args.limit]

    rows: list[ImageParitySummary] = []
    unmatched: list[UnmatchedMaskSummary] = []
    missing_files = []
    for image_id in image_ids:
        if not (args.baseline_dir / f"{image_id}_sam_masks.json").exists():
            missing_files.append({"image_id": image_id, "side": "baseline"})
            continue
        if not (args.candidate_dir / f"{image_id}_sam_masks.json").exists():
            missing_files.append({"image_id": image_id, "side": "candidate"})
            continue
        row, details = compare_image(image_id, args.baseline_dir, args.candidate_dir, args.min_area, args.match_iou)
        rows.append(row)
        unmatched.extend(details)

    top_extra = sorted(rows, key=lambda row: row.extra_pixel_ratio, reverse=True)[:10]
    top_missing = sorted(rows, key=lambda row: row.missing_pixel_ratio, reverse=True)[:10]
    top_unmatched_baseline = sorted(rows, key=lambda row: row.unmatched_baseline_area_ratio, reverse=True)[:10]
    top_unmatched_candidate = sorted(rows, key=lambda row: row.unmatched_candidate_area_ratio, reverse=True)[:10]
    report = {
        "baseline_dir": str(args.baseline_dir),
        "candidate_dir": str(args.candidate_dir),
        "images_requested": len(image_ids),
        "images_compared": len(rows),
        "missing_files": missing_files[:100],
        "summary": {
            "mean_coverage_delta": mean(rows, "coverage_delta"),
            "mean_extra_pixel_ratio": mean(rows, "extra_pixel_ratio"),
            "mean_missing_pixel_ratio": mean(rows, "missing_pixel_ratio"),
            "mean_union_iou": mean(rows, "union_iou"),
            "mean_unmatched_baseline_masks": mean(rows, "unmatched_baseline_masks"),
            "mean_unmatched_candidate_masks": mean(rows, "unmatched_candidate_masks"),
            "mean_unmatched_baseline_area_ratio": mean(rows, "unmatched_baseline_area_ratio"),
            "mean_unmatched_candidate_area_ratio": mean(rows, "unmatched_candidate_area_ratio"),
        },
        "top_extra_pixel_rows": [asdict(row) for row in top_extra],
        "top_missing_pixel_rows": [asdict(row) for row in top_missing],
        "top_unmatched_baseline_area_rows": [asdict(row) for row in top_unmatched_baseline],
        "top_unmatched_candidate_area_rows": [asdict(row) for row in top_unmatched_candidate],
        "unmatched_detail_top_area": [
            asdict(row)
            for row in sorted(unmatched, key=lambda item: item.area_ratio, reverse=True)[:100]
        ],
        "rows": [asdict(row) for row in rows],
    }

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.csv_output:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else ["image_id"])
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))
    if args.unmatched_csv_output:
        args.unmatched_csv_output.parent.mkdir(parents=True, exist_ok=True)
        with args.unmatched_csv_output.open("w", newline="", encoding="utf-8") as f:
            fieldnames = list(asdict(unmatched[0]).keys()) if unmatched else ["image_id"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in unmatched:
                writer.writerow(asdict(row))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
