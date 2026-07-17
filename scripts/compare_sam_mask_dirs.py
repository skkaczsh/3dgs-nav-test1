#!/usr/bin/env python3
"""Compare two SAM mask output directories.

This is intended for the SAM2 PyTorch vs SAM2 TensorRT/C++ promotion gate.
It compares already-generated ``*_sam_masks.json`` files and reports mask
count, coverage, and greedy mask-overlap statistics per image.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scripts.sam_rle import decode_rle


@dataclass
class ImageComparison:
    image_id: str
    baseline_masks: int
    candidate_masks: int
    baseline_coverage: float
    candidate_coverage: float
    coverage_delta: float
    matched_masks: int
    mean_matched_iou: float
    median_matched_iou: float
    min_matched_iou: float
    unmatched_baseline_masks: int
    unmatched_candidate_masks: int
    low_iou_matches: int
    status: str


def load_manifest_image_ids(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items", data if isinstance(data, list) else [])
    ids = []
    for item in items:
        if isinstance(item, str):
            ids.append(item)
        elif isinstance(item, dict) and item.get("image_id"):
            ids.append(str(item["image_id"]))
    return ids


def infer_image_ids(baseline_dir: Path, candidate_dir: Path) -> list[str]:
    baseline = {p.name.removesuffix("_sam_masks.json") for p in baseline_dir.glob("*_sam_masks.json")}
    candidate = {p.name.removesuffix("_sam_masks.json") for p in candidate_dir.glob("*_sam_masks.json")}
    return sorted(baseline | candidate)


def mask_array(mask: dict) -> np.ndarray:
    segmentation = mask.get("segmentation")
    if isinstance(segmentation, dict) and "counts" in segmentation and "size" in segmentation:
        arr = decode_rle(segmentation)
    else:
        arr = np.asarray(segmentation, dtype=bool)
    if arr.ndim != 2:
        raise ValueError(f"segmentation must be 2D, got shape={arr.shape}")
    return arr


def load_masks(path: Path, min_area: int) -> list[np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    masks = data.get("masks", [])
    out = []
    for mask in masks:
        area = int(mask.get("area", 0))
        if area < min_area:
            continue
        out.append(mask_array(mask))
    return out


def coverage(masks: list[np.ndarray]) -> float:
    if not masks:
        return 0.0
    owner = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        owner |= mask
    return float(owner.sum() / max(owner.size, 1))


def pairwise_iou(a: list[np.ndarray], b: list[np.ndarray]) -> np.ndarray:
    if not a or not b:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    matrix = np.zeros((len(a), len(b)), dtype=np.float32)
    b_areas = [int(mask.sum()) for mask in b]
    for i, amask in enumerate(a):
        a_area = int(amask.sum())
        for j, bmask in enumerate(b):
            inter = int(np.logical_and(amask, bmask).sum())
            union = a_area + b_areas[j] - inter
            matrix[i, j] = float(inter / union) if union else 0.0
    return matrix


def greedy_match(ious: np.ndarray, threshold: float) -> list[float]:
    if ious.size == 0:
        return []
    work = ious.copy()
    matches = []
    while work.size:
        idx = int(work.argmax())
        score = float(work.flat[idx])
        if score < threshold:
            break
        i, j = np.unravel_index(idx, work.shape)
        matches.append(score)
        work[i, :] = -1.0
        work[:, j] = -1.0
    return matches


def compare_one(
    image_id: str,
    baseline_dir: Path,
    candidate_dir: Path,
    min_area: int,
    iou_threshold: float,
    low_iou_threshold: float,
) -> ImageComparison:
    baseline_path = baseline_dir / f"{image_id}_sam_masks.json"
    candidate_path = candidate_dir / f"{image_id}_sam_masks.json"
    if not baseline_path.exists() or not candidate_path.exists():
        return ImageComparison(
            image_id=image_id,
            baseline_masks=0,
            candidate_masks=0,
            baseline_coverage=0.0,
            candidate_coverage=0.0,
            coverage_delta=0.0,
            matched_masks=0,
            mean_matched_iou=0.0,
            median_matched_iou=0.0,
            min_matched_iou=0.0,
            unmatched_baseline_masks=0,
            unmatched_candidate_masks=0,
            low_iou_matches=0,
            status="missing_baseline" if not baseline_path.exists() else "missing_candidate",
        )

    baseline = load_masks(baseline_path, min_area)
    candidate = load_masks(candidate_path, min_area)
    base_cov = coverage(baseline)
    cand_cov = coverage(candidate)
    matches = greedy_match(pairwise_iou(baseline, candidate), iou_threshold)
    matched = len(matches)
    return ImageComparison(
        image_id=image_id,
        baseline_masks=len(baseline),
        candidate_masks=len(candidate),
        baseline_coverage=base_cov,
        candidate_coverage=cand_cov,
        coverage_delta=cand_cov - base_cov,
        matched_masks=matched,
        mean_matched_iou=float(np.mean(matches)) if matches else 0.0,
        median_matched_iou=float(np.median(matches)) if matches else 0.0,
        min_matched_iou=float(np.min(matches)) if matches else 0.0,
        unmatched_baseline_masks=max(len(baseline) - matched, 0),
        unmatched_candidate_masks=max(len(candidate) - matched, 0),
        low_iou_matches=sum(1 for score in matches if score < low_iou_threshold),
        status="ok",
    )


def summarize(rows: Iterable[ImageComparison]) -> dict:
    rows = list(rows)
    ok = [row for row in rows if row.status == "ok"]
    def mean(attr: str) -> float:
        return float(np.mean([getattr(row, attr) for row in ok])) if ok else 0.0

    return {
        "images": len(rows),
        "ok_images": len(ok),
        "missing_baseline": sum(1 for row in rows if row.status == "missing_baseline"),
        "missing_candidate": sum(1 for row in rows if row.status == "missing_candidate"),
        "mean_baseline_masks": mean("baseline_masks"),
        "mean_candidate_masks": mean("candidate_masks"),
        "mean_baseline_coverage": mean("baseline_coverage"),
        "mean_candidate_coverage": mean("candidate_coverage"),
        "mean_coverage_delta": mean("coverage_delta"),
        "mean_matched_iou": mean("mean_matched_iou"),
        "mean_unmatched_baseline_masks": mean("unmatched_baseline_masks"),
        "mean_unmatched_candidate_masks": mean("unmatched_candidate_masks"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--image-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-area", type=int, default=500)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--low-iou-threshold", type=float, default=0.75)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, default=None)
    args = parser.parse_args()

    image_ids = list(args.image_id)
    if args.manifest:
        image_ids.extend(load_manifest_image_ids(args.manifest))
    if not image_ids:
        image_ids = infer_image_ids(args.baseline_dir, args.candidate_dir)
    image_ids = sorted(dict.fromkeys(image_ids))
    if args.limit:
        image_ids = image_ids[: args.limit]

    rows = [
        compare_one(
            image_id,
            args.baseline_dir,
            args.candidate_dir,
            args.min_area,
            args.iou_threshold,
            args.low_iou_threshold,
        )
        for image_id in image_ids
    ]
    report = {
        "baseline_dir": str(args.baseline_dir),
        "candidate_dir": str(args.candidate_dir),
        "min_area": args.min_area,
        "iou_threshold": args.iou_threshold,
        "low_iou_threshold": args.low_iou_threshold,
        "summary": summarize(rows),
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
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
