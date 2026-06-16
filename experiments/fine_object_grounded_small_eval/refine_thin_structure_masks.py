#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def skeletonize(mask: np.ndarray) -> np.ndarray:
    work = (mask > 0).astype(np.uint8) * 255
    skel = np.zeros_like(work)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(work) > 0:
        eroded = cv2.erode(work, kernel)
        opened = cv2.dilate(eroded, kernel)
        residue = cv2.subtract(work, opened)
        skel = cv2.bitwise_or(skel, residue)
        work = eroded
    return skel > 0


def component_aspect(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if len(xs) < 2:
        return 1.0
    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    short = max(min(w, h), 1e-6)
    long = max(w, h)
    return float(long / short)


def keep_components(
    mask: np.ndarray,
    *,
    min_component_px: int,
    min_component_aspect: float,
    fallback_keep_topk: int,
) -> tuple[np.ndarray, dict]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    kept = np.zeros_like(mask, dtype=bool)
    rows = []
    for comp_id in range(1, count):
        comp = labels == comp_id
        area = int(stats[comp_id, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        aspect = component_aspect(comp)
        rows.append(
            {
                "component_id": comp_id,
                "area": area,
                "aspect": aspect,
                "keep": bool(area >= min_component_px and aspect >= min_component_aspect),
            }
        )
    for row in rows:
        if row["keep"]:
            kept |= labels == row["component_id"]
    if not kept.any() and rows:
        rows_sorted = sorted(rows, key=lambda r: (r["aspect"], r["area"]), reverse=True)
        for row in rows_sorted[: max(1, fallback_keep_topk)]:
            kept |= labels == row["component_id"]
            row["keep"] = True
    return kept, {"components": rows, "component_count": len(rows)}


def refine_mask(
    mask: np.ndarray,
    *,
    dilate_px: int,
    min_component_px: int,
    min_component_aspect: float,
    fallback_keep_topk: int,
) -> tuple[np.ndarray, dict]:
    binary = mask > 0
    original_area = int(binary.sum())
    if original_area == 0:
        return binary, {
            "original_area": 0,
            "skeleton_px": 0,
            "refined_area": 0,
            "area_ratio": 0.0,
            "mode": "empty",
        }

    skel = skeletonize(binary.astype(np.uint8))
    skeleton_px = int(skel.sum())
    if skeleton_px == 0:
        return binary, {
            "original_area": original_area,
            "skeleton_px": 0,
            "refined_area": original_area,
            "area_ratio": 1.0,
            "mode": "fallback_original_no_skeleton",
        }

    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        band = cv2.dilate(skel.astype(np.uint8) * 255, kernel) > 0
    else:
        band = skel
    refined = binary & band
    refined, comp_info = keep_components(
        refined,
        min_component_px=min_component_px,
        min_component_aspect=min_component_aspect,
        fallback_keep_topk=fallback_keep_topk,
    )
    refined_area = int(refined.sum())
    if refined_area == 0:
        refined = binary
        refined_area = original_area
        mode = "fallback_original_empty_after_component_filter"
    else:
        mode = "skeleton_band"
    return refined, {
        "original_area": original_area,
        "skeleton_px": skeleton_px,
        "refined_area": refined_area,
        "area_ratio": float(refined_area / max(original_area, 1)),
        "mode": mode,
        **comp_info,
    }


def update_row(row: dict, refined_mask_path: Path, refined: np.ndarray, stats: dict) -> dict:
    out = dict(row)
    h, w = refined.shape[:2]
    area = int(refined.sum())
    out["mask_path"] = str(refined_mask_path)
    out["mask_area"] = area
    out["mask_area_ratio"] = float(area / max(h * w, 1))
    out["thin_refine"] = stats
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dilate-px", type=int, default=2)
    parser.add_argument("--min-component-px", type=int, default=16)
    parser.add_argument("--min-component-aspect", type=float, default=2.6)
    parser.add_argument("--fallback-keep-topk", type=int, default=2)
    args = parser.parse_args()

    rows = read_jsonl(args.accepted_jsonl)
    refined_rows = []
    summary_rows = []
    mode_counts = Counter()
    for i, row in enumerate(rows):
        mask_path = Path(row["mask_path"])
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            summary_rows.append({"index": i, "sample_id": row.get("sample_id"), "status": "missing_mask", "mask_path": str(mask_path)})
            continue
        refined, stats = refine_mask(
            mask,
            dilate_px=args.dilate_px,
            min_component_px=args.min_component_px,
            min_component_aspect=args.min_component_aspect,
            fallback_keep_topk=args.fallback_keep_topk,
        )
        rel_dir = Path(row.get("sample_id", f"sample_{i:03d}"))
        refined_mask_path = args.output_dir / "masks" / rel_dir / mask_path.name
        refined_mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(refined_mask_path), (refined.astype(np.uint8) * 255))
        refined_row = update_row(row, refined_mask_path, refined, stats)
        refined_rows.append(refined_row)
        mode_counts[stats["mode"]] += 1
        summary_rows.append(
            {
                "index": i,
                "sample_id": row.get("sample_id"),
                "status": "ok",
                "original_area": stats["original_area"],
                "refined_area": stats["refined_area"],
                "area_ratio": stats["area_ratio"],
                "skeleton_px": stats["skeleton_px"],
                "mode": stats["mode"],
                "component_count": stats.get("component_count", 0),
            }
        )

    write_jsonl(args.output_dir / "accepted_detections_refined.jsonl", refined_rows)
    report = {
        "accepted_jsonl": str(args.accepted_jsonl),
        "output_jsonl": str(args.output_dir / "accepted_detections_refined.jsonl"),
        "items": len(summary_rows),
        "ok": int(sum(1 for row in summary_rows if row["status"] == "ok")),
        "mode_counts": dict(mode_counts),
        "mean_area_ratio": float(np.mean([row["area_ratio"] for row in summary_rows if row["status"] == "ok"]) if any(row["status"] == "ok" for row in summary_rows) else 0.0),
        "rows": summary_rows,
        "params": {
            "dilate_px": args.dilate_px,
            "min_component_px": args.min_component_px,
            "min_component_aspect": args.min_component_aspect,
            "fallback_keep_topk": args.fallback_keep_topk,
        },
    }
    (args.output_dir / "refine_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["items", "ok", "mode_counts", "mean_area_ratio"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
