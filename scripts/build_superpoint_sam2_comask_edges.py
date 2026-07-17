#!/usr/bin/env python3
"""Turn shared-view SAM2 masks into conservative immutable-contact edge evidence."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.sam_rle import decode_rle
except ModuleNotFoundError:  # Supports direct script execution.
    from sam_rle import decode_rle


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def edge_key(object_a: int, object_b: int) -> tuple[int, int]:
    return (min(object_a, object_b), max(object_a, object_b))


def mask_support(
    masks: list[np.ndarray], uv_a: np.ndarray, uv_b: np.ndarray, max_mask_area_ratio: float,
) -> tuple[float, float]:
    """Return compact same-mask support and evidence of two distinct compact masks."""
    if not masks or not len(uv_a) or not len(uv_b):
        return 0.0, 0.0
    height, width = masks[0].shape
    def pixels(uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.clip(np.rint(uv[:, 0]).astype(np.int32), 0, width - 1),
            np.clip(np.rint(uv[:, 1]).astype(np.int32), 0, height - 1),
        )
    xa, ya = pixels(uv_a)
    xb, yb = pixels(uv_b)
    best_a = 0.0
    best_b = 0.0
    same = 0.0
    for mask in masks:
        area_ratio = float(mask.mean())
        if area_ratio > max_mask_area_ratio:
            continue
        compactness = 1.0 - area_ratio
        cover_a = float(mask[ya, xa].mean()) * compactness
        cover_b = float(mask[yb, xb].mean()) * compactness
        best_a = max(best_a, cover_a)
        best_b = max(best_b, cover_b)
        same = max(same, min(cover_a, cover_b))
    separate = max(0.0, min(best_a, best_b) - same)
    return same, separate


def summarize_views(values: list[tuple[float, float]], min_views: int) -> dict[str, float | int]:
    if not values:
        return {"view_count": 0, "same_mask_lcb": 0.0, "separation_lcb": 0.0, "sam2_affinity": 1.0}
    same = np.asarray([item[0] for item in values], dtype=np.float32)
    separate = np.asarray([item[1] for item in values], dtype=np.float32)
    if len(values) < min_views:
        return {
            "view_count": len(values), "same_mask_lcb": 0.0, "separation_lcb": 0.0,
            "sam2_affinity": 1.0,
        }
    same_lcb = max(0.0, float(same.mean() - same.std()))
    separate_lcb = max(0.0, float(separate.mean() - separate.std()))
    return {
        "view_count": len(values),
        "same_mask_lcb": round(same_lcb, 6),
        "separation_lcb": round(separate_lcb, 6),
        "sam2_affinity": round(min(max(1.0 - separate_lcb + same_lcb, 0.0), 1.0), 6),
    }


def choose_view_rows(rows: list[dict[str, Any]]) -> dict[tuple[int, int], dict[int, dict[str, Any]]]:
    by_view: dict[tuple[int, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        view = by_view[(int(row["frame_id"]), int(row["cam_id"]))]
        object_id = int(row["object_id"])
        previous = view.get(object_id)
        row_edge_only = bool(row.get("edge_only"))
        previous_edge_only = bool(previous and previous.get("edge_only"))
        if previous is None or (previous_edge_only and not row_edge_only) or (
            previous_edge_only == row_edge_only
            and int(row.get("projected_points") or 0) > int(previous.get("projected_points") or 0)
        ):
            view[object_id] = row
    return by_view


def report_summary(rows: list[dict[str, Any]], min_views: int) -> dict[str, Any]:
    affinities = sorted(float(row["sam2_affinity"]) for row in rows)
    multi_view = [row for row in rows if int(row["view_count"]) >= min_views]
    def quantile(fraction: float) -> float:
        if not affinities:
            return 1.0
        return round(affinities[int(fraction * (len(affinities) - 1))], 6)
    return {
        "multi_view_edges": len(multi_view),
        "neutral_edges": sum(float(row["sam2_affinity"]) == 1.0 for row in rows),
        "strong_separation_edges": sum(float(row["sam2_affinity"]) < 0.8 for row in multi_view),
        "strong_same_mask_edges": sum(float(row["same_mask_lcb"]) >= 0.5 for row in multi_view),
        "sam2_affinity_quantiles": {"p10": quantile(0.10), "p50": quantile(0.50), "p90": quantile(0.90)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-jsonl", type=Path, required=True)
    parser.add_argument("--contact-edges", type=Path, required=True)
    parser.add_argument("--sam-mask-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-mask-area-ratio", type=float, default=0.60)
    parser.add_argument("--min-views", type=int, default=2)
    args = parser.parse_args()

    contact = {edge_key(int(row["object_a"]), int(row["object_b"])) for row in read_jsonl(args.contact_edges)}
    by_view = choose_view_rows(read_jsonl(args.evidence_jsonl))
    supports: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    masks_loaded = 0
    for (frame_id, cam_id), objects in by_view.items():
        path = args.sam_mask_dir / f"cam{cam_id}_{frame_id:06d}_sam_masks.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        masks = [decode_rle(item["segmentation"]) for item in payload.get("masks", [])]
        masks_loaded += 1
        object_ids = sorted(objects)
        for index, object_a in enumerate(object_ids):
            for object_b in object_ids[index + 1:]:
                key = edge_key(object_a, object_b)
                if key not in contact:
                    continue
                uv_a = np.asarray(objects[object_a].get("projected_uv_samples") or [], dtype=np.float32).reshape(-1, 2)
                uv_b = np.asarray(objects[object_b].get("projected_uv_samples") or [], dtype=np.float32).reshape(-1, 2)
                supports[key].append(mask_support(masks, uv_a, uv_b, args.max_mask_area_ratio))

    rows = [{"object_a": key[0], "object_b": key[1], **summarize_views(values, args.min_views)}
            for key, values in sorted(supports.items())]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = {
        "candidate_contact_edges": len(contact), "sam_mask_views_loaded": masks_loaded,
        "shared_view_edges": len(rows), **report_summary(rows, args.min_views),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
