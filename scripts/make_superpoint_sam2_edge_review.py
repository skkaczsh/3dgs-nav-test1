#!/usr/bin/env python3
"""Render concise human QA sheets for high-impact SAM2 contact-edge cuts.

This is deliberately a review artifact, not another segmentation stage. It
uses the same first-touch-visible sample pixels that produced the edge evidence
and never changes Superpoint ownership or semantic labels.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from scripts.sam_rle import decode_rle


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def edge_key(row: dict[str, Any]) -> tuple[int, int]:
    return tuple(sorted((int(row["object_a"]), int(row["object_b"]))))


def choose_shared_views(
    edge: tuple[int, int], evidence_rows: list[dict[str, Any]], max_views: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Select shared views by the weaker object's first-touch support."""
    by_view: dict[tuple[int, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in evidence_rows:
        object_id = int(row["object_id"])
        if object_id not in edge:
            continue
        key = (int(row["frame_id"]), int(row["cam_id"]))
        previous = by_view[key].get(object_id)
        if previous is None or int(row.get("projected_points") or 0) > int(previous.get("projected_points") or 0):
            by_view[key][object_id] = row
    shared = [
        (rows[edge[0]], rows[edge[1]]) for rows in by_view.values()
        if edge[0] in rows and edge[1] in rows
    ]
    shared.sort(key=lambda pair: min(
        int(pair[0].get("projected_points") or 0), int(pair[1].get("projected_points") or 0),
    ), reverse=True)
    return shared[:max_views]


def sampled_pixels(row: dict[str, Any], width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    uv = np.asarray(row.get("projected_uv_samples") or [], dtype=np.float32).reshape(-1, 2)
    if not len(uv):
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    x = np.clip(np.rint(uv[:, 0]).astype(np.int32), 0, width - 1)
    y = np.clip(np.rint(uv[:, 1]).astype(np.int32), 0, height - 1)
    return x, y


def best_mask(masks: list[np.ndarray], row: dict[str, Any]) -> np.ndarray | None:
    if not masks:
        return None
    height, width = masks[0].shape
    x, y = sampled_pixels(row, width, height)
    if not len(x):
        return None
    return max(masks, key=lambda mask: float(mask[y, x].mean()))


def tint(image: np.ndarray, mask: np.ndarray | None, bgr: tuple[int, int, int], alpha: float) -> None:
    if mask is None:
        return
    image[mask] = np.round((1.0 - alpha) * image[mask] + alpha * np.asarray(bgr)).astype(np.uint8)


def draw_points(image: np.ndarray, row: dict[str, Any], bgr: tuple[int, int, int]) -> None:
    height, width = image.shape[:2]
    x, y = sampled_pixels(row, width, height)
    for px, py in zip(x, y):
        cv2.circle(image, (int(px), int(py)), 2, bgr, thickness=-1, lineType=cv2.LINE_AA)


def sheet_for_view(
    edge: tuple[int, int], edge_row: dict[str, Any], row_a: dict[str, Any], row_b: dict[str, Any], mask_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    source = Path(str(row_a.get("image_path") or ""))
    raw = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if raw is None:
        return None
    frame_id, cam_id = int(row_a["frame_id"]), int(row_a["cam_id"])
    mask_path = mask_dir / f"cam{cam_id}_{frame_id:06d}_sam_masks.json"
    if not mask_path.exists():
        return None
    payload = json.loads(mask_path.read_text(encoding="utf-8"))
    masks = [decode_rle(item["segmentation"]) for item in payload.get("masks", [])]
    projection = raw.copy()
    draw_points(projection, row_a, (0, 0, 255))
    draw_points(projection, row_b, (0, 255, 255))
    sam = raw.copy()
    tint(sam, best_mask(masks, row_a), (255, 0, 255), 0.36)
    tint(sam, best_mask(masks, row_b), (0, 255, 0), 0.36)
    draw_points(sam, row_a, (0, 0, 255))
    draw_points(sam, row_b, (0, 255, 255))
    title = (
        f"edge {edge[0]}-{edge[1]}  cam{cam_id} frame {frame_id}  "
        f"sam2={float(edge_row['sam2_affinity']):.3f}"
    )
    for panel in (projection, sam):
        cv2.rectangle(panel, (0, 0), (min(panel.shape[1], 920), 36), (0, 0, 0), thickness=-1)
        cv2.putText(panel, title, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return np.hstack((projection, sam)), {
        "frame_id": frame_id,
        "cam_id": cam_id,
        "image_path": str(source),
        "mask_path": str(mask_path),
        "projected_points": {str(edge[0]): int(row_a.get("projected_points") or 0), str(edge[1]): int(row_b.get("projected_points") or 0)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam2-edges", type=Path, required=True)
    parser.add_argument("--evidence-jsonl", type=Path, required=True)
    parser.add_argument("--sam-mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-affinity", type=float, default=0.8)
    parser.add_argument("--max-edges", type=int, default=20)
    parser.add_argument("--max-views-per-edge", type=int, default=3)
    args = parser.parse_args()

    evidence = read_jsonl(args.evidence_jsonl)
    candidates = [row for row in read_jsonl(args.sam2_edges) if float(row.get("sam2_affinity") or 1.0) < args.max_affinity]
    candidates.sort(key=lambda row: float(row["sam2_affinity"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for index, edge_row in enumerate(candidates[:args.max_edges], start=1):
        edge = edge_key(edge_row)
        panels: list[np.ndarray] = []
        view_rows: list[dict[str, Any]] = []
        for row_a, row_b in choose_shared_views(edge, evidence, args.max_views_per_edge):
            rendered = sheet_for_view(edge, edge_row, row_a, row_b, args.sam_mask_dir)
            if rendered is None:
                continue
            panel, metadata = rendered
            panels.append(panel)
            view_rows.append(metadata)
        if not panels:
            continue
        path = args.output_dir / f"edge_{index:02d}_{edge[0]}_{edge[1]}.jpg"
        cv2.imwrite(str(path), np.vstack(panels))
        manifest.append({
            "edge": list(edge),
            "sam2_affinity": float(edge_row["sam2_affinity"]),
            "view_count": int(edge_row.get("view_count") or 0),
            "sheet": str(path),
            "views": view_rows,
        })
    report = {"candidate_edges": len(candidates), "rendered_edges": len(manifest), "items": manifest}
    (args.output_dir / "sam2_edge_review_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"candidate_edges": len(candidates), "rendered_edges": len(manifest)}))


if __name__ == "__main__":
    main()
