#!/usr/bin/env python3
"""Measure repeated image-space color boundaries on immutable contact edges.

This is edge evidence only. It never creates labels or changes Superpoint
ownership. A strong, repeatable image contrast lowers an existing 3D contact
edge's smoothing affinity; a weak or missing observation leaves that edge
unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def edge_key(object_a: int, object_b: int) -> tuple[int, int]:
    return (min(object_a, object_b), max(object_a, object_b))


def sample_view_contrast(
    image: np.ndarray,
    uv_a: np.ndarray,
    uv_b: np.ndarray,
    max_pairs: int,
    max_pixel_gap: float,
) -> float | None:
    """Return the mean RGB contrast of nearest cross-object projected samples."""
    if not len(uv_a) or not len(uv_b):
        return None
    diff = uv_a[:, None, :] - uv_b[None, :, :]
    squared_distance = np.sum(diff * diff, axis=2)
    flat = squared_distance.ravel()
    take = min(max_pairs, len(flat))
    nearest = np.argpartition(flat, take - 1)[:take]
    keep = nearest[flat[nearest] <= max_pixel_gap * max_pixel_gap]
    if not len(keep):
        return None
    a_idx, b_idx = np.unravel_index(keep, squared_distance.shape)
    height, width = image.shape[:2]
    xy_a = np.rint(uv_a[a_idx]).astype(np.int32)
    xy_b = np.rint(uv_b[b_idx]).astype(np.int32)
    valid = (
        (xy_a[:, 0] >= 0) & (xy_a[:, 0] < width) & (xy_a[:, 1] >= 0) & (xy_a[:, 1] < height)
        & (xy_b[:, 0] >= 0) & (xy_b[:, 0] < width) & (xy_b[:, 1] >= 0) & (xy_b[:, 1] < height)
    )
    if not np.any(valid):
        return None
    rgb_a = image[xy_a[valid, 1], xy_a[valid, 0], ::-1].astype(np.float32)
    rgb_b = image[xy_b[valid, 1], xy_b[valid, 0], ::-1].astype(np.float32)
    return float(np.mean(np.linalg.norm(rgb_a - rgb_b, axis=1)))


def summarize_contrasts(contrasts: list[float], sigma: float, min_views: int) -> dict[str, float | int]:
    """Use a lower confidence bound so one high-contrast view cannot cut an edge."""
    values = np.asarray(contrasts, dtype=np.float32)
    view_count = int(len(values))
    if not view_count:
        return {"view_count": 0, "contrast_mean": 0.0, "contrast_lcb": 0.0, "photometric_affinity": 1.0}
    mean = float(values.mean())
    lcb = max(0.0, mean - float(values.std())) if view_count >= min_views else 0.0
    affinity = math.exp(-0.5 * (lcb / max(sigma, 1e-6)) ** 2)
    return {
        "view_count": view_count,
        "contrast_mean": round(mean, 6),
        "contrast_lcb": round(lcb, 6),
        "photometric_affinity": round(affinity, 6),
    }


def build_rows(
    evidence_rows: list[dict[str, Any]],
    contact_edges: list[dict[str, Any]],
    max_pairs: int = 64,
    max_pixel_gap: float = 12.0,
    color_sigma: float = 45.0,
    min_views: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    contact = {edge_key(int(row["object_a"]), int(row["object_b"])) for row in contact_edges}
    by_view: dict[tuple[int, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in evidence_rows:
        by_view[(int(row["frame_id"]), int(row["cam_id"]))][int(row["object_id"])] = row

    contrasts: dict[tuple[int, int], list[float]] = defaultdict(list)
    images: dict[str, np.ndarray] = {}
    for members in by_view.values():
        object_ids = sorted(members)
        for index, object_a in enumerate(object_ids):
            for object_b in object_ids[index + 1:]:
                key = edge_key(object_a, object_b)
                if key not in contact:
                    continue
                first, second = members[object_a], members[object_b]
                image_path = str(first.get("image_path") or "")
                if image_path != str(second.get("image_path") or ""):
                    continue
                if image_path not in images:
                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    if image is None:
                        continue
                    images[image_path] = image
                contrast = sample_view_contrast(
                    images[image_path],
                    np.asarray(first.get("projected_uv_samples") or [], dtype=np.float32).reshape(-1, 2),
                    np.asarray(second.get("projected_uv_samples") or [], dtype=np.float32).reshape(-1, 2),
                    max_pairs,
                    max_pixel_gap,
                )
                if contrast is not None:
                    contrasts[key].append(contrast)

    rows = [
        {"object_a": key[0], "object_b": key[1], **summarize_contrasts(values, color_sigma, min_views)}
        for key, values in sorted(contrasts.items())
    ]
    return rows, {
        "contact_edges": len(contact),
        "shared_view_edges": len(rows),
        "views": len(by_view),
        "images_loaded": len(images),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-jsonl", type=Path, required=True)
    parser.add_argument("--contact-edges", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=64)
    parser.add_argument("--max-pixel-gap", type=float, default=12.0)
    parser.add_argument("--color-sigma", type=float, default=45.0)
    parser.add_argument("--min-views", type=int, default=2)
    args = parser.parse_args()
    rows, report = build_rows(
        read_jsonl(args.evidence_jsonl), read_jsonl(args.contact_edges), args.max_pairs,
        args.max_pixel_gap, args.color_sigma, args.min_views,
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
