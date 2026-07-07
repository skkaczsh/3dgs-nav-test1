#!/usr/bin/env python3
"""Diagnose spatial candidate edges versus stored region-input edges."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_geo_patch_graph import (  # noqa: E402
    BUCKET_IDS,
    bucket_guard_veto,
    edge_keep,
    positive_offsets,
    sorted_linear_index,
)
from scripts.optimize_patch_graph_energy import read_region_input  # noqa: E402


def load_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def params_from_report(report: dict[str, Any]) -> dict[str, Any]:
    params = report.get("params")
    if not isinstance(params, dict):
        params = {}
    return params


def voxel_keys(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    keys = np.floor(xyz / voxel_size).astype(np.int64, copy=False)
    keys -= keys.min(axis=0)
    return keys


def reason_counts(
    arrays: dict[str, np.ndarray],
    src: np.ndarray,
    dst: np.ndarray,
    keep: np.ndarray,
    params: dict[str, Any],
) -> Counter[str]:
    if len(src) == 0:
        return Counter()
    rgb_dist = np.linalg.norm(arrays["rgb"][src] - arrays["rgb"][dst], axis=1)
    dz = np.abs(arrays["xyz"][src, 2] - arrays["xyz"][dst, 2])
    rough_delta = np.abs(arrays["roughness"][src] - arrays["roughness"][dst])
    color_std_delta = np.abs(arrays["local_color_std"][src] - arrays["local_color_std"][dst])
    plane_residual = np.abs(np.sum((arrays["xyz"][dst] - arrays["xyz"][src]) * arrays["normal"][src], axis=1))

    max_color = float(params.get("max_color_distance", 150.0))
    max_height = float(params.get("max_height_delta", 0.3))
    max_plane = float(params.get("max_plane_residual", 0.26))
    bucket_guard = str(params.get("bucket_guard", "same-bucket-or-fine-color"))
    failed = ~keep

    counts: Counter[str] = Counter()
    checks = [
        ("bucket_veto", bucket_guard_veto(arrays["buckets"][src], arrays["buckets"][dst], bucket_guard)),
        ("color_texture_veto", (rgb_dist > max_color * 1.75) & (color_std_delta > 55.0)),
        ("roughness_veto", rough_delta > 0.36),
        ("height_veto", dz > max_height * 3.0),
        ("plane_veto", plane_residual > max_plane * 3.5),
    ]
    remaining = failed.copy()
    for name, mask in checks:
        hit = remaining & mask
        counts[name] = int(np.count_nonzero(hit))
        remaining &= ~mask
    counts["score_or_other"] = int(np.count_nonzero(remaining))
    return counts


def iter_spatial_pairs(arrays: dict[str, np.ndarray], radius: int) -> tuple[int, np.ndarray, np.ndarray]:
    keys = arrays["keys"]
    linear, order, sorted_linear, (stride_x, stride_y) = sorted_linear_index(keys)
    n = len(keys)
    src_chunks: list[np.ndarray] = []
    dst_chunks: list[np.ndarray] = []
    total = 0
    for dx, dy, dz in positive_offsets(radius):
        query = linear + dx * stride_x + dy * stride_y + dz
        pos = np.searchsorted(sorted_linear, query)
        in_bounds = pos < n
        safe_pos = np.minimum(pos, n - 1)
        found = in_bounds & (sorted_linear[safe_pos] == query)
        total += int(np.count_nonzero(found))
        if np.any(found):
            src_chunks.append(np.nonzero(found)[0].astype(np.int32))
            dst_chunks.append(order[safe_pos[found]].astype(np.int32))
    if not src_chunks:
        return 0, np.empty(0, np.int32), np.empty(0, np.int32)
    return total, np.concatenate(src_chunks), np.concatenate(dst_chunks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = load_report(args.report)
    params = params_from_report(report)
    arrays, stored_src, stored_dst = read_region_input(args.region_input)
    arrays["keys"] = voxel_keys(arrays["xyz"], float(report.get("voxel_size") or params.get("voxel_size") or 0.1))

    total_spatial, src, dst = iter_spatial_pairs(arrays, int(params.get("connect_radius_voxels", 2)))
    weights = {
        "texture": float(params.get("texture_weight", 0.12)),
        "shape": float(params.get("shape_weight", 0.30)),
        "height": float(params.get("height_weight", 0.12)),
        "bucket": float(params.get("bucket_weight", 0.14)),
        "normal": float(params.get("normal_weight", 0.12)),
        "plane": float(params.get("plane_weight", 0.20)),
    }
    keep = edge_keep(
        arrays,
        src,
        dst,
        float(params.get("min_edge_score", 0.46)),
        float(params.get("max_color_distance", 150.0)),
        float(params.get("max_height_delta", 0.30)),
        float(params.get("max_normal_angle", 120.0)),
        float(params.get("max_plane_residual", 0.26)),
        str(params.get("bucket_guard", "same-bucket-or-fine-color")),
        weights,
        float(params.get("color_bridge_distance_factor", 0.70)),
        float(params.get("color_bridge_texture_delta", 42.0)),
    )
    bucket_pairs = Counter(
        tuple(sorted((int(a), int(b))))
        for a, b in zip(arrays["buckets"][src], arrays["buckets"][dst], strict=True)
    )
    kept_bucket_pairs = Counter(
        tuple(sorted((int(a), int(b))))
        for a, b in zip(arrays["buckets"][src[keep]], arrays["buckets"][dst[keep]], strict=True)
    )
    bucket_names = {int(v): k for k, v in BUCKET_IDS.items()}
    out = {
        "schema": "region-input-edge-diagnosis/v1",
        "region_input": str(args.region_input),
        "report": str(args.report),
        "voxel_count": int(len(arrays["xyz"])),
        "connect_radius_voxels": int(params.get("connect_radius_voxels", 2)),
        "spatial_candidate_edges": int(total_spatial),
        "recomputed_kept_edges": int(np.count_nonzero(keep)),
        "stored_edges": int(len(stored_src)),
        "stored_vs_recomputed_delta": int(len(stored_src) - np.count_nonzero(keep)),
        "kept_ratio": float(np.count_nonzero(keep) / max(total_spatial, 1)),
        "reject_reason_counts": dict(reason_counts(arrays, src, dst, keep, params)),
        "candidate_bucket_pairs_top": [
            {"pair": [bucket_names.get(a, str(a)), bucket_names.get(b, str(b))], "count": int(c)}
            for (a, b), c in bucket_pairs.most_common(12)
        ],
        "kept_bucket_pairs_top": [
            {"pair": [bucket_names.get(a, str(a)), bucket_names.get(b, str(b))], "count": int(c)}
            for (a, b), c in kept_bucket_pairs.most_common(12)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
