#!/usr/bin/env python3
"""Report candidate-edge sparsity for Superpoint Graph clustering."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.optimize_patch_graph_energy import bbox_gap, build_edge_counts, compute_patch_stats, entropy, read_labels, read_region_input


SIZE_BINS = [
    ("1", 1, 1),
    ("2_9", 2, 9),
    ("10_99", 10, 99),
    ("100_999", 100, 999),
    ("1000_9999", 1000, 9999),
    ("10000_plus", 10000, None),
]


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    return int(values[min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))])


def size_bins(values: list[int]) -> dict[str, int]:
    counts = {name: 0 for name, _lo, _hi in SIZE_BINS}
    for value in values:
        for name, lo, hi in SIZE_BINS:
            if value >= lo and (hi is None or value <= hi):
                counts[name] += 1
                break
    return counts


def linearized_cells(xyz: np.ndarray, cell_size: float) -> tuple[np.ndarray, tuple[int, int]]:
    cells = np.floor(xyz.astype(np.float64, copy=False) / max(cell_size, 1e-9)).astype(np.int64, copy=False)
    shifted = cells - cells.min(axis=0)[None, :]
    spans = shifted.max(axis=0) + 1
    stride_y = int(spans[2])
    stride_x = int(spans[1] * spans[2])
    return shifted[:, 0] * stride_x + shifted[:, 1] * stride_y + shifted[:, 2], (stride_x, stride_y)


def diagnose_missing_neighbors(
    arrays,
    labels,
    stats,
    isolated: list[int],
    *,
    top_n: int,
    cell_size: float,
    radius: int,
    max_cells_per_patch: int,
) -> list[dict]:
    if top_n <= 0 or not isolated:
        return []
    selected = sorted(isolated, key=lambda patch_id: stats[patch_id].count, reverse=True)[:top_n]
    linear, (stride_x, stride_y) = linearized_cells(arrays["xyz"], cell_size)
    order = np.argsort(linear, kind="stable")
    sorted_linear = linear[order]
    sorted_labels = labels[order]
    offsets = [
        dx * stride_x + dy * stride_y + dz
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        for dz in range(-radius, radius + 1)
    ]
    out = []
    for patch_id in selected:
        patch_cells = np.unique(linear[labels == patch_id])
        if len(patch_cells) > max_cells_per_patch:
            step = max(1, len(patch_cells) // max_cells_per_patch)
            patch_cells = patch_cells[::step][:max_cells_per_patch]
        neighbor_counts: Counter[int] = Counter()
        for offset in offsets:
            query = np.unique(patch_cells + offset)
            left = np.searchsorted(sorted_linear, query, side="left")
            right = np.searchsorted(sorted_linear, query, side="right")
            found = left < right
            for lo, hi in zip(left[found].tolist(), right[found].tolist(), strict=True):
                for label in sorted_labels[lo:hi].tolist():
                    if int(label) != patch_id:
                        neighbor_counts[int(label)] += 1
        candidates = []
        same_geometry_candidates = []
        for neighbor_id, contact_points in neighbor_counts.most_common(8):
            neighbor = stats.get(neighbor_id)
            if neighbor is None:
                continue
            same_geometry = neighbor.geometry_type == stats[patch_id].geometry_type
            row = {
                "patch_id": int(neighbor_id),
                "contact_points": int(contact_points),
                "voxel_count": int(neighbor.count),
                "geometry_type": neighbor.geometry_type,
                "same_geometry": same_geometry,
                "bbox_gap": bbox_gap(stats[patch_id], neighbor),
            }
            candidates.append(row)
            if same_geometry:
                same_geometry_candidates.append(row)
        if len(same_geometry_candidates) < 8:
            seen = {row["patch_id"] for row in same_geometry_candidates}
            for neighbor_id, contact_points in neighbor_counts.most_common():
                if len(same_geometry_candidates) >= 8:
                    break
                if int(neighbor_id) in seen:
                    continue
                neighbor = stats.get(int(neighbor_id))
                if neighbor is None or neighbor.geometry_type != stats[patch_id].geometry_type:
                    continue
                same_geometry_candidates.append(
                    {
                        "patch_id": int(neighbor_id),
                        "contact_points": int(contact_points),
                        "voxel_count": int(neighbor.count),
                        "geometry_type": neighbor.geometry_type,
                        "same_geometry": True,
                        "bbox_gap": bbox_gap(stats[patch_id], neighbor),
                    }
                )
        out.append(
            {
                "patch_id": int(patch_id),
                "voxel_count": int(stats[patch_id].count),
                "geometry_type": stats[patch_id].geometry_type,
                "sampled_cell_count": int(len(patch_cells)),
                "neighbor_candidate_count": int(len(neighbor_counts)),
                "same_geometry_neighbor_count": int(
                    sum(1 for neighbor_id in neighbor_counts if stats.get(int(neighbor_id)) and stats[int(neighbor_id)].geometry_type == stats[patch_id].geometry_type)
                ),
                "top_neighbors": candidates,
                "top_same_geometry_neighbors": same_geometry_candidates,
            }
        )
    return out


def summarize(
    arrays,
    labels,
    src,
    dst,
    large_isolated_min_voxels: int,
    *,
    neighbor_top_n: int = 0,
    neighbor_cell_size: float = 0.05,
    neighbor_radius: int = 1,
    neighbor_max_cells_per_patch: int = 20000,
) -> dict:
    stats = compute_patch_stats(arrays, labels)
    edges = build_edge_counts(labels, src, dst)
    degree: Counter[int] = Counter()
    for a, b in edges:
        degree[int(a)] += 1
        degree[int(b)] += 1

    patch_ids = set(stats)
    isolated = [patch_id for patch_id in patch_ids if degree[patch_id] == 0]
    voxel_counts = [stat.count for stat in stats.values()]
    isolated_voxel_counts = [stats[patch_id].count for patch_id in isolated]
    edge_shared = [int(shared) for shared in edges.values()]
    total_voxels = sum(voxel_counts)

    report = {
        "schema": "superpoint-graph-edge-sparsity/v1",
        "patch_count": len(stats),
        "edge_pair_count": len(edges),
        "isolated_patch_count": len(isolated),
        "isolated_patch_ratio": len(isolated) / max(len(stats), 1),
        "total_voxels": int(total_voxels),
        "isolated_voxels": int(sum(isolated_voxel_counts)),
        "isolated_voxel_ratio": sum(isolated_voxel_counts) / max(total_voxels, 1),
        "degree_p50": percentile([degree[patch_id] for patch_id in patch_ids], 0.50),
        "degree_p90": percentile([degree[patch_id] for patch_id in patch_ids], 0.90),
        "degree_p99": percentile([degree[patch_id] for patch_id in patch_ids], 0.99),
        "patch_voxels_p50": percentile(voxel_counts, 0.50),
        "patch_voxels_p90": percentile(voxel_counts, 0.90),
        "patch_voxels_p99": percentile(voxel_counts, 0.99),
        "isolated_voxels_p50": percentile(isolated_voxel_counts, 0.50),
        "isolated_voxels_p90": percentile(isolated_voxel_counts, 0.90),
        "isolated_voxels_p99": percentile(isolated_voxel_counts, 0.99),
        "edge_shared_p50": percentile(edge_shared, 0.50),
        "edge_shared_p90": percentile(edge_shared, 0.90),
        "edge_shared_p99": percentile(edge_shared, 0.99),
        "patch_size_bins": size_bins(voxel_counts),
        "isolated_size_bins": size_bins(isolated_voxel_counts),
        "isolated_geometry_counts": dict(Counter(stats[patch_id].geometry_type for patch_id in isolated)),
        "large_isolated_top20": sorted(
            [
                {
                    "patch_id": int(patch_id),
                    "voxel_count": int(stats[patch_id].count),
                    "geometry_type": stats[patch_id].geometry_type,
                    "bucket_entropy": entropy(stats[patch_id].bucket_counts),
                }
                for patch_id in isolated
                if stats[patch_id].count >= large_isolated_min_voxels
            ],
            key=lambda row: row["voxel_count"],
            reverse=True,
        )[:20],
    }
    report["missing_neighbor_diagnostics"] = diagnose_missing_neighbors(
        arrays,
        labels,
        stats,
        isolated,
        top_n=neighbor_top_n,
        cell_size=neighbor_cell_size,
        radius=neighbor_radius,
        max_cells_per_patch=neighbor_max_cells_per_patch,
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-input", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--large-isolated-min-voxels", type=int, default=1000)
    parser.add_argument("--neighbor-top-n", type=int, default=0)
    parser.add_argument("--neighbor-cell-size", type=float, default=0.05)
    parser.add_argument("--neighbor-radius", type=int, default=1)
    parser.add_argument("--neighbor-max-cells-per-patch", type=int, default=20000)
    args = parser.parse_args()

    arrays, src, dst = read_region_input(args.region_input)
    labels = read_labels(args.labels)
    report = summarize(
        arrays,
        labels,
        src,
        dst,
        args.large_isolated_min_voxels,
        neighbor_top_n=args.neighbor_top_n,
        neighbor_cell_size=args.neighbor_cell_size,
        neighbor_radius=args.neighbor_radius,
        neighbor_max_cells_per_patch=args.neighbor_max_cells_per_patch,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
