#!/usr/bin/env python3
"""Report candidate-edge sparsity for Superpoint Graph clustering."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.optimize_patch_graph_energy import build_edge_counts, compute_patch_stats, entropy, read_labels, read_region_input


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


def summarize(arrays, labels, src, dst, large_isolated_min_voxels: int) -> dict:
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

    return {
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-input", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--large-isolated-min-voxels", type=int, default=1000)
    args = parser.parse_args()

    arrays, src, dst = read_region_input(args.region_input)
    labels = read_labels(args.labels)
    report = summarize(arrays, labels, src, dst, args.large_isolated_min_voxels)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
