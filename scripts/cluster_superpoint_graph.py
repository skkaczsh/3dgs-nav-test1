#!/usr/bin/env python3
"""Cluster existing GeoPatches as a Superpoint Graph.

This is intentionally smaller than optimize_patch_graph_energy.py: no split,
no boundary transfer, no post-pass attachment.  It treats the input patches as
superpoints, scores adjacent edges once, and unions compatible components.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.optimize_patch_graph_energy import (
    build_edge_counts,
    build_edge_features,
    compute_patch_stats,
    entropy,
    merge_patch_stats,
    read_labels,
    read_region_input,
    structural_merge_veto,
    write_jsonl,
    write_labels,
    write_ply,
)


class DSU:
    def __init__(self, ids: list[int]) -> None:
        self.parent = {int(i): int(i) for i in ids}

    def find(self, x: int) -> int:
        p = self.parent[int(x)]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, keep: int, drop: int) -> None:
        self.parent[self.find(drop)] = self.find(keep)


def edge_score(feature: dict[str, float], max_color_distance: float) -> float:
    color = 1.0 - min(1.0, max(0.0, feature.get("contact_color_distance", max_color_distance)) / max(max_color_distance, 1e-6))
    color_p90 = 1.0 - min(1.0, max(0.0, feature.get("contact_color_p90", max_color_distance)) / max(max_color_distance, 1e-6))
    support = max(0.0, min(1.0, feature.get("contact_support", 0.0)))
    normal = max(0.0, min(1.0, feature.get("contact_normal_score", 0.0)))
    rough = 1.0 - min(1.0, max(0.0, feature.get("contact_roughness_delta", 1.0)) / 0.35)
    planar = 1.0 - min(1.0, max(0.0, feature.get("contact_planarity_delta", 1.0)) / 0.35)
    linear = 1.0 - min(1.0, max(0.0, feature.get("contact_linearity_delta", 1.0)) / 0.35)
    return 0.28 * color + 0.16 * color_p90 + 0.18 * support + 0.14 * normal + 0.10 * rough + 0.07 * planar + 0.07 * linear


def remap_labels(labels: np.ndarray, dsu: DSU) -> np.ndarray:
    max_label = int(labels.max())
    remap = np.arange(max_label + 1, dtype=np.int32)
    for label in np.unique(labels).tolist():
        remap[int(label)] = dsu.find(int(label))
    return remap[labels]


def cluster(arrays: dict[str, np.ndarray], labels: np.ndarray, src: np.ndarray, dst: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    stats = compute_patch_stats(arrays, labels)
    dsu = DSU(sorted(stats))
    edge_counts = build_edge_counts(labels, src, dst)
    edge_features = build_edge_features(labels, src, dst, arrays)
    rows = []
    for pair, shared in edge_counts.items():
        a, b = pair
        feature = dict(edge_features.get(pair, {}))
        feature["contact_support"] = float(shared) / max(float(min(stats[a].count, stats[b].count)), 1.0)
        rows.append((edge_score(feature, args.max_color_distance), int(shared), pair, feature))
    rows.sort(reverse=True)

    accepted = 0
    rejects: dict[str, int] = {}
    for score, shared, (a0, b0), feature in rows:
        a = dsu.find(a0)
        b = dsu.find(b0)
        if a == b:
            continue
        if score < args.min_edge_score:
            rejects["score"] = rejects.get("score", 0) + 1
            continue
        sa = stats[a]
        sb = stats[b]
        if min(sa.count, sb.count) < args.min_patch_voxels:
            rejects["small_patch"] = rejects.get("small_patch", 0) + 1
            continue
        vetoed, reason, _ = structural_merge_veto(sa, sb, args)
        if vetoed:
            rejects[reason] = rejects.get(reason, 0) + 1
            continue
        merged = merge_patch_stats(sa, sb)
        if entropy(merged.bucket_counts) > args.max_merged_entropy:
            rejects["merged_entropy"] = rejects.get("merged_entropy", 0) + 1
            continue
        keep, drop = (a, b) if sa.count >= sb.count else (b, a)
        dsu.union(keep, drop)
        stats[keep] = merge_patch_stats(stats[keep], stats[drop])
        del stats[drop]
        accepted += 1

    out = remap_labels(labels, dsu)
    report = {
        "schema": "superpoint-graph-cluster/v1",
        "input_patch_count": int(len(set(labels.tolist()))),
        "output_patch_count": int(len(set(out.tolist()))),
        "edge_count": int(len(rows)),
        "accepted_edges": int(accepted),
        "reject_counts": rejects,
        "params": vars(args),
    }
    return out, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-input", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", default="superpoint_graph")
    parser.add_argument("--min-edge-score", type=float, default=0.82)
    parser.add_argument("--max-color-distance", type=float, default=90.0)
    parser.add_argument("--max-merged-entropy", type=float, default=1.05)
    parser.add_argument("--min-patch-voxels", type=int, default=4)
    parser.add_argument("--enable-structural-merge-veto", action="store_true")
    parser.add_argument("--structural-veto-min-bucket-ratio", type=float, default=0.20)
    parser.add_argument("--structural-veto-min-voxels", type=int, default=1000)
    parser.add_argument("--preview-stride", type=int, default=10)
    parser.add_argument("--max-source-patch-ids", type=int, default=24)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    arrays, src, dst = read_region_input(args.region_input)
    labels = read_labels(args.labels)
    out, report = cluster(arrays, labels, src, dst, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    write_labels(args.output_dir / f"{stem}_labels.bin", out)
    report["preview_points"] = write_ply(args.output_dir / f"{stem}_stride{args.preview_stride}.ply", arrays, out, args.preview_stride)
    report["jsonl_patch_count"] = write_jsonl(args.output_dir / f"{stem}.jsonl", compute_patch_stats(arrays, out), args)
    (args.output_dir / f"{stem}_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
