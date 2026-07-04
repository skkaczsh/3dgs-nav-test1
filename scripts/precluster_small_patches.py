#!/usr/bin/env python3
"""Pre-cluster adjacent small geo patches before graph-energy optimization.

This stage is intentionally narrow: it only rewrites patch labels.  It does not
create geometry, change voxel rows, or assign semantics.  The goal is to turn
tiny mutually-compatible fragments into medium anchors so the downstream energy
optimizer is not dominated by "anchor too small" rejects.
"""

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

from scripts.optimize_patch_graph_energy import (  # noqa: E402
    BUCKET_NAMES,
    build_edge_features,
    compute_patch_stats,
    read_labels,
    read_region_input,
    write_labels,
)


class UnionFind:
    def __init__(self, ids: list[int], counts: dict[int, int], buckets: dict[int, str]) -> None:
        self.parent = {int(pid): int(pid) for pid in ids}
        self.counts = {int(pid): int(counts[pid]) for pid in ids}
        self.buckets = {int(pid): buckets[pid] for pid in ids}
        self.sources = {int(pid): 1 for pid in ids}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, a: int, b: int) -> int:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return ra
        if self.counts[ra] < self.counts[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.counts[ra] += self.counts[rb]
        self.sources[ra] += self.sources[rb]
        if self.buckets[ra] == "unknown" and self.buckets[rb] != "unknown":
            self.buckets[ra] = self.buckets[rb]
        return ra


def bucket_compatible(a: str, b: str, *, allow_rough_bridge: bool) -> bool:
    if a == b:
        return True
    if "unknown" in {a, b}:
        return True
    if allow_rough_bridge and {a, b} <= {"rough_mixed", "thin_linear", "vertical", "horizontal"}:
        return True
    return False


def candidate_score(
    *,
    shared_edges: int,
    min_count: int,
    color_distance: float,
    normal_score: float,
    args: argparse.Namespace,
) -> float:
    contact = min(1.0, float(shared_edges) / max(float(min_count), 1.0))
    color = max(0.0, 1.0 - color_distance / max(args.max_contact_color_distance, 1e-6))
    normal = max(0.0, min(1.0, normal_score))
    return (
        args.contact_weight * contact
        + args.color_weight * color
        + args.normal_weight * normal
    )


def build_plan(
    arrays: dict[str, np.ndarray],
    labels: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    stats = compute_patch_stats(arrays, labels)
    ids = sorted(stats)
    counts = {pid: stats[pid].count for pid in ids}
    buckets = {pid: stats[pid].geometry_type for pid in ids}
    uf = UnionFind(ids, counts, buckets)

    edge_features = build_edge_features(labels, arrays["src"], arrays["dst"], arrays)
    candidates: list[tuple[float, int, int, dict[str, Any]]] = []
    reason_counts: Counter[str] = Counter()

    for (a, b), feat in edge_features.items():
        if a not in stats or b not in stats:
            continue
        ca = stats[a].count
        cb = stats[b].count
        if min(ca, cb) > args.max_small_voxels:
            reason_counts["both_too_large"] += 1
            continue
        if max(ca, cb) > args.max_partner_voxels:
            reason_counts["partner_too_large"] += 1
            continue
        if int(feat["shared_edges"]) < args.min_shared_edges:
            reason_counts["shared_edges"] += 1
            continue
        if float(feat["contact_color_distance"]) > args.max_contact_color_distance:
            reason_counts["contact_color_distance"] += 1
            continue
        if float(feat["contact_normal_score"]) < args.min_contact_normal_score:
            reason_counts["contact_normal_score"] += 1
            continue
        if not bucket_compatible(stats[a].geometry_type, stats[b].geometry_type, allow_rough_bridge=args.allow_rough_bridge):
            reason_counts["bucket"] += 1
            continue
        score = candidate_score(
            shared_edges=int(feat["shared_edges"]),
            min_count=min(ca, cb),
            color_distance=float(feat["contact_color_distance"]),
            normal_score=float(feat["contact_normal_score"]),
            args=args,
        )
        if score < args.min_score:
            reason_counts["score"] += 1
            continue
        candidates.append((score, a, b, feat))

    candidates.sort(reverse=True, key=lambda row: row[0])
    accepted: list[dict[str, Any]] = []
    for score, a, b, feat in candidates:
        ra = uf.find(a)
        rb = uf.find(b)
        if ra == rb:
            reason_counts["same_cluster"] += 1
            continue
        merged_count = uf.counts[ra] + uf.counts[rb]
        if merged_count > args.max_cluster_voxels:
            reason_counts["cluster_too_large"] += 1
            continue
        if uf.sources[ra] + uf.sources[rb] > args.max_source_patches:
            reason_counts["too_many_sources"] += 1
            continue
        if not bucket_compatible(uf.buckets[ra], uf.buckets[rb], allow_rough_bridge=args.allow_rough_bridge):
            reason_counts["cluster_bucket"] += 1
            continue
        root = uf.union(ra, rb)
        accepted.append(
            {
                "a": int(a),
                "b": int(b),
                "root": int(root),
                "score": float(score),
                "shared_edges": int(feat["shared_edges"]),
                "contact_color_distance": float(feat["contact_color_distance"]),
                "contact_normal_score": float(feat["contact_normal_score"]),
                "merged_count": int(uf.counts[root]),
                "source_patches": int(uf.sources[root]),
            }
        )

    remap = {pid: uf.find(pid) for pid in ids}
    output_labels = labels.copy()
    for old, new in remap.items():
        if old != new:
            output_labels[labels == old] = new

    output_stats = compute_patch_stats(arrays, output_labels)
    params = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    report = {
        "schema": "small-patch-precluster/v1",
        "input_patch_count": len(stats),
        "output_patch_count": len(output_stats),
        "input_point_count": int(len(labels)),
        "accepted_merge_count": len(accepted),
        "candidate_count": len(candidates),
        "reject_reason_counts": dict(reason_counts),
        "input_small_patch_count": sum(1 for row in stats.values() if row.count <= args.max_small_voxels),
        "output_small_patch_count": sum(1 for row in output_stats.values() if row.count <= args.max_small_voxels),
        "params": params,
    }
    return output_labels, report, accepted[: args.max_log_rows]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-input", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-labels", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--merge-log", type=Path)
    parser.add_argument("--max-small-voxels", type=int, default=64)
    parser.add_argument("--max-partner-voxels", type=int, default=1600)
    parser.add_argument("--max-cluster-voxels", type=int, default=2400)
    parser.add_argument("--max-source-patches", type=int, default=32)
    parser.add_argument("--min-shared-edges", type=int, default=2)
    parser.add_argument("--max-contact-color-distance", type=float, default=82.0)
    parser.add_argument("--min-contact-normal-score", type=float, default=0.20)
    parser.add_argument("--min-score", type=float, default=0.58)
    parser.add_argument("--contact-weight", type=float, default=0.42)
    parser.add_argument("--color-weight", type=float, default=0.38)
    parser.add_argument("--normal-weight", type=float, default=0.20)
    parser.add_argument("--allow-rough-bridge", action="store_true")
    parser.add_argument("--max-log-rows", type=int, default=60000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    arrays, src, dst = read_region_input(args.region_input)
    arrays["src"] = src
    arrays["dst"] = dst
    labels = read_labels(args.labels)
    if len(labels) != len(arrays["xyz"]):
        raise ValueError(f"labels length {len(labels)} != region point count {len(arrays['xyz'])}")
    output_labels, report, accepted = build_plan(arrays, labels, args)
    write_labels(args.output_labels, output_labels)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.merge_log:
        args.merge_log.parent.mkdir(parents=True, exist_ok=True)
        with args.merge_log.open("w", encoding="utf-8") as fh:
            for row in accepted:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
