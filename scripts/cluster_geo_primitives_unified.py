#!/usr/bin/env python3
"""Cluster over-segmented geo patches with one unified primitive graph.

This is a test route for replacing case-specific post-process patches with a
single graph aggregation stage.  Region growing still produces conservative
GeoPrimitives.  This script builds candidate edges from spatial adjacency and
AABB containment/overlap, scores every edge with the same feature set, and
merges edges whose score passes a threshold.
"""

from __future__ import annotations

import argparse
import heapq
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from optimize_geo_patch_merges import (
    BUCKET_NAMES,
    PatchStats,
    compatible_bucket_score,
    compute_patch_stats,
    dominant_geometry,
    normal_score,
    read_labels,
    read_region_input,
    write_ply,
)


ROUGH_TYPES = {"rough_mixed", "thin_linear", "unknown", "mixed"}
STABLE_TYPES = {"horizontal", "vertical"}


@dataclass(frozen=True)
class Edge:
    a: int
    b: int
    source: str
    score: float
    reason: str
    features: dict[str, float | str]


class DSU:
    def __init__(self, values: list[int]) -> None:
        self.parent = {int(v): int(v) for v in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True


def patch_edges(labels: np.ndarray, src: np.ndarray, dst: np.ndarray) -> Counter[tuple[int, int]]:
    a = labels[src]
    b = labels[dst]
    mask = a != b
    out: Counter[tuple[int, int]] = Counter()
    for pa, pb in zip(a[mask].tolist(), b[mask].tolist(), strict=True):
        pa = int(pa)
        pb = int(pb)
        if pa > pb:
            pa, pb = pb, pa
        out[(pa, pb)] += 1
    return out


def bbox_volume(stats: PatchStats) -> float:
    return float(np.prod(np.maximum(stats.bbox_max - stats.bbox_min, 1e-3)))


def bbox_features(a: PatchStats, b: PatchStats) -> dict[str, float]:
    dims = np.maximum(0.0, np.minimum(a.bbox_max, b.bbox_max) - np.maximum(a.bbox_min, b.bbox_min))
    overlap = float(np.prod(dims))
    va = bbox_volume(a)
    vb = bbox_volume(b)
    union = va + vb - overlap
    return {
        "bbox_overlap_volume": overlap,
        "bbox_ratio_min": overlap / max(min(va, vb), 1e-9),
        "bbox_ratio_max": overlap / max(max(va, vb), 1e-9),
        "bbox_iou": overlap / max(union, 1e-9),
        "centroid_distance": float(np.linalg.norm(a.centroid - b.centroid)),
        "aspect_max": max(aspect_ratio(a), aspect_ratio(b)),
    }


def aspect_ratio(stats: PatchStats) -> float:
    extent = np.maximum(stats.bbox_max - stats.bbox_min, 1e-3)
    return float(np.max(extent) / max(float(np.min(extent)), 1e-3))


def base_features(a: PatchStats, b: PatchStats, shared_edges: int) -> dict[str, float | str]:
    color_dist = float(np.linalg.norm(a.mean_rgb - b.mean_rgb))
    color = max(0.0, min(1.0, 1.0 - color_dist / 130.0))
    bucket = compatible_bucket_score(a.geometry_type, b.geometry_type)
    normal = normal_score(a.mean_normal, b.mean_normal)
    contact = min(1.0, shared_edges / max(float(min(a.count, b.count)), 1.0))
    balance = min(a.count, b.count) / max(float(max(a.count, b.count)), 1.0)
    bbox = bbox_features(a, b)
    return {
        "color": color,
        "color_dist": color_dist,
        "bucket": bucket,
        "normal": normal,
        "contact": contact,
        "balance": balance,
        "geom_a": a.geometry_type,
        "geom_b": b.geometry_type,
        **bbox,
    }


def score_edge(a: PatchStats, b: PatchStats, shared_edges: int, source: str, args: argparse.Namespace) -> tuple[float, str, dict[str, float | str]]:
    f = base_features(a, b, shared_edges)
    rough_pair = a.geometry_type in ROUGH_TYPES and b.geometry_type in ROUGH_TYPES
    stable_mismatch = a.geometry_type in STABLE_TYPES and b.geometry_type in STABLE_TYPES and a.geometry_type != b.geometry_type
    stable_to_rough = (a.geometry_type in STABLE_TYPES and b.geometry_type in ROUGH_TYPES) or (
        b.geometry_type in STABLE_TYPES and a.geometry_type in ROUGH_TYPES
    )

    contact_score = 0.34 * float(f["contact"]) + 0.24 * float(f["color"]) + 0.18 * float(f["bucket"]) + 0.14 * float(f["normal"]) + 0.10 * float(f["balance"])
    containment_score = 0.30 * float(f["bbox_ratio_min"]) + 0.24 * float(f["color"]) + 0.18 * float(f["bucket"]) + 0.14 * float(f["contact"]) + 0.14 * float(f["balance"])
    porous_score = 0.38 * float(f["bbox_ratio_min"]) + 0.24 * float(f["color"]) + 0.18 * float(f["bucket"]) + 0.12 * float(f["contact"]) + 0.08 * float(f["balance"])
    interleaved_score = 0.28 * float(f["bbox_ratio_min"]) + 0.22 * float(f["bbox_iou"]) + 0.20 * float(f["normal"]) + 0.16 * float(f["color"]) + 0.14 * float(f["balance"])

    scores = {
        "contact_score": contact_score,
        "containment_score": containment_score,
        "porous_score": porous_score if rough_pair else -1.0,
        "interleaved_score": interleaved_score,
    }
    f.update(scores)

    reason = "contact"
    score = contact_score
    if containment_score > score:
        score = containment_score
        reason = "containment"
    if rough_pair and porous_score > score:
        score = porous_score
        reason = "porous"
    if interleaved_score > score:
        score = interleaved_score
        reason = "interleaved"

    penalty = 0.0
    if stable_mismatch:
        penalty += args.stable_mismatch_penalty
    if stable_to_rough and float(f["bbox_ratio_min"]) < args.stable_rough_min_bbox_ratio:
        penalty += args.stable_rough_penalty
    if float(f["color_dist"]) > args.hard_color_distance and not rough_pair:
        penalty += args.hard_color_penalty
    score -= penalty
    f["penalty"] = penalty
    f["score_after_penalty"] = score
    return score, reason, f


def build_bbox_candidate_pairs(stats: dict[int, PatchStats], args: argparse.Namespace) -> set[tuple[int, int]]:
    selected = [s for s in stats.values() if s.count >= args.bbox_min_patch_voxels]
    selected.sort(key=lambda s: s.count, reverse=True)
    if args.bbox_top_n > 0:
        selected = selected[: args.bbox_top_n]
    pairs: set[tuple[int, int]] = set()
    for i, a in enumerate(selected):
        for b in selected[i + 1 :]:
            f = bbox_features(a, b)
            if f["bbox_overlap_volume"] <= 0:
                continue
            long_pair = (
                min(a.count, b.count) >= args.long_patch_min_voxels
                and f["bbox_ratio_min"] >= args.long_patch_min_bbox_ratio
                and f["bbox_iou"] >= args.long_patch_min_bbox_iou
                and f["aspect_max"] >= args.long_patch_aspect_ratio
            )
            if not long_pair and f["bbox_ratio_min"] < args.bbox_candidate_min_ratio and f["bbox_iou"] < args.bbox_candidate_min_iou:
                continue
            if not long_pair and f["centroid_distance"] > args.bbox_candidate_max_centroid and f["bbox_ratio_min"] < args.long_pair_min_ratio:
                continue
            pa, pb = sorted((a.patch_id, b.patch_id))
            pairs.add((pa, pb))
    return pairs


def build_edges(
    stats: dict[int, PatchStats],
    labels: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    args: argparse.Namespace,
) -> list[Edge]:
    adjacency = patch_edges(labels, src, dst)
    candidate_pairs = set(adjacency)
    candidate_pairs.update(build_bbox_candidate_pairs(stats, args))
    edges: list[Edge] = []
    for pa, pb in candidate_pairs:
        if pa not in stats or pb not in stats:
            continue
        shared = adjacency.get((pa, pb), 0)
        source = "adjacency" if shared else "bbox"
        score, reason, features = score_edge(stats[pa], stats[pb], shared, source, args)
        threshold = args.min_edge_score
        if reason == "porous":
            threshold = args.min_porous_score
        elif reason == "interleaved":
            threshold = args.min_interleaved_score
        elif source == "bbox":
            threshold = max(threshold, args.min_bbox_edge_score)
        if score < threshold:
            continue
        edges.append(Edge(pa, pb, source, score, reason, features))
    edges.sort(key=lambda e: e.score, reverse=True)
    return edges[: args.max_edges]


def merge_patch_stats(keep_id: int, a: PatchStats, b: PatchStats) -> PatchStats:
    total = a.count + b.count
    centroid = (a.centroid * a.count + b.centroid * b.count) / total
    mean_rgb = (a.mean_rgb * a.count + b.mean_rgb * b.count) / total
    normal = a.mean_normal * a.count + b.mean_normal * b.count
    norm = float(np.linalg.norm(normal))
    mean_normal = normal / norm if norm > 1e-9 else normal
    bucket_counts = Counter(a.bucket_counts)
    bucket_counts.update(b.bucket_counts)
    source_patch_ids = set(a.source_patch_ids)
    source_patch_ids.update(b.source_patch_ids)
    return PatchStats(
        patch_id=keep_id,
        count=total,
        centroid=centroid,
        mean_rgb=mean_rgb,
        mean_normal=mean_normal,
        bbox_min=np.minimum(a.bbox_min, b.bbox_min),
        bbox_max=np.maximum(a.bbox_max, b.bbox_max),
        bucket_counts=bucket_counts,
        geometry_type=dominant_geometry(bucket_counts),
        source_patch_ids=source_patch_ids,
    )


def build_component_stats(stats: dict[int, PatchStats], dsu: DSU) -> dict[int, PatchStats]:
    component_stats: dict[int, PatchStats] = {}
    for patch_id, patch_stats in stats.items():
        root = dsu.find(patch_id)
        if root in component_stats:
            component_stats[root] = merge_patch_stats(root, component_stats[root], patch_stats)
        else:
            component_stats[root] = PatchStats(
                patch_id=root,
                count=patch_stats.count,
                centroid=patch_stats.centroid.copy(),
                mean_rgb=patch_stats.mean_rgb.copy(),
                mean_normal=patch_stats.mean_normal.copy(),
                bbox_min=patch_stats.bbox_min.copy(),
                bbox_max=patch_stats.bbox_max.copy(),
                bucket_counts=Counter(patch_stats.bucket_counts),
                geometry_type=patch_stats.geometry_type,
                source_patch_ids=set(patch_stats.source_patch_ids),
            )
    return component_stats


def write_component_jsonl(path: Path, component_stats: dict[int, PatchStats], args: argparse.Namespace) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for patch_id in sorted(component_stats):
            s = component_stats[patch_id]
            source_patch_ids = sorted(int(v) for v in s.source_patch_ids)
            row = {
                "patch_id": int(patch_id),
                "object": int(patch_id),
                "voxel_count": int(s.count),
                "status": "small_patch" if s.count < args.small_patch_voxels else "geo_patch",
                "geometry_type": s.geometry_type,
                "semantic_label": s.geometry_type,
                "description": f"unified geometry primitive: {s.geometry_type}",
                "bucket_counts": {BUCKET_NAMES[int(k)]: int(v) for k, v in s.bucket_counts.items()},
                "centroid": s.centroid.astype(float).tolist(),
                "bbox_3d": {"min": s.bbox_min.astype(float).tolist(), "max": s.bbox_max.astype(float).tolist()},
                "extent": (s.bbox_max - s.bbox_min).astype(float).tolist(),
                "mean_rgb": s.mean_rgb.astype(float).tolist(),
                "mean_normal": s.mean_normal.astype(float).tolist(),
                "source_patch_count": len(source_patch_ids),
                "source_patch_ids": source_patch_ids,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(component_stats)


def cluster(labels: np.ndarray, stats: dict[int, PatchStats], edges: list[Edge], args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]], dict[int, PatchStats]]:
    dsu = DSU(list(stats))
    merge_log: list[dict[str, Any]] = []
    component_size = {pid: stats[pid].count for pid in stats}
    for edge in edges:
        ra = dsu.find(edge.a)
        rb = dsu.find(edge.b)
        if ra == rb:
            continue
        if component_size.get(ra, stats[edge.a].count) + component_size.get(rb, stats[edge.b].count) > args.max_component_voxels:
            continue
        keep, drop = (ra, rb) if component_size.get(ra, 0) >= component_size.get(rb, 0) else (rb, ra)
        if not dsu.union(keep, drop):
            continue
        component_size[keep] = component_size.get(keep, 0) + component_size.get(drop, 0)
        component_size.pop(drop, None)
        if len(merge_log) < args.max_log_rows:
            merge_log.append(
                {
                    "keep": int(keep),
                    "drop": int(drop),
                    "edge_a": int(edge.a),
                    "edge_b": int(edge.b),
                    "source": edge.source,
                    "reason": edge.reason,
                    "score": float(edge.score),
                    **edge.features,
                }
            )

    max_label = int(labels.max())
    table = np.arange(max_label + 1, dtype=np.int32)
    for patch_id in stats:
        root = dsu.find(patch_id)
        if patch_id <= max_label:
            table[patch_id] = root
    out = table[labels]
    component_stats = build_component_stats(stats, dsu)
    report = {
        "schema": "geo-primitive-unified-cluster/v1",
        "input_patch_count": int(len(stats)),
        "edge_count": int(len(edges)),
        "accepted_merge_count": int(len(merge_log)),
        "output_patch_count": int(len(component_stats)),
        "edge_reason_counts": dict(Counter(edge.reason for edge in edges)),
        "edge_source_counts": dict(Counter(edge.source for edge in edges)),
        "params": vars(args),
    }
    return out, report, merge_log, component_stats


def cluster_dynamic(labels: np.ndarray, stats: dict[int, PatchStats], edges: list[Edge], args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]], dict[int, PatchStats]]:
    dsu = DSU(list(stats))
    component_stats = dict(stats)
    component_size = {pid: stats[pid].count for pid in stats}
    neighbors: dict[int, set[int]] = {pid: set() for pid in stats}
    shared_edges: Counter[tuple[int, int]] = Counter()
    source_rank: dict[tuple[int, int], str] = {}

    for edge in edges:
        pa, pb = sorted((edge.a, edge.b))
        neighbors[pa].add(pb)
        neighbors[pb].add(pa)
        approx_shared = int(round(float(edge.features.get("contact", 0.0)) * min(stats[pa].count, stats[pb].count)))
        shared_edges[(pa, pb)] += max(approx_shared, 0)
        if edge.source == "adjacency" or (pa, pb) not in source_rank:
            source_rank[(pa, pb)] = edge.source

    heap: list[tuple[float, int, int, int, str, dict[str, float | str]]] = []
    version = {pid: 0 for pid in stats}
    pushes = 0

    def push_edge(a: int, b: int) -> None:
        nonlocal pushes
        ra = dsu.find(a)
        rb = dsu.find(b)
        if ra == rb or ra not in component_stats or rb not in component_stats:
            return
        pa, pb = sorted((ra, rb))
        shared = shared_edges.get((pa, pb), 0)
        source = source_rank.get((pa, pb), "bbox")
        score, reason, features = score_edge(component_stats[pa], component_stats[pb], shared, source, args)
        threshold = args.min_edge_score
        if reason == "porous":
            threshold = args.min_porous_score
        elif reason == "interleaved":
            threshold = args.min_interleaved_score
        elif source == "bbox":
            threshold = max(threshold, args.min_bbox_edge_score)
        if score < threshold:
            return
        pushes += 1
        heapq.heappush(heap, (-score, pa, pb, version[pa] + version[pb], reason, features))

    for pa, pb in shared_edges:
        push_edge(pa, pb)

    merge_log: list[dict[str, Any]] = []
    stale_pops = 0
    rejected_size = 0
    while heap and len(merge_log) < args.max_dynamic_merges:
        neg_score, pa, pb, edge_version, reason, features = heapq.heappop(heap)
        ra = dsu.find(pa)
        rb = dsu.find(pb)
        if ra == rb or ra not in component_stats or rb not in component_stats:
            stale_pops += 1
            continue
        ca, cb = sorted((ra, rb))
        if edge_version != version.get(ca, 0) + version.get(cb, 0):
            stale_pops += 1
            push_edge(ca, cb)
            continue
        if component_size.get(ca, component_stats[ca].count) + component_size.get(cb, component_stats[cb].count) > args.max_component_voxels:
            rejected_size += 1
            continue

        keep, drop = (ca, cb) if component_size.get(ca, 0) >= component_size.get(cb, 0) else (cb, ca)
        score = -neg_score
        if not dsu.union(keep, drop):
            continue

        keep_stats = component_stats[keep]
        drop_stats = component_stats[drop]
        component_stats[keep] = merge_patch_stats(keep, keep_stats, drop_stats)
        component_stats.pop(drop, None)
        component_size[keep] = component_size.get(keep, 0) + component_size.get(drop, 0)
        component_size.pop(drop, None)
        version[keep] = version.get(keep, 0) + 1
        version.pop(drop, None)

        merged_neighbors = (neighbors.get(keep, set()) | neighbors.get(drop, set())) - {keep, drop}
        neighbors[keep] = set()
        neighbors.pop(drop, None)
        for nbr in merged_neighbors:
            rn = dsu.find(nbr)
            if rn == keep or rn not in component_stats:
                continue
            old_a = tuple(sorted((keep, rn)))
            old_b = tuple(sorted((drop, rn)))
            new_key = tuple(sorted((keep, rn)))
            shared_edges[new_key] = shared_edges.get(old_a, 0) + shared_edges.get(old_b, 0)
            if source_rank.get(old_a) == "adjacency" or source_rank.get(old_b) == "adjacency":
                source_rank[new_key] = "adjacency"
            else:
                source_rank[new_key] = source_rank.get(old_a) or source_rank.get(old_b) or "bbox"
            neighbors[keep].add(rn)
            neighbors.setdefault(rn, set()).discard(drop)
            neighbors[rn].add(keep)
            push_edge(keep, rn)

        if len(merge_log) < args.max_log_rows:
            merge_log.append(
                {
                    "keep": int(keep),
                    "drop": int(drop),
                    "edge_a": int(ca),
                    "edge_b": int(cb),
                    "reason": reason,
                    "score": float(score),
                    **features,
                }
            )

    max_label = int(labels.max())
    table = np.arange(max_label + 1, dtype=np.int32)
    for patch_id in stats:
        root = dsu.find(patch_id)
        if patch_id <= max_label:
            table[patch_id] = root
    out = table[labels]
    report = {
        "schema": "geo-primitive-unified-dynamic-cluster/v1",
        "input_patch_count": int(len(stats)),
        "initial_edge_count": int(len(edges)),
        "accepted_merge_count": int(len(merge_log)),
        "output_patch_count": int(len(component_stats)),
        "stale_heap_pops": stale_pops,
        "rejected_size_count": rejected_size,
        "heap_push_count": pushes,
        "merge_reason_counts": dict(Counter(row["reason"] for row in merge_log)),
        "params": vars(args),
    }
    return out, report, merge_log, component_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-input", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bbox-top-n", type=int, default=3000)
    parser.add_argument("--bbox-min-patch-voxels", type=int, default=64)
    parser.add_argument("--bbox-candidate-min-ratio", type=float, default=0.72)
    parser.add_argument("--bbox-candidate-min-iou", type=float, default=0.10)
    parser.add_argument("--bbox-candidate-max-centroid", type=float, default=18.0)
    parser.add_argument("--long-pair-min-ratio", type=float, default=0.85)
    parser.add_argument("--long-patch-min-voxels", type=int, default=100000)
    parser.add_argument("--long-patch-aspect-ratio", type=float, default=2.5)
    parser.add_argument("--long-patch-min-bbox-ratio", type=float, default=0.80)
    parser.add_argument("--long-patch-min-bbox-iou", type=float, default=0.12)
    parser.add_argument("--min-edge-score", type=float, default=0.70)
    parser.add_argument("--min-bbox-edge-score", type=float, default=0.76)
    parser.add_argument("--min-porous-score", type=float, default=0.74)
    parser.add_argument("--min-interleaved-score", type=float, default=0.80)
    parser.add_argument("--stable-mismatch-penalty", type=float, default=0.24)
    parser.add_argument("--stable-rough-penalty", type=float, default=0.18)
    parser.add_argument("--stable-rough-min-bbox-ratio", type=float, default=0.80)
    parser.add_argument("--hard-color-distance", type=float, default=150.0)
    parser.add_argument("--hard-color-penalty", type=float, default=0.18)
    parser.add_argument("--max-component-voxels", type=int, default=900000)
    parser.add_argument("--max-edges", type=int, default=120000)
    parser.add_argument("--max-log-rows", type=int, default=50000)
    parser.add_argument("--small-patch-voxels", type=int, default=8)
    parser.add_argument("--preview-stride", type=int, default=5)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--max-dynamic-merges", type=int, default=5000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    arrays, src, dst = read_region_input(args.region_input)
    labels = read_labels(args.labels)
    if len(labels) != len(arrays["xyz"]):
        raise ValueError(f"label count mismatch: labels={len(labels)} voxels={len(arrays['xyz'])}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = compute_patch_stats(arrays, labels)
    edges = build_edges(stats, labels, src, dst, args)
    if args.dynamic:
        out, report, merge_log, component_stats = cluster_dynamic(labels, stats, edges, args)
    else:
        out, report, merge_log, component_stats = cluster(labels, stats, edges, args)
    report["output_ply"] = str(args.output_dir / f"geo_primitives_unified_stride{args.preview_stride}.ply")
    report["output_jsonl"] = str(args.output_dir / "geo_primitives_unified.jsonl")
    report["preview_points"] = write_ply(Path(report["output_ply"]), arrays, out, args.preview_stride)
    report["jsonl_patch_count"] = write_component_jsonl(Path(report["output_jsonl"]), component_stats, args)
    (args.output_dir / "primitive_graph_edges.jsonl").write_text(
        "".join(json.dumps({"a": e.a, "b": e.b, "source": e.source, "reason": e.reason, "score": e.score, **e.features}, ensure_ascii=False) + "\n" for e in edges[: args.max_log_rows]),
        encoding="utf-8",
    )
    (args.output_dir / "unified_cluster_merge_log.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merge_log),
        encoding="utf-8",
    )
    (args.output_dir / "unified_cluster_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
