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
    bbox_gap,
    build_edge_counts,
    build_edge_features,
    compute_patch_stats,
    entropy,
    fh_threshold,
    merge_patch_stats,
    normal_score,
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


def edge_dissimilarity(feature: dict[str, float], max_color_distance: float) -> float:
    return 1.0 - edge_score(feature, max_color_distance)


def contact_bridge(a, b, feature: dict[str, float], args: argparse.Namespace) -> bool:
    if getattr(args, "disable_contact_bridge", False):
        return False
    if a.geometry_type != b.geometry_type or a.geometry_type in {"unknown", "mixed"}:
        return False
    support = float(feature.get("contact_support", 0.0))
    color = float(feature.get("contact_color_distance", args.max_color_distance))
    color_p90 = float(feature.get("contact_color_p90", args.max_color_distance))
    return (
        support >= getattr(args, "contact_bridge_min_support", 0.25)
        and color <= getattr(args, "contact_bridge_max_color_distance", 65.0)
        and color_p90 <= getattr(args, "contact_bridge_max_color_p90", 80.0)
    )


def near_bbox_bridge(a, b, feature: dict[str, float], args: argparse.Namespace) -> bool:
    if getattr(args, "disable_near_bbox_candidates", False):
        return False
    if a.geometry_type != b.geometry_type or a.geometry_type in {"unknown", "mixed"}:
        return False
    return (
        float(feature.get("bbox_gap", 1e9)) <= getattr(args, "near_candidate_max_gap", 0.15)
        and float(feature.get("contact_color_distance", args.max_color_distance)) <= getattr(args, "near_candidate_max_color_distance", 70.0)
        and float(feature.get("contact_normal_score", 0.0)) >= getattr(args, "near_candidate_min_normal_score", 0.65)
    )


def build_near_bbox_candidates(stats: dict[int, object], existing_pairs: set[tuple[int, int]], args: argparse.Namespace) -> dict[tuple[int, int], dict[str, float]]:
    if getattr(args, "disable_near_bbox_candidates", False):
        return {}
    patches = [
        stat
        for stat in stats.values()
        if stat.count >= getattr(args, "near_candidate_min_voxels", 1000)
        and stat.geometry_type not in {"unknown", "mixed"}
    ]
    out: dict[tuple[int, int], dict[str, float]] = {}
    for i, a in enumerate(patches):
        kept = 0
        for b in patches[i + 1 :]:
            pair = (min(a.patch_id, b.patch_id), max(a.patch_id, b.patch_id))
            if pair in existing_pairs:
                continue
            gap = bbox_gap(a, b)
            if gap > args.near_candidate_max_gap:
                continue
            color = float(np.linalg.norm(a.mean_rgb - b.mean_rgb))
            normal = normal_score(a.mean_normal, b.mean_normal)
            feature = {
                "bbox_gap": gap,
                "contact_color_distance": color,
                "contact_color_p90": color,
                "contact_normal_score": normal,
                "contact_support": 0.0,
            }
            if near_bbox_bridge(a, b, feature, args):
                out[pair] = feature
                kept += 1
                if kept >= args.near_candidate_max_per_patch:
                    break
    return out


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
    near_candidates = build_near_bbox_candidates(stats, set(edge_counts), args)
    for pair, feature in near_candidates.items():
        rows.append((edge_score(feature, args.max_color_distance), 0, pair, feature))
    rows.sort(reverse=True)

    accepted = 0
    rejects: dict[str, int] = {}
    accepted_reasons: dict[str, int] = {}
    for score, shared, (a0, b0), feature in rows:
        a = dsu.find(a0)
        b = dsu.find(b0)
        if a == b:
            continue
        sa = stats[a]
        sb = stats[b]
        if min(sa.count, sb.count) < args.min_patch_voxels:
            rejects["small_patch"] = rejects.get("small_patch", 0) + 1
            continue
        accepted_reason = "score"
        if score < args.min_edge_score:
            if near_bbox_bridge(sa, sb, feature, args):
                accepted_reason = "near_bbox_bridge"
            elif not contact_bridge(sa, sb, feature, args):
                rejects["score"] = rejects.get("score", 0) + 1
                continue
            else:
                accepted_reason = "contact_bridge"
        dissimilarity = edge_dissimilarity(feature, args.max_color_distance)
        if args.fh_k > 0 and dissimilarity > min(fh_threshold(sa, args.fh_k), fh_threshold(sb, args.fh_k)):
            rejects["fh_threshold"] = rejects.get("fh_threshold", 0) + 1
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
        stats[keep].internal_diff = max(stats[keep].internal_diff, dissimilarity)
        del stats[drop]
        accepted += 1
        accepted_reasons[accepted_reason] = accepted_reasons.get(accepted_reason, 0) + 1

    out = remap_labels(labels, dsu)
    report = {
        "schema": "superpoint-graph-cluster/v1",
        "input_patch_count": int(len(set(labels.tolist()))),
        "output_patch_count": int(len(set(out.tolist()))),
        "edge_count": int(len(rows)),
        "touch_edge_count": int(len(edge_counts)),
        "near_bbox_candidate_count": int(len(near_candidates)),
        "accepted_edges": int(accepted),
        "accepted_reasons": accepted_reasons,
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
    parser.add_argument("--fh-k", type=float, default=0.0)
    parser.add_argument("--min-patch-voxels", type=int, default=4)
    parser.add_argument("--disable-contact-bridge", action="store_true")
    parser.add_argument("--contact-bridge-min-support", type=float, default=0.25)
    parser.add_argument("--contact-bridge-max-color-distance", type=float, default=65.0)
    parser.add_argument("--contact-bridge-max-color-p90", type=float, default=80.0)
    parser.add_argument("--disable-near-bbox-candidates", action="store_true")
    parser.add_argument("--near-candidate-min-voxels", type=int, default=1000)
    parser.add_argument("--near-candidate-max-gap", type=float, default=0.15)
    parser.add_argument("--near-candidate-max-color-distance", type=float, default=70.0)
    parser.add_argument("--near-candidate-min-normal-score", type=float, default=0.65)
    parser.add_argument("--near-candidate-max-per-patch", type=int, default=8)
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
