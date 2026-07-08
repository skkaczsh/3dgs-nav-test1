#!/usr/bin/env python3
"""Cluster existing GeoPatches as a Superpoint Graph.

This is intentionally smaller than optimize_patch_graph_energy.py: no split,
no boundary transfer, no post-pass attachment.  It treats the input patches as
superpoints, scores adjacent edges once, and unions compatible components.
"""

from __future__ import annotations

import argparse
import csv
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
    base = 0.28 * color + 0.16 * color_p90 + 0.18 * support + 0.14 * normal + 0.10 * rough + 0.07 * planar + 0.07 * linear
    external_weight = max(0.0, min(1.0, feature.get("external_edge_weight", 0.0)))
    if external_weight <= 0.0:
        return base
    external = max(0.0, min(1.0, feature.get("external_similarity", base)))
    return (1.0 - external_weight) * base + external_weight * external


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


def edge_pair(a: int, b: int) -> tuple[int, int]:
    return (min(int(a), int(b)), max(int(a), int(b)))


def row_pair(row: dict[str, object]) -> tuple[int, int]:
    a = row.get("patch_a", row.get("a", row.get("src")))
    b = row.get("patch_b", row.get("b", row.get("dst")))
    if a is None or b is None:
        raise ValueError(f"external edge row missing patch ids: {row}")
    return edge_pair(int(a), int(b))


def row_similarity(row: dict[str, object], max_distance: float) -> float:
    if "similarity" in row:
        return float(row["similarity"])
    if "external_similarity" in row:
        return float(row["external_similarity"])
    if "distance" in row:
        return 1.0 - min(1.0, max(0.0, float(row["distance"])) / max(max_distance, 1e-6))
    if "external_distance" in row:
        return 1.0 - min(1.0, max(0.0, float(row["external_distance"])) / max(max_distance, 1e-6))
    raise ValueError(f"external edge row missing similarity or distance: {row}")


def load_external_edge_evidence(path: Path | None, max_distance: float) -> dict[tuple[int, int], float]:
    if path is None:
        return {}
    rows: list[dict[str, object]] = []
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        with path.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    return {row_pair(row): max(0.0, min(1.0, row_similarity(row, max_distance))) for row in rows}


def add_external_evidence(feature: dict[str, float], pair: tuple[int, int], evidence: dict[tuple[int, int], float], weight: float) -> None:
    if pair not in evidence:
        return
    feature["external_similarity"] = evidence[pair]
    feature["external_edge_weight"] = weight


def is_stable_surface(stat) -> bool:
    return stat.geometry_type in {"horizontal", "vertical"}


def is_uncertain_fragment(stat) -> bool:
    return stat.geometry_type in {"unknown", "mixed", "rough_mixed"}


def linearized_cells(xyz: np.ndarray, cell_size: float) -> tuple[np.ndarray, tuple[int, int]]:
    cells = np.floor(xyz.astype(np.float64, copy=False) / max(cell_size, 1e-9)).astype(np.int64, copy=False)
    shifted = cells - cells.min(axis=0)[None, :]
    spans = shifted.max(axis=0) + 1
    stride_y = int(spans[2])
    stride_x = int(spans[1] * spans[2])
    return shifted[:, 0] * stride_x + shifted[:, 1] * stride_y + shifted[:, 2], (stride_x, stride_y)


def uncertain_fragment_bridge(a, b, feature: dict[str, float], args: argparse.Namespace) -> bool:
    if not getattr(args, "enable_uncertain_fragment_candidates", False):
        return False
    stable, uncertain = (a, b) if is_stable_surface(a) and is_uncertain_fragment(b) else (b, a)
    if not is_stable_surface(stable) or not is_uncertain_fragment(uncertain):
        return False
    if stable.count < args.uncertain_min_stable_voxels or uncertain.count > args.uncertain_max_fragment_voxels:
        return False
    return (
        float(feature.get("uncertain_contact_points", 0.0)) >= args.uncertain_min_contact_points
        and float(feature.get("contact_color_distance", args.max_color_distance)) <= args.uncertain_max_color_distance
        and float(feature.get("bbox_gap", 1e9)) <= args.uncertain_max_bbox_gap
    )


def build_uncertain_fragment_candidates(
    arrays: dict[str, np.ndarray],
    labels: np.ndarray,
    stats: dict[int, object],
    existing_pairs: set[tuple[int, int]],
    args: argparse.Namespace,
) -> dict[tuple[int, int], dict[str, float]]:
    if not getattr(args, "enable_uncertain_fragment_candidates", False):
        return {}
    stable_ids = [
        pid
        for pid, stat in stats.items()
        if is_stable_surface(stat) and stat.count >= args.uncertain_min_stable_voxels
    ]
    stable_ids.sort(key=lambda pid: stats[pid].count, reverse=True)
    stable_ids = stable_ids[: args.uncertain_max_stable_patches]
    if not stable_ids:
        return {}

    linear, (stride_x, stride_y) = linearized_cells(arrays["xyz"], args.uncertain_cell_size)
    order = np.argsort(linear, kind="stable")
    sorted_linear = linear[order]
    sorted_labels = labels[order]
    offsets = [
        dx * stride_x + dy * stride_y + dz
        for dx in range(-args.uncertain_radius, args.uncertain_radius + 1)
        for dy in range(-args.uncertain_radius, args.uncertain_radius + 1)
        for dz in range(-args.uncertain_radius, args.uncertain_radius + 1)
    ]

    out: dict[tuple[int, int], dict[str, float]] = {}
    for stable_id in stable_ids:
        stable = stats[stable_id]
        patch_cells = np.unique(linear[labels == stable_id])
        if len(patch_cells) > args.uncertain_max_cells_per_patch:
            step = max(1, len(patch_cells) // args.uncertain_max_cells_per_patch)
            patch_cells = patch_cells[::step][: args.uncertain_max_cells_per_patch]
        neighbor_counts: dict[int, int] = {}
        for offset in offsets:
            query = np.unique(patch_cells + offset)
            left = np.searchsorted(sorted_linear, query, side="left")
            right = np.searchsorted(sorted_linear, query, side="right")
            found = left < right
            for lo, hi in zip(left[found].tolist(), right[found].tolist(), strict=True):
                for label in sorted_labels[lo:hi].tolist():
                    label = int(label)
                    if label == stable_id:
                        continue
                    neighbor_counts[label] = neighbor_counts.get(label, 0) + 1
        kept = 0
        for uncertain_id, contact_points in sorted(neighbor_counts.items(), key=lambda row: row[1], reverse=True):
            uncertain = stats.get(uncertain_id)
            if uncertain is None:
                continue
            pair = (min(stable_id, uncertain_id), max(stable_id, uncertain_id))
            if pair in existing_pairs or pair in out:
                continue
            color = float(np.linalg.norm(stable.mean_rgb - uncertain.mean_rgb))
            feature = {
                "bbox_gap": bbox_gap(stable, uncertain),
                "contact_color_distance": color,
                "contact_color_p90": color,
                "contact_normal_score": normal_score(stable.mean_normal, uncertain.mean_normal),
                "contact_support": min(1.0, float(contact_points) / max(float(min(stable.count, uncertain.count)), 1.0)),
                "uncertain_contact_points": float(contact_points),
                "uncertain_fragment_candidate": 1.0,
            }
            if uncertain_fragment_bridge(stable, uncertain, feature, args):
                out[pair] = feature
                kept += 1
                if kept >= args.uncertain_max_candidates_per_stable:
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
    external_evidence = getattr(args, "external_edge_evidence", None) or {}
    external_weight = float(getattr(args, "external_edge_weight", 0.0))
    rows = []
    for pair, shared in edge_counts.items():
        a, b = pair
        feature = dict(edge_features.get(pair, {}))
        feature["contact_support"] = float(shared) / max(float(min(stats[a].count, stats[b].count)), 1.0)
        add_external_evidence(feature, pair, external_evidence, external_weight)
        rows.append((edge_score(feature, args.max_color_distance), int(shared), pair, feature))
    near_candidates = build_near_bbox_candidates(stats, set(edge_counts), args)
    for pair, feature in near_candidates.items():
        add_external_evidence(feature, pair, external_evidence, external_weight)
        rows.append((edge_score(feature, args.max_color_distance), 0, pair, feature))
    uncertain_candidates = build_uncertain_fragment_candidates(arrays, labels, stats, set(edge_counts) | set(near_candidates), args)
    for pair, feature in uncertain_candidates.items():
        add_external_evidence(feature, pair, external_evidence, external_weight)
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
        can_uncertain_bridge = uncertain_fragment_bridge(sa, sb, feature, args)
        if min(sa.count, sb.count) < args.min_patch_voxels and not can_uncertain_bridge:
            rejects["small_patch"] = rejects.get("small_patch", 0) + 1
            continue
        accepted_reason = "score"
        if score < args.min_edge_score:
            if can_uncertain_bridge:
                accepted_reason = "uncertain_fragment_bridge"
            elif near_bbox_bridge(sa, sb, feature, args):
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
    params = {k: v for k, v in vars(args).items() if k != "external_edge_evidence"}
    params["external_edge_evidence_count"] = int(len(external_evidence))
    report = {
        "schema": "superpoint-graph-cluster/v1",
        "input_patch_count": int(len(set(labels.tolist()))),
        "output_patch_count": int(len(set(out.tolist()))),
        "edge_count": int(len(rows)),
        "touch_edge_count": int(len(edge_counts)),
        "near_bbox_candidate_count": int(len(near_candidates)),
        "uncertain_fragment_candidate_count": int(len(uncertain_candidates)),
        "external_edge_evidence_count": int(len(external_evidence)),
        "accepted_edges": int(accepted),
        "accepted_reasons": accepted_reasons,
        "reject_counts": rejects,
        "params": params,
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
    parser.add_argument("--enable-uncertain-fragment-candidates", action="store_true")
    parser.add_argument("--uncertain-cell-size", type=float, default=0.05)
    parser.add_argument("--uncertain-radius", type=int, default=1)
    parser.add_argument("--uncertain-min-stable-voxels", type=int, default=10000)
    parser.add_argument("--uncertain-max-fragment-voxels", type=int, default=5000)
    parser.add_argument("--uncertain-min-contact-points", type=int, default=16)
    parser.add_argument("--uncertain-max-color-distance", type=float, default=75.0)
    parser.add_argument("--uncertain-max-bbox-gap", type=float, default=0.06)
    parser.add_argument("--uncertain-max-cells-per-patch", type=int, default=30000)
    parser.add_argument("--uncertain-max-stable-patches", type=int, default=200)
    parser.add_argument("--uncertain-max-candidates-per-stable", type=int, default=8)
    parser.add_argument("--enable-structural-merge-veto", action="store_true")
    parser.add_argument("--structural-veto-min-bucket-ratio", type=float, default=0.20)
    parser.add_argument("--structural-veto-min-voxels", type=int, default=1000)
    parser.add_argument("--preview-stride", type=int, default=10)
    parser.add_argument("--max-source-patch-ids", type=int, default=24)
    parser.add_argument("--external-edge-evidence", type=Path)
    parser.add_argument("--external-edge-weight", type=float, default=0.15)
    parser.add_argument("--external-edge-max-distance", type=float, default=2.0)
    args = parser.parse_args()
    args.external_edge_evidence = load_external_edge_evidence(args.external_edge_evidence, args.external_edge_max_distance)
    return args


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
