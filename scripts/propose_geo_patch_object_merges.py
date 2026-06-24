#!/usr/bin/env python3
"""Propose geometry-patch object merge candidates without changing ownership.

This is an object-building aid after the patch optimizer.  It keeps the
one-voxel-one-owner patch labels intact and reports adjacent patch pairs that
look like the same higher-level object according to local contact, color,
geometry bucket, normal, bbox gap, and size balance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from optimize_patch_graph_energy import (
    PatchStats,
    bbox_gap,
    compatible_bucket_score,
    compute_patch_stats,
    normal_score,
    normalize_rows,
    read_labels,
    read_region_input,
)


def build_edge_counts(labels: np.ndarray, src: np.ndarray, dst: np.ndarray) -> dict[tuple[int, int], int]:
    if len(src) == 0:
        return {}
    a = labels[src]
    b = labels[dst]
    mask = a != b
    if not np.any(mask):
        return {}

    a = a[mask].astype(np.int64, copy=False)
    b = b[mask].astype(np.int64, copy=False)
    hi = a > b
    aa = a.copy()
    a = np.where(hi, b, a)
    b = np.where(hi, aa, b)
    max_label = int(labels.max())
    keys = a * (max_label + 1) + b
    uk, uc = np.unique(keys, return_counts=True)
    return {
        (int(k // (max_label + 1)), int(k % (max_label + 1))): int(c)
        for k, c in zip(uk.tolist(), uc.tolist())
    }


def build_grid6_edges(arrays: dict[str, np.ndarray], voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Build complete 6-neighbor voxel adjacency from XYZ, independent of the source graph."""
    xyz = arrays["xyz"].astype(np.float64, copy=False)
    if len(xyz) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    mins = xyz.min(axis=0)
    cells = np.floor((xyz - mins) / float(voxel_size) + 0.5).astype(np.int64)
    max_cell = cells.max(axis=0) + 3
    nx = int(max_cell[0])
    ny = int(max_cell[1])
    keys = cells[:, 0] + nx * (cells[:, 1] + ny * cells[:, 2])
    order = np.argsort(keys)
    sorted_keys = keys[order]

    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    deltas = [1, nx, nx * ny]
    for delta in deltas:
        target = keys + int(delta)
        pos = np.searchsorted(sorted_keys, target)
        valid = (pos < len(sorted_keys)) & (sorted_keys[np.minimum(pos, len(sorted_keys) - 1)] == target)
        if not np.any(valid):
            continue
        src_idx = np.flatnonzero(valid).astype(np.int32, copy=False)
        dst_idx = order[pos[valid]].astype(np.int32, copy=False)
        keep = src_idx != dst_idx
        src_parts.append(src_idx[keep])
        dst_parts.append(dst_idx[keep])
    if not src_parts:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    return np.concatenate(src_parts).astype(np.int32, copy=False), np.concatenate(dst_parts).astype(np.int32, copy=False)


def build_edge_contact_features(
    arrays: dict[str, np.ndarray],
    labels: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
) -> dict[tuple[int, int], dict[str, float]]:
    """Summarize appearance/normal evidence only at the patch contact boundary."""
    if len(src) == 0:
        return {}
    la = labels[src]
    lb = labels[dst]
    mask = la != lb
    if not np.any(mask):
        return {}

    src_m = src[mask]
    dst_m = dst[mask]
    la = la[mask].astype(np.int64, copy=False)
    lb = lb[mask].astype(np.int64, copy=False)
    swap = la > lb
    a = np.where(swap, lb, la)
    b = np.where(swap, la, lb)
    idx_a = np.where(swap, dst_m, src_m)
    idx_b = np.where(swap, src_m, dst_m)

    max_label = int(labels.max())
    keys = a * (max_label + 1) + b
    uk, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)

    rgb = arrays["rgb"].astype(np.float64, copy=False)
    normals = normalize_rows(arrays["normal"].astype(np.float64, copy=False))
    rgb_a = np.zeros((len(uk), 3), dtype=np.float64)
    rgb_b = np.zeros((len(uk), 3), dtype=np.float64)
    n_a = np.zeros((len(uk), 3), dtype=np.float64)
    n_b = np.zeros((len(uk), 3), dtype=np.float64)
    np.add.at(rgb_a, inv, rgb[idx_a])
    np.add.at(rgb_b, inv, rgb[idx_b])
    np.add.at(n_a, inv, normals[idx_a])
    np.add.at(n_b, inv, normals[idx_b])

    rgb_a /= counts[:, None]
    rgb_b /= counts[:, None]
    contact_color_distance = np.linalg.norm(rgb_a - rgb_b, axis=1)
    nrm_a = np.linalg.norm(n_a, axis=1)
    nrm_b = np.linalg.norm(n_b, axis=1)
    dot = np.sum(n_a * n_b, axis=1)
    contact_normal_score = np.full(len(uk), 0.5, dtype=np.float64)
    ok = (nrm_a > 1e-9) & (nrm_b > 1e-9)
    contact_normal_score[ok] = np.clip(dot[ok] / (nrm_a[ok] * nrm_b[ok]), 0.0, 1.0)

    out: dict[tuple[int, int], dict[str, float]] = {}
    for i, key in enumerate(uk.tolist()):
        out[(int(key // (max_label + 1)), int(key % (max_label + 1)))] = {
            "shared_edges": int(counts[i]),
            "contact_color_distance": float(contact_color_distance[i]),
            "contact_normal_score": float(contact_normal_score[i]),
        }
    return out


def size_balance_score(a: PatchStats, b: PatchStats) -> float:
    small = max(float(min(a.count, b.count)), 1.0)
    large = max(float(max(a.count, b.count)), 1.0)
    return math.sqrt(small / large)


def structural_multimaterial_allowed(a: PatchStats, b: PatchStats, f: dict[str, float], args: argparse.Namespace) -> bool:
    if not args.enable_structural_multimaterial:
        return False
    geom = {a.geometry_type, b.geometry_type}
    if geom == {"horizontal", "vertical"}:
        return False
    if "horizontal" in geom and not geom <= {"horizontal", "unknown", "mixed", "rough_mixed"}:
        return False
    stable_pair = geom <= {"horizontal", "vertical", "unknown", "mixed"}
    if stable_pair and f["normal_score"] < args.structural_min_normal_score:
        return False
    if f["structural_score"] < args.min_structural_score:
        return False
    if f["contact_ratio_min"] < args.structural_min_contact_ratio:
        return False
    if f["shared_edges"] < args.structural_min_shared_edges:
        return False
    if f["bbox_gap"] > args.structural_max_bbox_gap:
        return False
    return True


def candidate_features(
    a: PatchStats,
    b: PatchStats,
    shared_edges: int,
    contact_features: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, float | str]:
    patch_color_dist = float(np.linalg.norm(a.mean_rgb - b.mean_rgb))
    contact_color_dist = float(contact_features.get("contact_color_distance", -1.0))
    color_dist = contact_color_dist if args.use_contact_evidence and contact_color_dist >= 0 else patch_color_dist
    color_score = max(0.0, min(1.0, 1.0 - color_dist / max(args.max_color_distance, 1e-6)))
    bucket_score = compatible_bucket_score(a.geometry_type, b.geometry_type)
    patch_normal_score = normal_score(a.mean_normal, b.mean_normal)
    contact_normal_score = float(contact_features.get("contact_normal_score", -1.0))
    n_score = contact_normal_score if args.use_contact_evidence and contact_normal_score >= 0 else patch_normal_score
    gap = bbox_gap(a, b)
    gap_score = max(0.0, min(1.0, 1.0 - gap / max(args.max_bbox_gap, 1e-6)))
    contact_ratio = float(shared_edges) / max(float(min(a.count, b.count)), 1.0)
    contact_score = max(0.0, min(1.0, contact_ratio / max(args.contact_ratio_norm, 1e-6)))
    balance = size_balance_score(a, b)
    size_ratio = float(max(a.count, b.count)) / max(float(min(a.count, b.count)), 1.0)
    big_mixed_attachment = 1.0 if (
        max(a.count, b.count) >= args.big_anchor_voxels
        and min(a.count, b.count) <= args.small_fragment_voxels
        and "mixed" in {a.geometry_type, b.geometry_type}
    ) else 0.0
    score = (
        args.color_weight * color_score
        + args.bucket_weight * bucket_score
        + args.normal_weight * n_score
        + args.gap_weight * gap_score
        + args.contact_weight * contact_score
        + args.balance_weight * balance
    ) / max(
        args.color_weight
        + args.bucket_weight
        + args.normal_weight
        + args.gap_weight
        + args.contact_weight
        + args.balance_weight,
        1e-9,
    )
    structural_score = (
        args.structural_bucket_weight * bucket_score
        + args.structural_normal_weight * n_score
        + args.structural_gap_weight * gap_score
        + args.structural_contact_weight * contact_score
        + args.structural_balance_weight * balance
    ) / max(
        args.structural_bucket_weight
        + args.structural_normal_weight
        + args.structural_gap_weight
        + args.structural_contact_weight
        + args.structural_balance_weight,
        1e-9,
    )
    return {
        "score": score,
        "color_distance": color_dist,
        "patch_color_distance": patch_color_dist,
        "contact_color_distance": contact_color_dist,
        "color_score": color_score,
        "bucket_score": bucket_score,
        "normal_score": n_score,
        "patch_normal_score": patch_normal_score,
        "contact_normal_score": contact_normal_score,
        "bbox_gap": gap,
        "gap_score": gap_score,
        "shared_edges": float(shared_edges),
        "contact_ratio_min": contact_ratio,
        "contact_score": contact_score,
        "size_balance": balance,
        "size_ratio": size_ratio,
        "big_mixed_attachment": big_mixed_attachment,
        "structural_score": structural_score,
        "merge_class": "same_material",
    }


def reject_reason(a: PatchStats, b: PatchStats, f: dict[str, float | str], args: argparse.Namespace) -> str | None:
    if min(a.count, b.count) < args.min_patch_voxels:
        return "small_patch"
    if f["shared_edges"] < args.min_shared_edges:
        return "low_shared_edges"
    if f["contact_ratio_min"] < args.min_contact_ratio:
        return "low_contact_ratio"
    if f["bbox_gap"] > args.max_bbox_gap:
        return "bbox_gap"
    if f["color_distance"] > args.max_color_distance:
        if structural_multimaterial_allowed(a, b, f, args):
            f["merge_class"] = "structural_multimaterial"
            return None
        return "color_distance"
    if f["normal_score"] < args.min_normal_score and {a.geometry_type, b.geometry_type} <= {"horizontal", "vertical"}:
        if structural_multimaterial_allowed(a, b, f, args):
            f["merge_class"] = "structural_multimaterial"
            return None
        return "stable_normal_mismatch"
    if f["bucket_score"] < args.min_bucket_score:
        return "bucket_mismatch"
    if f["score"] < args.min_score:
        if structural_multimaterial_allowed(a, b, f, args):
            f["merge_class"] = "structural_multimaterial"
            return None
        return "score"
    return None


def propose(arrays: dict[str, np.ndarray], labels: np.ndarray, src: np.ndarray, dst: np.ndarray, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stats = compute_patch_stats(arrays, labels)
    edge_source = args.edge_source
    if edge_source == "grid6":
        src, dst = build_grid6_edges(arrays, args.grid_voxel_size)
    edge_features = build_edge_contact_features(arrays, labels, src, dst)
    edge_counts = {pair: int(row["shared_edges"]) for pair, row in edge_features.items()}
    rows: list[dict[str, Any]] = []
    reject_counts: Counter[str] = Counter()
    for (a_id, b_id), shared in edge_counts.items():
        a = stats.get(a_id)
        b = stats.get(b_id)
        if a is None or b is None:
            continue
        f = candidate_features(a, b, shared, edge_features.get((a_id, b_id), {}), args)
        reason = reject_reason(a, b, f, args)
        if reason:
            reject_counts[reason] += 1
            continue
        rows.append(
            {
                "patch_a": int(a_id),
                "patch_b": int(b_id),
                "voxels_a": int(a.count),
                "voxels_b": int(b.count),
                "geometry_a": a.geometry_type,
                "geometry_b": b.geometry_type,
                "centroid_a": a.centroid.tolist(),
                "centroid_b": b.centroid.tolist(),
                "mean_rgb_a": a.mean_rgb.tolist(),
                "mean_rgb_b": b.mean_rgb.tolist(),
                **{k: (float(v) if isinstance(v, (int, float)) else v) for k, v in f.items()},
            }
        )
    rows.sort(key=lambda row: (row["score"], row["shared_edges"]), reverse=True)
    if args.max_candidates > 0:
        rows = rows[: args.max_candidates]
    report = {
        "schema": "geo-patch-object-merge-candidates/v1",
        "patch_count": len(stats),
        "edge_pair_count": len(edge_counts),
        "edge_source": edge_source,
        "candidate_count": len(rows),
        "reject_counts": dict(reject_counts),
        "candidate_geometry_pairs": {
            " + ".join(pair): int(count)
            for pair, count in Counter(tuple(sorted((row["geometry_a"], row["geometry_b"]))) for row in rows).most_common()
        },
        "big_mixed_attachment_count": sum(1 for row in rows if row["big_mixed_attachment"] > 0),
        "merge_class_counts": dict(Counter(str(row.get("merge_class", "unknown")) for row in rows)),
        "params": vars(args),
    }
    return rows, report


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "patch_a",
        "patch_b",
        "voxels_a",
        "voxels_b",
        "geometry_a",
        "geometry_b",
        "score",
        "color_distance",
        "bucket_score",
        "normal_score",
        "bbox_gap",
        "shared_edges",
        "contact_ratio_min",
        "size_balance",
        "size_ratio",
        "big_mixed_attachment",
        "structural_score",
        "merge_class",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-input", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", default="geo_patch_object_merge_candidates")

    parser.add_argument("--edge-source", choices=["region", "grid6"], default="region")
    parser.add_argument("--grid-voxel-size", type=float, default=0.03)
    parser.add_argument("--min-patch-voxels", type=int, default=400)
    parser.add_argument("--min-shared-edges", type=int, default=8)
    parser.add_argument("--min-contact-ratio", type=float, default=0.015)
    parser.add_argument("--max-bbox-gap", type=float, default=0.18)
    parser.add_argument("--max-color-distance", type=float, default=90.0)
    parser.add_argument("--min-normal-score", type=float, default=0.45)
    parser.add_argument("--min-bucket-score", type=float, default=0.55)
    parser.add_argument("--min-score", type=float, default=0.62)
    parser.add_argument("--contact-ratio-norm", type=float, default=0.18)
    parser.add_argument("--max-candidates", type=int, default=20000)
    parser.add_argument("--big-anchor-voxels", type=int, default=100000)
    parser.add_argument("--small-fragment-voxels", type=int, default=1200)
    parser.add_argument("--use-contact-evidence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-structural-multimaterial", action="store_true")
    parser.add_argument("--min-structural-score", type=float, default=0.74)
    parser.add_argument("--structural-min-contact-ratio", type=float, default=0.035)
    parser.add_argument("--structural-min-shared-edges", type=int, default=24)
    parser.add_argument("--structural-min-normal-score", type=float, default=0.62)
    parser.add_argument("--structural-max-bbox-gap", type=float, default=0.08)

    parser.add_argument("--color-weight", type=float, default=0.25)
    parser.add_argument("--bucket-weight", type=float, default=0.20)
    parser.add_argument("--normal-weight", type=float, default=0.18)
    parser.add_argument("--gap-weight", type=float, default=0.12)
    parser.add_argument("--contact-weight", type=float, default=0.20)
    parser.add_argument("--balance-weight", type=float, default=0.05)
    parser.add_argument("--structural-bucket-weight", type=float, default=0.24)
    parser.add_argument("--structural-normal-weight", type=float, default=0.24)
    parser.add_argument("--structural-gap-weight", type=float, default=0.16)
    parser.add_argument("--structural-contact-weight", type=float, default=0.28)
    parser.add_argument("--structural-balance-weight", type=float, default=0.08)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    arrays, src, dst = read_region_input(args.region_input)
    labels = read_labels(args.labels)
    rows, report = propose(arrays, labels, src, dst, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / f"{args.output_stem}.jsonl"
    csv_path = args.output_dir / f"{args.output_stem}.csv"
    report_path = args.output_dir / f"{args.output_stem}_report.json"
    write_jsonl(jsonl_path, rows)
    write_csv(csv_path, rows)
    report.update(
        {
            "output_jsonl": str(jsonl_path),
            "output_csv": str(csv_path),
            "output_report": str(report_path),
        }
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
