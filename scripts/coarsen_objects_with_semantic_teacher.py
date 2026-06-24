#!/usr/bin/env python3
"""Coarsen geometry objects using a validated semantic teacher.

The input object labels remain an exclusive partition.  This stage only merges
neighboring objects when the teacher semantics, geometry bucket, color, and
contact evidence all agree.  It is intentionally stricter than patch-level
region growing because it operates after geometry ownership has already been
formed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.export_frame_target_objects_for_viewer import LABEL_TO_SEMANTIC, SEMANTIC_COLORS
from scripts.optimize_patch_graph_energy import (
    PatchStats,
    compute_patch_stats,
    normal_score,
    read_labels,
    read_region_input,
    write_labels,
)
from scripts.propose_geo_patch_object_merges import build_grid6_edges

SKIP_LABELS = {"unknown", "ignore", "sky", "water", "other"}

LABEL_RULES = {
    "floor": {
        "geometries": {"horizontal", "mixed", "unknown", "rough_mixed"},
        "min_shared_edges": 8,
        "min_contact_ratio": 0.002,
        "max_color_distance": 72.0,
        "min_normal_score": 0.72,
        "stable_normal_only": True,
    },
    "wall": {
        "geometries": {"vertical", "mixed", "unknown", "rough_mixed"},
        "min_shared_edges": 8,
        "min_contact_ratio": 0.002,
        "max_color_distance": 78.0,
        "min_normal_score": 0.42,
        "stable_normal_only": True,
    },
    "grass": {
        "geometries": {"horizontal", "rough_mixed", "mixed", "unknown"},
        "min_shared_edges": 4,
        "min_contact_ratio": 0.0015,
        "max_color_distance": 95.0,
        "min_normal_score": 0.20,
        "stable_normal_only": False,
    },
    "railing": {
        "geometries": {"thin_linear", "vertical", "rough_mixed", "mixed", "unknown"},
        "min_shared_edges": 3,
        "min_contact_ratio": 0.001,
        "max_color_distance": 95.0,
        "min_normal_score": 0.10,
        "stable_normal_only": False,
    },
    "car": {
        "geometries": {"rough_mixed", "mixed", "unknown"},
        "min_shared_edges": 3,
        "min_contact_ratio": 0.001,
        "max_color_distance": 88.0,
        "min_normal_score": 0.10,
        "stable_normal_only": False,
    },
}


@dataclass
class UnionFind:
    parent: dict[int, int]
    rank: dict[int, int]

    @classmethod
    def from_items(cls, items: np.ndarray) -> "UnionFind":
        ids = [int(v) for v in items.tolist()]
        return cls({v: v for v in ids}, {v: 0 for v in ids})

    def find(self, item: int) -> int:
        item = int(item)
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def object_key(row: dict[str, Any]) -> int | None:
    for key in ("viewer_object_id", "object_id"):
        try:
            return int(row.get(key))
        except (TypeError, ValueError):
            continue
    return None


def build_edge_counts(labels: np.ndarray, src: np.ndarray, dst: np.ndarray) -> dict[tuple[int, int], int]:
    a = labels[src].astype(np.int64, copy=False)
    b = labels[dst].astype(np.int64, copy=False)
    mask = a != b
    if not np.any(mask):
        return {}
    a = a[mask]
    b = b[mask]
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    max_label = int(labels.max())
    keys = lo * (max_label + 1) + hi
    unique, counts = np.unique(keys, return_counts=True)
    return {
        (int(k // (max_label + 1)), int(k % (max_label + 1))): int(c)
        for k, c in zip(unique.tolist(), counts.tolist(), strict=True)
    }


def load_object_rows(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(path):
        oid = object_key(row)
        if oid is not None:
            out[int(oid)] = row
    return out


def row_label(row: dict[str, Any] | None) -> str:
    if row is None:
        return "unknown"
    return str(row.get("semantic_label") or "unknown")


def row_confidence(row: dict[str, Any] | None) -> float:
    if row is None:
        return 0.0
    try:
        return float(row.get("teacher_semantic_confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def is_teacher_transferred(row: dict[str, Any] | None) -> bool:
    if row is None:
        return False
    return str(row.get("semantic_transfer_status") or "") == "teacher_semantic_transfer"


def semantic_id(label: str) -> int:
    return int(LABEL_TO_SEMANTIC.get(label, 0))


def geometry_allowed(label: str, stats: PatchStats) -> bool:
    rule = LABEL_RULES.get(label)
    if rule is None:
        return False
    return stats.geometry_type in rule["geometries"]


def pair_decision(
    a_id: int,
    b_id: int,
    shared_edges: int,
    stats: dict[int, PatchStats],
    rows: dict[int, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[bool, str, dict[str, float | str]]:
    a = stats.get(a_id)
    b = stats.get(b_id)
    if a is None or b is None:
        return False, "missing_stats", {}
    label_a = row_label(rows.get(a_id))
    label_b = row_label(rows.get(b_id))
    unknown_absorb = False
    if label_a != label_b:
        if (
            args.enable_unknown_absorb
            and "unknown" in {label_a, label_b}
            and (label_a in LABEL_RULES or label_b in LABEL_RULES)
        ):
            known_id = a_id if label_a != "unknown" else b_id
            unknown_id = b_id if label_a != "unknown" else a_id
            known = stats.get(known_id)
            unknown = stats.get(unknown_id)
            if known is None or unknown is None:
                return False, "missing_stats", {}
            if unknown.count > args.unknown_absorb_max_voxels:
                return False, "unknown_absorb_too_large", {"label_a": label_a, "label_b": label_b, "unknown_voxels": unknown.count}
            label = label_a if label_a != "unknown" else label_b
            unknown_absorb = True
        else:
            return False, "label_mismatch", {"label_a": label_a, "label_b": label_b}
    else:
        label = label_a
    if label in SKIP_LABELS or label not in LABEL_RULES:
        return False, "label_not_mergeable", {"label": label}
    if not geometry_allowed(label, a) or not geometry_allowed(label, b):
        return False, "geometry_veto", {"label": label, "geometry_a": a.geometry_type, "geometry_b": b.geometry_type}

    rule = LABEL_RULES[label]
    min_count = max(float(min(a.count, b.count)), 1.0)
    contact_ratio = float(shared_edges) / min_count
    color_distance = float(np.linalg.norm(a.mean_rgb - b.mean_rgb))
    n_score = normal_score(a.mean_normal, b.mean_normal)
    teacher_boost = is_teacher_transferred(rows.get(a_id)) or is_teacher_transferred(rows.get(b_id))
    confidence = max(row_confidence(rows.get(a_id)), row_confidence(rows.get(b_id)))

    min_shared = int(rule["min_shared_edges"])
    min_contact = float(rule["min_contact_ratio"])
    max_color = float(rule["max_color_distance"])
    min_normal = float(rule["min_normal_score"])
    if teacher_boost and confidence >= args.teacher_boost_confidence:
        min_shared = max(2, int(round(min_shared * args.teacher_boost_shared_factor)))
        min_contact *= args.teacher_boost_contact_factor
        max_color *= args.teacher_boost_color_factor
    if unknown_absorb:
        min_shared = max(args.unknown_absorb_min_shared_edges, int(round(min_shared * args.unknown_absorb_shared_factor)))
        min_contact *= args.unknown_absorb_contact_factor
        max_color *= args.unknown_absorb_color_factor
        min_normal *= args.unknown_absorb_normal_factor

    if shared_edges < min_shared:
        return False, "shared_edges", {"label": label, "shared_edges": shared_edges, "min_shared_edges": min_shared}
    if contact_ratio < min_contact:
        return False, "contact_ratio", {"label": label, "contact_ratio": contact_ratio, "min_contact_ratio": min_contact}
    if color_distance > max_color:
        return False, "color_distance", {"label": label, "color_distance": color_distance, "max_color_distance": max_color}
    stable_pair = {a.geometry_type, b.geometry_type} <= {"horizontal", "vertical"}
    if bool(rule["stable_normal_only"]):
        stable_pair = stable_pair or a.geometry_type == b.geometry_type
    if stable_pair and n_score < min_normal:
        return False, "normal_score", {"label": label, "normal_score": n_score, "min_normal_score": min_normal}
    if max(a.count, b.count) >= args.large_object_voxels and min(a.count, b.count) >= args.large_object_voxels:
        if contact_ratio < args.large_large_min_contact_ratio:
            return False, "large_large_contact_ratio", {"label": label, "contact_ratio": contact_ratio}

    return True, "accepted", {
        "label": label,
        "contact_ratio": contact_ratio,
        "color_distance": color_distance,
        "normal_score": n_score,
        "teacher_boost": float(teacher_boost),
        "unknown_absorb": float(unknown_absorb),
        "confidence": confidence,
    }


def remap_to_contiguous(labels: np.ndarray, uf: UnionFind) -> tuple[np.ndarray, dict[int, int], dict[int, list[int]]]:
    old_ids = np.unique(labels)
    root_to_new: dict[int, int] = {}
    old_to_new: dict[int, int] = {}
    members: dict[int, list[int]] = defaultdict(list)
    for old_id in old_ids.tolist():
        root = uf.find(int(old_id))
        if root not in root_to_new:
            root_to_new[root] = len(root_to_new) + 1
        new_id = root_to_new[root]
        old_to_new[int(old_id)] = new_id
        members[new_id].append(int(old_id))
    lut = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    for old_id, new_id in old_to_new.items():
        lut[old_id] = int(new_id)
    return lut[labels], old_to_new, members


def merged_label(member_ids: list[int], rows: dict[int, dict[str, Any]]) -> tuple[str, Counter[str], Counter[str]]:
    label_votes: Counter[str] = Counter()
    status_votes: Counter[str] = Counter()
    for oid in member_ids:
        row = rows.get(oid)
        label = row_label(row)
        try:
            weight = int(row.get("voxel_count") or row.get("point_count") or 1) if row else 1
        except (TypeError, ValueError):
            weight = 1
        label_votes[label] += max(weight, 1)
        status_votes[str(row.get("semantic_transfer_status") or "unknown") if row else "missing"] += 1
    non_unknown = Counter({k: v for k, v in label_votes.items() if k != "unknown"})
    label = non_unknown.most_common(1)[0][0] if non_unknown else (label_votes.most_common(1)[0][0] if label_votes else "unknown")
    return label, label_votes, status_votes


def write_objects_jsonl(
    path: Path,
    arrays: dict[str, np.ndarray],
    new_labels: np.ndarray,
    members: dict[int, list[int]],
    rows: dict[int, dict[str, Any]],
) -> int:
    stats = compute_patch_stats(arrays, new_labels)
    with path.open("w", encoding="utf-8") as f:
        for object_id in sorted(stats):
            s = stats[object_id]
            member_ids = sorted(members.get(int(object_id), []))
            label, label_votes, status_votes = merged_label(member_ids, rows)
            row = {
                "object_id": int(object_id),
                "voxel_count": int(s.count),
                "geometry_type": s.geometry_type,
                "semantic_label": label,
                "semantic_id": semantic_id(label),
                "description": f"teacher-coarsened object from {len(member_ids)} v9 objects",
                "source_object_count": len(member_ids),
                "source_object_ids": member_ids[:128],
                "source_object_ids_truncated": len(member_ids) > 128,
                "bbox_3d": {"min": s.bbox_min.tolist(), "max": s.bbox_max.tolist()},
                "centroid": s.centroid.tolist(),
                "mean_rgb": s.mean_rgb.tolist(),
                "mean_normal": s.mean_normal.tolist(),
                "bucket_counts": dict(s.bucket_counts),
                "teacher_label_votes": dict(label_votes),
                "teacher_status_votes": dict(status_votes),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(stats)


def write_semantic_ply(path: Path, arrays: dict[str, np.ndarray], labels: np.ndarray, object_labels: dict[int, str], stride: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    idx = np.arange(0, len(labels), max(int(stride), 1), dtype=np.int64)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(idx)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property int object\nproperty uchar semantic\n")
        f.write("end_header\n")
        for i in idx.tolist():
            oid = int(labels[i])
            label = object_labels.get(oid, "unknown")
            sem = semantic_id(label)
            color = SEMANTIC_COLORS.get(sem, SEMANTIC_COLORS[0])
            x, y, z = arrays["xyz"][i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {color[0]} {color[1]} {color[2]} {oid} {sem}\n")
    return int(len(idx))


def coarsen(args: argparse.Namespace) -> dict[str, Any]:
    arrays, src, dst = read_region_input(args.region_input)
    labels = read_labels(args.object_labels)
    if args.use_grid6:
        src, dst = build_grid6_edges(arrays, args.voxel_size)
    rows = load_object_rows(args.objects_jsonl)
    stats = compute_patch_stats(arrays, labels)
    edge_counts = build_edge_counts(labels, src, dst)
    uf = UnionFind.from_items(np.unique(labels))

    reason_counts: Counter[str] = Counter()
    accepted_by_label: Counter[str] = Counter()
    merge_examples: list[dict[str, Any]] = []
    for (a_id, b_id), shared_edges in sorted(edge_counts.items(), key=lambda item: item[1], reverse=True):
        ok, reason, details = pair_decision(a_id, b_id, shared_edges, stats, rows, args)
        reason_counts[reason] += 1
        if not ok:
            continue
        if uf.union(a_id, b_id):
            label = str(details.get("label", "unknown"))
            accepted_by_label[label] += 1
            if len(merge_examples) < args.max_merge_examples:
                merge_examples.append({"object_a": a_id, "object_b": b_id, "shared_edges": shared_edges, **details})

    new_labels, _old_to_new, members = remap_to_contiguous(labels, uf)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = args.output_dir / f"{args.output_stem}_labels.bin"
    objects_path = args.output_dir / f"{args.output_stem}.jsonl"
    ply_path = args.output_dir / f"{args.output_stem}_stride{args.preview_stride}.ply"
    report_path = args.output_dir / f"{args.output_stem}_report.json"

    write_labels(labels_path, new_labels)
    object_count = write_objects_jsonl(objects_path, arrays, new_labels, members, rows)
    object_rows = load_object_rows(objects_path)
    object_labels = {oid: row_label(row) for oid, row in object_rows.items()}
    preview_points = write_semantic_ply(ply_path, arrays, new_labels, object_labels, args.preview_stride)

    label_counts = Counter(row_label(row) for row in object_rows.values())
    report = {
        "schema": "teacher-semantic-object-coarsen/v1",
        "region_input": str(args.region_input),
        "input_object_labels": str(args.object_labels),
        "input_objects_jsonl": str(args.objects_jsonl),
        "output_labels": str(labels_path),
        "output_objects_jsonl": str(objects_path),
        "output_ply": str(ply_path),
        "input_object_count": len(stats),
        "output_object_count": object_count,
        "accepted_merge_rows": sum(accepted_by_label.values()),
        "accepted_by_label": dict(accepted_by_label),
        "candidate_edge_count": len(edge_counts),
        "edge_source": "grid6" if args.use_grid6 else "region_input",
        "reason_counts": dict(reason_counts),
        "label_object_counts": dict(label_counts),
        "preview_points": preview_points,
        "merge_examples": merge_examples,
        "params": vars(args),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-input", type=Path, required=True)
    parser.add_argument("--object-labels", type=Path, required=True)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", default="objects_teacher_coarsened")
    parser.add_argument("--preview-stride", type=int, default=10)
    parser.add_argument("--use-grid6", action="store_true")
    parser.add_argument("--voxel-size", type=float, default=0.03)
    parser.add_argument("--teacher-boost-confidence", type=float, default=0.45)
    parser.add_argument("--teacher-boost-shared-factor", type=float, default=0.7)
    parser.add_argument("--teacher-boost-contact-factor", type=float, default=0.7)
    parser.add_argument("--teacher-boost-color-factor", type=float, default=1.15)
    parser.add_argument("--large-object-voxels", type=int, default=120000)
    parser.add_argument("--large-large-min-contact-ratio", type=float, default=0.006)
    parser.add_argument("--enable-unknown-absorb", action="store_true")
    parser.add_argument("--unknown-absorb-max-voxels", type=int, default=2500)
    parser.add_argument("--unknown-absorb-min-shared-edges", type=int, default=6)
    parser.add_argument("--unknown-absorb-shared-factor", type=float, default=1.5)
    parser.add_argument("--unknown-absorb-contact-factor", type=float, default=2.0)
    parser.add_argument("--unknown-absorb-color-factor", type=float, default=0.75)
    parser.add_argument("--unknown-absorb-normal-factor", type=float, default=1.1)
    parser.add_argument("--max-merge-examples", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    report = coarsen(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
