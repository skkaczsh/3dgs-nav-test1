#!/usr/bin/env python3
"""Build object labels from optimized geo patches and clean merge candidates.

This stage groups patch ids into object ids but keeps voxel ownership exclusive:
each voxel maps to exactly one object.  Risky big-mixed attachments remain
separate and are reported for later review instead of being auto-merged.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from optimize_patch_graph_energy import compute_patch_stats, read_labels, read_region_input, write_labels, write_ply


class UnionFind:
    def __init__(self, items: list[int]) -> None:
        self.parent = {int(item): int(item) for item in items}
        self.rank = {int(item): 0 for item in items}

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


def candidate_is_attachment(row: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    """Accept only high-confidence small-fragment attachment to a large patch.

    Big mixed attachments are structurally different from same-scale object
    merges.  A small fragment can be absorbed by a large patch only when the
    boundary contact and local appearance are both strong.  This keeps the
    patch partition exclusive while avoiding a global threshold relaxation.
    """
    small = min(int(row.get("voxels_a", 0)), int(row.get("voxels_b", 0)))
    large = max(int(row.get("voxels_a", 0)), int(row.get("voxels_b", 0)))
    if small <= 0 or large <= 0:
        return False, "attachment_missing_size"
    if small > args.attachment_max_fragment_voxels:
        return False, "attachment_fragment_too_large"
    if large < args.attachment_min_anchor_voxels:
        return False, "attachment_anchor_too_small"
    if float(row.get("size_ratio", 0.0)) < args.attachment_min_size_ratio:
        return False, "attachment_size_ratio"
    if float(row.get("score", 0.0)) < args.attachment_min_score:
        return False, "attachment_score"
    if float(row.get("contact_ratio_min", 0.0)) < args.attachment_min_contact_ratio:
        return False, "attachment_contact_ratio"
    if float(row.get("shared_edges", 0.0)) < args.attachment_min_shared_edges:
        return False, "attachment_shared_edges"
    if float(row.get("color_distance", 1e9)) > args.attachment_max_color_distance:
        return False, "attachment_color_distance"
    if float(row.get("normal_score", 0.0)) < args.attachment_min_normal_score:
        return False, "attachment_normal"
    if float(row.get("bbox_gap", 1e9)) > args.attachment_max_bbox_gap:
        return False, "attachment_bbox_gap"
    return True, "accepted_attachment"


def candidate_is_structural_multimaterial(row: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    """Accept same-object joins where material/color changes across a real boundary.

    This is for cases such as car panels with different normals, or windows and
    facade patches on the same building face.  It deliberately ignores global
    color distance, but requires strong local contact, compatible geometry, and
    a tight bbox gap so it cannot act as a general over-merge switch.
    """
    if not args.enable_structural_multimaterial:
        return False, "structural_multimaterial_disabled"
    if str(row.get("merge_class", "")) != "structural_multimaterial":
        return False, "structural_multimaterial_missing"
    if float(row.get("structural_score", 0.0)) < args.min_structural_score:
        return False, "structural_score"
    if float(row.get("contact_ratio_min", 0.0)) < args.structural_min_contact_ratio:
        return False, "structural_contact_ratio"
    if float(row.get("shared_edges", 0.0)) < args.structural_min_shared_edges:
        return False, "structural_shared_edges"
    if float(row.get("bbox_gap", 1e9)) > args.structural_max_bbox_gap:
        return False, "structural_bbox_gap"
    geom = {str(row.get("geometry_a", "")), str(row.get("geometry_b", ""))}
    if geom == {"horizontal", "vertical"}:
        return False, "structural_horizontal_vertical"
    stable_like = geom <= {"horizontal", "vertical", "unknown", "mixed"}
    if stable_like and float(row.get("normal_score", 0.0)) < args.structural_min_normal_score:
        return False, "structural_normal"
    return True, "accepted_structural_multimaterial"


def candidate_is_clean(row: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    if str(row.get("merge_class", "")) == "structural_multimaterial":
        return candidate_is_structural_multimaterial(row, args)
    if float(row.get("big_mixed_attachment", 0.0)) > 0:
        if args.enable_attachment_model:
            return candidate_is_attachment(row, args)
        if not args.allow_big_mixed_attachment:
            return False, "big_mixed_attachment"
    if float(row.get("score", 0.0)) < args.min_score:
        return False, "score"
    if float(row.get("contact_ratio_min", 0.0)) < args.min_contact_ratio:
        return False, "contact_ratio"
    if float(row.get("shared_edges", 0.0)) < args.min_shared_edges:
        return False, "shared_edges"
    if float(row.get("color_distance", 1e9)) > args.max_color_distance:
        return False, "color_distance"
    if float(row.get("bbox_gap", 1e9)) > args.max_bbox_gap:
        return False, "bbox_gap"
    geom = {str(row.get("geometry_a", "")), str(row.get("geometry_b", ""))}
    if args.same_geometry_only and len(geom) > 1:
        return False, "geometry_mismatch"
    stable = {"horizontal", "vertical"}
    if geom <= stable and len(geom) > 1:
        return False, "stable_geometry_mismatch"
    if float(row.get("normal_score", 0.0)) < args.min_normal_score and geom <= stable:
        return False, "stable_normal"
    return True, "accepted"


def build_patch_to_object(labels: np.ndarray, candidates: list[dict[str, Any]], args: argparse.Namespace) -> tuple[dict[int, int], dict[str, Any]]:
    patch_ids = [int(v) for v in np.unique(labels).tolist()]
    uf = UnionFind(patch_ids)
    reason_counts: Counter[str] = Counter()
    accepted_rows = 0
    for row in candidates:
        ok, reason = candidate_is_clean(row, args)
        reason_counts[reason] += 1
        if not ok:
            continue
        a = int(row["patch_a"])
        b = int(row["patch_b"])
        if a not in uf.parent or b not in uf.parent:
            reason_counts["missing_patch"] += 1
            continue
        if uf.union(a, b):
            accepted_rows += 1

    root_to_object: dict[int, int] = {}
    patch_to_object: dict[int, int] = {}
    for patch_id in patch_ids:
        root = uf.find(patch_id)
        if root not in root_to_object:
            root_to_object[root] = len(root_to_object) + 1
        patch_to_object[patch_id] = root_to_object[root]
    report = {
        "input_patch_count": len(patch_ids),
        "input_candidate_count": len(candidates),
        "accepted_candidate_rows": accepted_rows,
        "output_object_count": len(root_to_object),
        "candidate_reason_counts": dict(reason_counts),
    }
    return patch_to_object, report


def remap_labels(labels: np.ndarray, patch_to_object: dict[int, int]) -> np.ndarray:
    max_label = int(labels.max())
    lut = np.zeros(max_label + 1, dtype=np.int32)
    for patch_id, object_id in patch_to_object.items():
        if patch_id >= len(lut):
            raise ValueError(f"patch id exceeds LUT size: {patch_id}>{max_label}")
        lut[patch_id] = int(object_id)
    return lut[labels]


def write_objects_jsonl(
    path: Path,
    arrays: dict[str, np.ndarray],
    object_labels: np.ndarray,
    patch_labels: np.ndarray,
    patch_to_object: dict[int, int],
) -> int:
    stats = compute_patch_stats(arrays, object_labels)
    patches_by_object: dict[int, list[int]] = defaultdict(list)
    for patch_id, object_id in patch_to_object.items():
        patches_by_object[int(object_id)].append(int(patch_id))
    with path.open("w", encoding="utf-8") as f:
        for object_id in sorted(stats):
            s = stats[object_id]
            patch_ids = sorted(patches_by_object.get(int(object_id), []))
            row = {
                "object_id": int(object_id),
                "voxel_count": int(s.count),
                "geometry_type": s.geometry_type,
                "semantic_label": s.geometry_type,
                "description": f"geo object from {len(patch_ids)} patches",
                "patch_count": len(patch_ids),
                "patch_ids": patch_ids[:64],
                "patch_ids_truncated": len(patch_ids) > 64,
                "bbox_3d": {"min": s.bbox_min.tolist(), "max": s.bbox_max.tolist()},
                "centroid": s.centroid.tolist(),
                "mean_rgb": s.mean_rgb.tolist(),
                "mean_normal": s.mean_normal.tolist(),
                "bucket_counts": dict(s.bucket_counts),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-input", type=Path, required=True)
    parser.add_argument("--patch-labels", type=Path, required=True)
    parser.add_argument("--candidates-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", default="geo_patch_objects")
    parser.add_argument("--preview-stride", type=int, default=10)

    parser.add_argument("--min-score", type=float, default=0.78)
    parser.add_argument("--min-contact-ratio", type=float, default=0.08)
    parser.add_argument("--min-shared-edges", type=int, default=32)
    parser.add_argument("--max-color-distance", type=float, default=55.0)
    parser.add_argument("--max-bbox-gap", type=float, default=0.08)
    parser.add_argument("--min-normal-score", type=float, default=0.65)
    parser.add_argument("--same-geometry-only", action="store_true")
    parser.add_argument("--allow-big-mixed-attachment", action="store_true")
    parser.add_argument("--enable-attachment-model", action="store_true")
    parser.add_argument("--attachment-min-score", type=float, default=0.84)
    parser.add_argument("--attachment-min-contact-ratio", type=float, default=0.18)
    parser.add_argument("--attachment-min-shared-edges", type=int, default=80)
    parser.add_argument("--attachment-max-color-distance", type=float, default=28.0)
    parser.add_argument("--attachment-min-normal-score", type=float, default=0.75)
    parser.add_argument("--attachment-max-bbox-gap", type=float, default=0.05)
    parser.add_argument("--attachment-max-fragment-voxels", type=int, default=1200)
    parser.add_argument("--attachment-min-anchor-voxels", type=int, default=100000)
    parser.add_argument("--attachment-min-size-ratio", type=float, default=500.0)
    parser.add_argument("--enable-structural-multimaterial", action="store_true")
    parser.add_argument("--min-structural-score", type=float, default=0.74)
    parser.add_argument("--structural-min-contact-ratio", type=float, default=0.035)
    parser.add_argument("--structural-min-shared-edges", type=int, default=24)
    parser.add_argument("--structural-min-normal-score", type=float, default=0.62)
    parser.add_argument("--structural-max-bbox-gap", type=float, default=0.08)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    arrays, _src, _dst = read_region_input(args.region_input)
    patch_labels = read_labels(args.patch_labels)
    candidates = read_jsonl(args.candidates_jsonl)
    patch_to_object, report = build_patch_to_object(patch_labels, candidates, args)
    object_labels = remap_labels(patch_labels, patch_to_object)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = args.output_dir / f"{args.output_stem}_labels.bin"
    ply_path = args.output_dir / f"{args.output_stem}_stride{args.preview_stride}.ply"
    objects_path = args.output_dir / f"{args.output_stem}.jsonl"
    report_path = args.output_dir / f"{args.output_stem}_report.json"
    write_labels(labels_path, object_labels)
    preview_points = write_ply(ply_path, arrays, object_labels, args.preview_stride)
    object_count = write_objects_jsonl(objects_path, arrays, object_labels, patch_labels, patch_to_object)
    report.update(
        {
            "schema": "geo-patch-objects/v1",
            "region_input": str(args.region_input),
            "patch_labels": str(args.patch_labels),
            "candidates_jsonl": str(args.candidates_jsonl),
            "output_labels": str(labels_path),
            "output_ply": str(ply_path),
            "output_objects_jsonl": str(objects_path),
            "output_report": str(report_path),
            "jsonl_object_count": object_count,
            "preview_points": preview_points,
            "params": vars(args),
        }
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
