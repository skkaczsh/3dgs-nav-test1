#!/usr/bin/env python3
"""Fuse strict accepted fine-object candidates into object-level QA records."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


STRICT_KEEP_STATUS = 1


def read_ascii_ply(path: Path) -> tuple[list[str], int, np.ndarray]:
    props: list[str] = []
    vertex_count = 0
    header_lines = 0
    in_vertex = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            header_lines += 1
            s = line.strip()
            if s.startswith("element vertex"):
                vertex_count = int(s.split()[-1])
                in_vertex = True
            elif s.startswith("element "):
                in_vertex = False
            elif in_vertex and s.startswith("property"):
                props.append(s.split()[-1])
            elif s == "end_header":
                break
    if vertex_count == 0:
        return props, header_lines, np.empty((0, len(props)), dtype=np.float32)
    data = np.loadtxt(path, skiprows=header_lines, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return props, header_lines, data


def bbox_distance(a: dict, b: dict) -> float:
    amin = np.array(a["bbox_3d"]["min"], dtype=np.float64)
    amax = np.array(a["bbox_3d"]["max"], dtype=np.float64)
    bmin = np.array(b["bbox_3d"]["min"], dtype=np.float64)
    bmax = np.array(b["bbox_3d"]["max"], dtype=np.float64)
    gap = np.maximum(0.0, np.maximum(bmin - amax, amin - bmax))
    return float(np.linalg.norm(gap))


def angle_degrees(a: list[float], b: list[float]) -> float:
    av = np.array(a, dtype=np.float64)
    bv = np.array(b, dtype=np.float64)
    an = np.linalg.norm(av)
    bn = np.linalg.norm(bv)
    if an < 1e-9 or bn < 1e-9:
        return 0.0
    dot = abs(float(np.dot(av / an, bv / bn)))
    return float(math.degrees(math.acos(np.clip(dot, -1.0, 1.0))))


def pca_normal_from_eigen_hint(row: dict) -> list[float]:
    # Candidate reports do not carry eigenvectors. Use a neutral normal and keep
    # normal-angle gating non-binding for now.
    return [0.0, 0.0, 1.0]


def object_color(object_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(object_id * 101 + 1907)
    return tuple(int(x) for x in rng.integers(70, 245, 3))


def load_kept_candidates(strict_review: Path) -> list[dict]:
    raw = json.loads(strict_review.read_text(encoding="utf-8"))
    accepted_report_path = Path(raw["accepted_report"])
    accepted = json.loads(accepted_report_path.read_text(encoding="utf-8"))
    accepted_by_id = {int(row["candidate_id"]): row for row in accepted.get("top_candidates", [])}
    status_by_id = {int(row["candidate_id"]): int(row.get("strict_status", 0)) for row in raw.get("candidates", [])}
    rows = []
    for candidate_id, status in status_by_id.items():
        if status != STRICT_KEEP_STATUS:
            continue
        row = dict(accepted_by_id[candidate_id])
        out = dict(row)
        out["centroid"] = row.get("centroid", None)
        out["normal"] = pca_normal_from_eigen_hint(row)
        rows.append(out)
    return sorted(rows, key=lambda r: int(r["candidate_id"]))


def load_candidates_from_accepted_report(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for row in raw.get("top_candidates", []):
        out = dict(row)
        out["candidate_id"] = int(out["candidate_id"])
        out["points"] = int(out["points"])
        out["source_type"] = str(out.get("source_type", "grounded_detector_mask"))
        out["source_cluster"] = int(out.get("source_cluster", out["candidate_id"]))
        out["subcluster"] = int(out.get("subcluster", -1))
        out["centroid"] = row.get("centroid", None)
        out["normal"] = pca_normal_from_eigen_hint(row)
        rows.append(out)
    return sorted(rows, key=lambda r: int(r["candidate_id"]))


def create_object(index: int, cand: dict) -> dict:
    return {
        "fine_object_id": f"fine_obj_{index:04d}",
        "candidate_ids": [int(cand["candidate_id"])],
        "candidate_count": 1,
        "point_count": int(cand["points"]),
        "bbox_3d": cand["bbox_3d"],
        "centroid": cand["centroid"],
        "mean_visual_color": cand["mean_visual_color"],
        "color_sum": (np.array(cand["mean_visual_color"], dtype=np.float64) * max(int(cand["points"]), 1)).tolist(),
        "linearity_mean": float(cand.get("linearity", 0.0)),
        "planarity_mean": float(cand.get("planarity", 0.0)),
        "source_types": Counter([cand["source_type"]]),
        "source_clusters": Counter([str(cand["source_cluster"])]),
        "_candidate_records": [cand],
    }


def merge_object(obj: dict, cand: dict) -> None:
    old_count = max(int(obj["point_count"]), 1)
    new_count = max(int(cand["points"]), 1)
    total = old_count + new_count
    obj["candidate_ids"].append(int(cand["candidate_id"]))
    obj["candidate_count"] = len(obj["candidate_ids"])
    obj["point_count"] = int(total)
    bmin = np.minimum(np.array(obj["bbox_3d"]["min"], dtype=np.float64), np.array(cand["bbox_3d"]["min"], dtype=np.float64))
    bmax = np.maximum(np.array(obj["bbox_3d"]["max"], dtype=np.float64), np.array(cand["bbox_3d"]["max"], dtype=np.float64))
    obj["bbox_3d"] = {"min": [float(x) for x in bmin], "max": [float(x) for x in bmax]}
    centroid = (np.array(obj["centroid"]) * old_count + np.array(cand["centroid"]) * new_count) / total
    obj["centroid"] = [float(x) for x in centroid]
    color_sum = np.array(obj["color_sum"], dtype=np.float64) + np.array(cand["mean_visual_color"], dtype=np.float64) * new_count
    obj["color_sum"] = [float(x) for x in color_sum]
    obj["mean_visual_color"] = [float(x) for x in color_sum / total]
    obj["_candidate_records"].append(cand)
    obj["linearity_mean"] = float(np.mean([r.get("linearity", 0.0) for r in obj["_candidate_records"]]))
    obj["planarity_mean"] = float(np.mean([r.get("planarity", 0.0) for r in obj["_candidate_records"]]))
    obj["source_types"].update([cand["source_type"]])
    obj["source_clusters"].update([str(cand["source_cluster"])])


def match_object(obj: dict, cand: dict, args: argparse.Namespace) -> tuple[bool, dict]:
    centroid_dist = float(np.linalg.norm(np.array(obj["centroid"]) - np.array(cand["centroid"])))
    bd = bbox_distance(obj, cand)
    color_dist = float(np.linalg.norm(np.array(obj["mean_visual_color"]) - np.array(cand["mean_visual_color"])))
    near = centroid_dist <= args.centroid_distance or bd <= args.bbox_distance
    color_ok = color_dist <= args.color_distance
    same_source_cluster = str(cand["source_cluster"]) in obj["source_clusters"]
    merge = near and color_ok and (same_source_cluster or centroid_dist <= args.cross_source_centroid_distance)
    return merge, {
        "centroid_distance": centroid_dist,
        "bbox_distance": bd,
        "color_distance": color_dist,
        "same_source_cluster": same_source_cluster,
        "reason": "fine_candidate_geometry_color" if merge else "no_match",
    }


def finalize_object(obj: dict) -> dict:
    return {
        "fine_object_id": obj["fine_object_id"],
        "candidate_ids": obj["candidate_ids"],
        "candidate_count": obj["candidate_count"],
        "point_count": obj["point_count"],
        "bbox_3d": obj["bbox_3d"],
        "centroid": obj["centroid"],
        "mean_visual_color": obj["mean_visual_color"],
        "linearity_mean": obj["linearity_mean"],
        "planarity_mean": obj["planarity_mean"],
        "source_types": dict(obj["source_types"]),
        "source_clusters": dict(obj["source_clusters"]),
        "status": "stable_fine_object" if obj["candidate_count"] > 1 else "single_fine_candidate",
    }


def fuse(candidates: list[dict], args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    objects: list[dict] = []
    decisions = []
    for cand in candidates:
        best_idx = None
        best_meta = None
        best_score = float("inf")
        for idx, obj in enumerate(objects):
            ok, meta = match_object(obj, cand, args)
            if ok:
                score = meta["bbox_distance"] + meta["color_distance"] / 255.0
                if score < best_score:
                    best_idx = idx
                    best_meta = meta
                    best_score = score
        if best_idx is None:
            obj = create_object(len(objects) + 1, cand)
            objects.append(obj)
            decisions.append({"candidate_id": cand["candidate_id"], "fine_object_id": obj["fine_object_id"], "action": "new_object"})
        else:
            obj = objects[best_idx]
            merge_object(obj, cand)
            decisions.append({"candidate_id": cand["candidate_id"], "fine_object_id": obj["fine_object_id"], "action": "merge", **(best_meta or {})})
    return objects, decisions


def write_object_ply(path: Path, accepted_ply: Path, objects: list[dict]) -> int:
    props, _, data = read_ascii_ply(accepted_ply)
    idx = {name: i for i, name in enumerate(props)}
    object_by_candidate = {}
    for number, obj in enumerate(objects, start=1):
        for candidate_id in obj["candidate_ids"]:
            object_by_candidate[int(candidate_id)] = number
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(data)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property int fine_object\n")
        f.write("property int accepted_candidate\n")
        f.write("property uchar semantic\n")
        f.write("property uchar visual_red\nproperty uchar visual_green\nproperty uchar visual_blue\n")
        f.write("end_header\n")
        for row in data:
            candidate_id = int(row[idx["accepted_candidate"]])
            object_number = object_by_candidate[candidate_id]
            color = object_color(object_number)
            f.write(
                f"{row[idx['x']]:.6f} {row[idx['y']]:.6f} {row[idx['z']]:.6f} "
                f"{color[0]} {color[1]} {color[2]} {object_number} {candidate_id} {int(row[idx['semantic']])} "
                f"{int(row[idx['visual_red']])} {int(row[idx['visual_green']])} {int(row[idx['visual_blue']])}\n"
            )
    return int(len(data))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-review-json", type=Path)
    parser.add_argument("--accepted-report-json", type=Path)
    parser.add_argument("--strict-filtered-ply", type=Path, required=True)
    parser.add_argument("--output-objects-jsonl", type=Path, required=True)
    parser.add_argument("--output-decisions-jsonl", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--centroid-distance", type=float, default=0.9)
    parser.add_argument("--cross-source-centroid-distance", type=float, default=0.45)
    parser.add_argument("--bbox-distance", type=float, default=0.25)
    parser.add_argument("--color-distance", type=float, default=45.0)
    args = parser.parse_args()

    if args.accepted_report_json:
        candidates = load_candidates_from_accepted_report(args.accepted_report_json)
    elif args.strict_review_json:
        candidates = load_kept_candidates(args.strict_review_json)
    else:
        raise ValueError("one of --accepted-report-json or --strict-review-json is required")
    objects, decisions = fuse(candidates, args)
    final_objects = [finalize_object(obj) for obj in objects]
    args.output_objects_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_objects_jsonl.open("w", encoding="utf-8") as f:
        for obj in final_objects:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    args.output_decisions_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_decisions_jsonl.open("w", encoding="utf-8") as f:
        for row in decisions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    vertex_count = write_object_ply(args.output_ply, args.strict_filtered_ply, objects)
    status_counts = Counter(obj["status"] for obj in final_objects)
    report = {
        "strict_review_json": str(args.strict_review_json) if args.strict_review_json else "",
        "accepted_report_json": str(args.accepted_report_json) if args.accepted_report_json else "",
        "strict_filtered_ply": str(args.strict_filtered_ply),
        "output_objects_jsonl": str(args.output_objects_jsonl),
        "output_decisions_jsonl": str(args.output_decisions_jsonl),
        "output_ply": str(args.output_ply),
        "params": {
            "centroid_distance": args.centroid_distance,
            "cross_source_centroid_distance": args.cross_source_centroid_distance,
            "bbox_distance": args.bbox_distance,
            "color_distance": args.color_distance,
        },
        "candidate_count": int(len(candidates)),
        "fine_object_count": int(len(final_objects)),
        "point_count": vertex_count,
        "merge_count": int(sum(1 for row in decisions if row["action"] == "merge")),
        "status_counts": dict(status_counts),
        "top_objects": sorted(final_objects, key=lambda row: row["point_count"], reverse=True)[:100],
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["candidate_count", "fine_object_count", "point_count", "merge_count", "status_counts"]}, indent=2))


if __name__ == "__main__":
    main()
