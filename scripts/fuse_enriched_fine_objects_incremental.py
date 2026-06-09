#!/usr/bin/env python3
"""Fuse enriched accepted fine-object candidates in scan order."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


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
        return props, header_lines, np.empty((0, len(props)), dtype=np.float64)
    data = np.loadtxt(path, skiprows=header_lines, dtype=np.float64)
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


def pca_shape(points: np.ndarray) -> dict:
    if len(points) < 3:
        return {
            "eigenvalues": [0.0, 0.0, 0.0],
            "linearity": 0.0,
            "planarity": 0.0,
            "scattering": 0.0,
            "normal": [0.0, 0.0, 1.0],
        }
    centered = points - points.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 0.0)
    vecs = vecs[:, order]
    denom = max(float(vals[0]), 1e-12)
    return {
        "eigenvalues": [float(x) for x in vals.tolist()],
        "linearity": float((vals[0] - vals[1]) / denom),
        "planarity": float((vals[1] - vals[2]) / denom),
        "scattering": float(vals[2] / denom),
        "normal": [float(x) for x in vecs[:, -1].tolist()],
    }


def object_color(object_number: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(object_number * 103 + 2027)
    return tuple(int(x) for x in rng.integers(65, 245, 3))


def build_candidates(props: list[str], data: np.ndarray) -> list[dict]:
    idx = {name: i for i, name in enumerate(props)}
    required = {
        "x",
        "y",
        "z",
        "semantic",
        "accepted_candidate",
        "source_type",
        "source_cluster",
        "subcluster",
        "visual_red",
        "visual_green",
        "visual_blue",
        "frame",
        "camera",
        "mask",
    }
    if not required.issubset(idx):
        raise ValueError(f"missing required enriched PLY properties. required={required} available={props}")

    candidates: list[dict] = []
    candidate_ids = sorted(set(int(x) for x in data[:, idx["accepted_candidate"]].tolist()))
    for candidate_id in candidate_ids:
        rows = data[data[:, idx["accepted_candidate"]].astype(np.int64) == candidate_id]
        xyz = rows[:, [idx["x"], idx["y"], idx["z"]]]
        bmin = xyz.min(axis=0)
        bmax = xyz.max(axis=0)
        centroid = xyz.mean(axis=0)
        visual = rows[:, [idx["visual_red"], idx["visual_green"], idx["visual_blue"]]]
        frames = rows[:, idx["frame"]].astype(np.int64)
        frames = frames[frames >= 0]
        cameras = rows[:, idx["camera"]].astype(np.int64)
        masks = rows[:, idx["mask"]].astype(np.int64)
        source_clusters = rows[:, idx["source_cluster"]].astype(np.int64)
        shape = pca_shape(xyz)
        candidates.append(
            {
                "candidate_id": int(candidate_id),
                "semantic": int(Counter(rows[:, idx["semantic"]].astype(np.int64).tolist()).most_common(1)[0][0]),
                "source_type": int(Counter(rows[:, idx["source_type"]].astype(np.int64).tolist()).most_common(1)[0][0]),
                "source_cluster": int(Counter(source_clusters.tolist()).most_common(1)[0][0]),
                "subcluster": int(Counter(rows[:, idx["subcluster"]].astype(np.int64).tolist()).most_common(1)[0][0]),
                "points": int(len(rows)),
                "bbox_3d": {"min": [float(x) for x in bmin], "max": [float(x) for x in bmax]},
                "centroid": [float(x) for x in centroid],
                "mean_visual_color": [float(x) for x in visual.mean(axis=0)],
                "frame_min": int(frames.min()) if len(frames) else -1,
                "frame_max": int(frames.max()) if len(frames) else -1,
                "frame_count": int(len(set(int(x) for x in frames.tolist()))),
                "camera_counts": dict(Counter(int(x) for x in cameras.tolist() if int(x) >= 0)),
                "mask_count": int(len(set(int(x) for x in masks.tolist() if int(x) >= 0))),
                "linearity": shape["linearity"],
                "planarity": shape["planarity"],
                "scattering": shape["scattering"],
                "normal": shape["normal"],
            }
        )
    return sorted(candidates, key=lambda row: (row["frame_min"] if row["frame_min"] >= 0 else 10**9, row["candidate_id"]))


def create_object(index: int, cand: dict) -> dict:
    return {
        "fine_object_id": f"inc_fine_obj_{index:04d}",
        "object_number": index,
        "candidate_ids": [int(cand["candidate_id"])],
        "candidate_count": 1,
        "point_count": int(cand["points"]),
        "semantic_votes": Counter({int(cand["semantic"]): int(cand["points"])}),
        "bbox_3d": cand["bbox_3d"],
        "centroid": cand["centroid"],
        "mean_visual_color": cand["mean_visual_color"],
        "color_sum": (np.array(cand["mean_visual_color"], dtype=np.float64) * max(int(cand["points"]), 1)).tolist(),
        "frame_min": int(cand["frame_min"]),
        "frame_max": int(cand["frame_max"]),
        "frame_count": int(cand["frame_count"]),
        "source_clusters": Counter([str(cand["source_cluster"])]),
        "source_types": Counter([str(cand["source_type"])]),
        "_candidate_records": [cand],
    }


def merge_object(obj: dict, cand: dict) -> None:
    old_count = max(int(obj["point_count"]), 1)
    new_count = max(int(cand["points"]), 1)
    total = old_count + new_count
    obj["candidate_ids"].append(int(cand["candidate_id"]))
    obj["candidate_count"] = len(obj["candidate_ids"])
    obj["point_count"] = int(total)
    obj["semantic_votes"].update({int(cand["semantic"]): new_count})
    bmin = np.minimum(np.array(obj["bbox_3d"]["min"], dtype=np.float64), np.array(cand["bbox_3d"]["min"], dtype=np.float64))
    bmax = np.maximum(np.array(obj["bbox_3d"]["max"], dtype=np.float64), np.array(cand["bbox_3d"]["max"], dtype=np.float64))
    obj["bbox_3d"] = {"min": [float(x) for x in bmin], "max": [float(x) for x in bmax]}
    obj["centroid"] = [float(x) for x in ((np.array(obj["centroid"]) * old_count + np.array(cand["centroid"]) * new_count) / total)]
    color_sum = np.array(obj["color_sum"], dtype=np.float64) + np.array(cand["mean_visual_color"], dtype=np.float64) * new_count
    obj["color_sum"] = [float(x) for x in color_sum]
    obj["mean_visual_color"] = [float(x) for x in color_sum / total]
    frames = [int(r["frame_min"]) for r in obj["_candidate_records"] + [cand] if int(r["frame_min"]) >= 0]
    frame_maxes = [int(r["frame_max"]) for r in obj["_candidate_records"] + [cand] if int(r["frame_max"]) >= 0]
    obj["frame_min"] = min(frames) if frames else -1
    obj["frame_max"] = max(frame_maxes) if frame_maxes else -1
    frame_set = set()
    for r in obj["_candidate_records"] + [cand]:
        if int(r["frame_min"]) >= 0 and int(r["frame_max"]) >= 0:
            frame_set.update(range(int(r["frame_min"]), int(r["frame_max"]) + 1))
    obj["frame_count"] = len(frame_set)
    obj["source_clusters"].update([str(cand["source_cluster"])])
    obj["source_types"].update([str(cand["source_type"])])
    obj["_candidate_records"].append(cand)


def frame_gap(obj: dict, cand: dict) -> int:
    if obj["frame_max"] < 0 or cand["frame_min"] < 0:
        return 10**9
    if cand["frame_min"] > obj["frame_max"]:
        return int(cand["frame_min"] - obj["frame_max"])
    if obj["frame_min"] > cand["frame_max"]:
        return int(obj["frame_min"] - cand["frame_max"])
    return 0


def match_object(obj: dict, cand: dict, args: argparse.Namespace) -> tuple[bool, dict]:
    gap = frame_gap(obj, cand)
    temporal_ok = gap <= args.active_frame_window
    centroid_dist = float(np.linalg.norm(np.array(obj["centroid"]) - np.array(cand["centroid"])))
    bd = bbox_distance(obj, cand)
    color_dist = float(np.linalg.norm(np.array(obj["mean_visual_color"]) - np.array(cand["mean_visual_color"])))
    same_source_cluster = str(cand["source_cluster"]) in obj["source_clusters"]
    same_semantic = int(cand["semantic"]) in obj["semantic_votes"]
    near = centroid_dist <= args.centroid_distance or bd <= args.bbox_distance
    cross_source_ok = same_source_cluster or centroid_dist <= args.cross_source_centroid_distance
    merge = temporal_ok and same_semantic and near and cross_source_ok and color_dist <= args.color_distance
    reason = "incremental_geometry_color" if merge else "no_match"
    if not temporal_ok:
        reason = "frame_window"
    elif not same_semantic:
        reason = "semantic_mismatch"
    elif not near:
        reason = "spatial_distance"
    elif not cross_source_ok:
        reason = "cross_source_distance"
    elif color_dist > args.color_distance:
        reason = "color_distance"
    return merge, {
        "frame_gap": int(gap),
        "centroid_distance": centroid_dist,
        "bbox_distance": bd,
        "color_distance": color_dist,
        "same_source_cluster": bool(same_source_cluster),
        "same_semantic": bool(same_semantic),
        "reason": reason,
    }


def finalize_object(obj: dict) -> dict:
    vote_total = sum(obj["semantic_votes"].values())
    semantic, votes = obj["semantic_votes"].most_common(1)[0]
    records = obj["_candidate_records"]
    return {
        "fine_object_id": obj["fine_object_id"],
        "object_number": int(obj["object_number"]),
        "semantic": int(semantic),
        "semantic_vote_ratio": float(votes / max(vote_total, 1)),
        "semantic_votes": {str(k): int(v) for k, v in obj["semantic_votes"].items()},
        "candidate_ids": obj["candidate_ids"],
        "candidate_count": int(obj["candidate_count"]),
        "point_count": int(obj["point_count"]),
        "bbox_3d": obj["bbox_3d"],
        "centroid": obj["centroid"],
        "mean_visual_color": obj["mean_visual_color"],
        "frame_min": int(obj["frame_min"]),
        "frame_max": int(obj["frame_max"]),
        "frame_span": int(max(0, obj["frame_max"] - obj["frame_min"] + 1)) if obj["frame_min"] >= 0 else 0,
        "frame_count": int(obj["frame_count"]),
        "source_clusters": dict(obj["source_clusters"]),
        "source_types": dict(obj["source_types"]),
        "linearity_mean": float(np.mean([r["linearity"] for r in records])),
        "planarity_mean": float(np.mean([r["planarity"] for r in records])),
        "status": "stable_incremental_fine_object" if obj["candidate_count"] > 1 else "single_incremental_fine_candidate",
    }


def fuse(candidates: list[dict], args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    objects: list[dict] = []
    decisions: list[dict] = []
    for cand in candidates:
        best_idx = None
        best_meta = None
        best_score = float("inf")
        for idx, obj in enumerate(objects):
            if cand["frame_min"] >= 0 and obj["frame_max"] >= 0 and cand["frame_min"] - obj["frame_max"] > args.active_frame_window:
                continue
            ok, meta = match_object(obj, cand, args)
            if ok:
                score = meta["bbox_distance"] + meta["centroid_distance"] + meta["color_distance"] / 255.0 + meta["frame_gap"] / 1000.0
                if score < best_score:
                    best_idx = idx
                    best_meta = meta
                    best_score = score
        if best_idx is None:
            obj = create_object(len(objects) + 1, cand)
            objects.append(obj)
            decisions.append(
                {
                    "candidate_id": int(cand["candidate_id"]),
                    "fine_object_id": obj["fine_object_id"],
                    "action": "new_object",
                    "frame_min": int(cand["frame_min"]),
                    "frame_max": int(cand["frame_max"]),
                }
            )
        else:
            obj = objects[best_idx]
            merge_object(obj, cand)
            decisions.append(
                {
                    "candidate_id": int(cand["candidate_id"]),
                    "fine_object_id": obj["fine_object_id"],
                    "action": "merge",
                    "frame_min": int(cand["frame_min"]),
                    "frame_max": int(cand["frame_max"]),
                    **(best_meta or {}),
                }
            )
    return objects, decisions


def build_zones(final_objects: list[dict], zone_size: int) -> list[dict]:
    by_zone: dict[int, list[dict]] = {}
    for obj in final_objects:
        frame = int(obj["frame_min"])
        zone = frame // zone_size if frame >= 0 else -1
        by_zone.setdefault(zone, []).append(obj)
    zones = []
    for zone, objs in sorted(by_zone.items()):
        mins = np.array([o["bbox_3d"]["min"] for o in objs], dtype=np.float64)
        maxes = np.array([o["bbox_3d"]["max"] for o in objs], dtype=np.float64)
        zones.append(
            {
                "zone_id": f"zone_{zone:03d}" if zone >= 0 else "zone_unknown",
                "frame_min": int(zone * zone_size) if zone >= 0 else -1,
                "frame_max": int(zone * zone_size + zone_size - 1) if zone >= 0 else -1,
                "object_count": int(len(objs)),
                "point_count": int(sum(o["point_count"] for o in objs)),
                "object_ids": [o["fine_object_id"] for o in objs],
                "bbox_3d": {"min": [float(x) for x in mins.min(axis=0)], "max": [float(x) for x in maxes.max(axis=0)]},
            }
        )
    return zones


def write_object_ply(path: Path, props: list[str], data: np.ndarray, objects: list[dict]) -> int:
    idx = {name: i for i, name in enumerate(props)}
    object_by_candidate: dict[int, int] = {}
    for obj in objects:
        for candidate_id in obj["candidate_ids"]:
            object_by_candidate[int(candidate_id)] = int(obj["object_number"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(data)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property uchar semantic\n")
        f.write("property int accepted_candidate\n")
        f.write("property int fine_object\n")
        f.write("property int incremental_fine_object\n")
        f.write("property uchar source_type\n")
        f.write("property int source_cluster\n")
        f.write("property int subcluster\n")
        f.write("property uchar visual_red\nproperty uchar visual_green\nproperty uchar visual_blue\n")
        f.write("property int frame\nproperty int camera\nproperty int mask\nproperty int point_index\n")
        f.write("property uchar trace_status\n")
        f.write("end_header\n")
        for row in data:
            candidate_id = int(row[idx["accepted_candidate"]])
            object_number = object_by_candidate.get(candidate_id, -1)
            color = object_color(object_number) if object_number > 0 else (40, 40, 40)
            f.write(
                f"{row[idx['x']]:.6f} {row[idx['y']]:.6f} {row[idx['z']]:.6f} "
                f"{color[0]} {color[1]} {color[2]} "
                f"{int(row[idx['semantic']])} {candidate_id} {int(row[idx['fine_object']])} {object_number} "
                f"{int(row[idx['source_type']])} {int(row[idx['source_cluster']])} {int(row[idx['subcluster']])} "
                f"{int(row[idx['visual_red']])} {int(row[idx['visual_green']])} {int(row[idx['visual_blue']])} "
                f"{int(row[idx['frame']])} {int(row[idx['camera']])} {int(row[idx['mask']])} "
                f"{int(row[idx['point_index']])} {int(row[idx['trace_status']])}\n"
            )
    return int(len(data))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enriched-ply", type=Path, required=True)
    parser.add_argument("--output-objects-jsonl", type=Path, required=True)
    parser.add_argument("--output-decisions-jsonl", type=Path, required=True)
    parser.add_argument("--output-zones-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--centroid-distance", type=float, default=0.45)
    parser.add_argument("--cross-source-centroid-distance", type=float, default=0.25)
    parser.add_argument("--bbox-distance", type=float, default=0.05)
    parser.add_argument("--color-distance", type=float, default=30.0)
    parser.add_argument("--active-frame-window", type=int, default=120)
    parser.add_argument("--zone-size", type=int, default=100)
    args = parser.parse_args()

    props, _, data = read_ascii_ply(args.enriched_ply)
    candidates = build_candidates(props, data)
    objects, decisions = fuse(candidates, args)
    final_objects = [finalize_object(obj) for obj in objects]
    zones = build_zones(final_objects, args.zone_size)

    args.output_objects_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_objects_jsonl.open("w", encoding="utf-8") as f:
        for obj in final_objects:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    args.output_decisions_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_decisions_jsonl.open("w", encoding="utf-8") as f:
        for row in decisions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.output_zones_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_zones_json.write_text(json.dumps(zones, ensure_ascii=False, indent=2), encoding="utf-8")
    point_count = write_object_ply(args.output_ply, props, data, objects)

    status_counts = Counter(obj["status"] for obj in final_objects)
    report = {
        "enriched_ply": str(args.enriched_ply),
        "output_objects_jsonl": str(args.output_objects_jsonl),
        "output_decisions_jsonl": str(args.output_decisions_jsonl),
        "output_zones_json": str(args.output_zones_json),
        "output_ply": str(args.output_ply),
        "params": {
            "centroid_distance": args.centroid_distance,
            "cross_source_centroid_distance": args.cross_source_centroid_distance,
            "bbox_distance": args.bbox_distance,
            "color_distance": args.color_distance,
            "active_frame_window": args.active_frame_window,
            "zone_size": args.zone_size,
        },
        "candidate_count": int(len(candidates)),
        "fine_object_count": int(len(final_objects)),
        "point_count": int(point_count),
        "merge_count": int(sum(1 for row in decisions if row["action"] == "merge")),
        "zone_count": int(len(zones)),
        "status_counts": dict(status_counts),
        "frame_span": {
            "min": int(min([c["frame_min"] for c in candidates if c["frame_min"] >= 0], default=-1)),
            "max": int(max([c["frame_max"] for c in candidates if c["frame_max"] >= 0], default=-1)),
        },
        "top_objects": sorted(final_objects, key=lambda row: row["point_count"], reverse=True)[:100],
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: report[k] for k in ["candidate_count", "fine_object_count", "point_count", "merge_count", "zone_count", "status_counts"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
