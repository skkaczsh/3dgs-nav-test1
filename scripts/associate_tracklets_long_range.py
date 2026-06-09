#!/usr/bin/env python3
"""Long-range association for fine-object tracklets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def load_tracklets(path: Path) -> list[dict]:
    files = sorted(path.glob("tracklets.jsonl")) if path.is_dir() else [path]
    rows: list[dict] = []
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return sorted(rows, key=lambda r: (int(r.get("frame_min", r.get("frame_id", 0))), str(r.get("tracklet_id", r.get("target_id", "")))))


def bbox_distance(a: dict, b: dict) -> float:
    amin = np.array(a["bbox_3d"]["min"], dtype=np.float64)
    amax = np.array(a["bbox_3d"]["max"], dtype=np.float64)
    bmin = np.array(b["bbox_3d"]["min"], dtype=np.float64)
    bmax = np.array(b["bbox_3d"]["max"], dtype=np.float64)
    gap = np.maximum(0.0, np.maximum(bmin - amax, amin - bmax))
    return float(np.linalg.norm(gap))


def dominant_vote(row: dict, key: str) -> str:
    votes = {str(k): int(v) for k, v in row.get(key, {}).items()}
    if not votes:
        return ""
    return max(votes.items(), key=lambda kv: kv[1])[0]


def create_object(index: int, tracklet: dict) -> dict:
    count = int(tracklet.get("point_count", tracklet.get("cluster_size", 1)))
    return {
        "long_object_id": f"long_obj_{index:04d}",
        "object_number": int(index),
        "label": tracklet["label"],
        "label_votes": Counter({tracklet["label"]: count}),
        "tracklet_ids": [tracklet["tracklet_id"]],
        "tracklet_count": 1,
        "target_count": int(tracklet.get("target_count", 1)),
        "point_count": count,
        "frames": list(tracklet.get("frames", [tracklet.get("frame_id", 0)])),
        "frame_min": int(tracklet.get("frame_min", tracklet.get("frame_id", 0))),
        "frame_max": int(tracklet.get("frame_max", tracklet.get("frame_id", 0))),
        "bbox_3d": tracklet["bbox_3d"],
        "centroid": tracklet["centroid"],
        "mean_color": tracklet["mean_color"],
        "color_sum": (np.array(tracklet["mean_color"], dtype=np.float64) * max(count, 1)).tolist(),
        "accepted_candidate_votes": Counter({str(k): int(v) for k, v in tracklet.get("accepted_candidate_votes", {}).items()}),
        "source_cluster_votes": Counter({str(k): int(v) for k, v in tracklet.get("source_cluster_votes", {}).items()}),
        "_tracklet_records": [tracklet],
    }


def merge_object(obj: dict, tracklet: dict) -> None:
    old_count = max(int(obj["point_count"]), 1)
    new_count = max(int(tracklet.get("point_count", tracklet.get("cluster_size", 1))), 1)
    total = old_count + new_count
    obj["tracklet_ids"].append(tracklet["tracklet_id"])
    obj["tracklet_count"] = len(obj["tracklet_ids"])
    obj["target_count"] = int(obj["target_count"] + int(tracklet.get("target_count", 1)))
    obj["point_count"] = int(total)
    obj["label_votes"].update({tracklet["label"]: new_count})
    obj["frames"] = sorted(set(obj["frames"] + list(tracklet.get("frames", []))))
    obj["frame_min"] = int(min(obj["frame_min"], int(tracklet.get("frame_min", tracklet.get("frame_id", 0)))))
    obj["frame_max"] = int(max(obj["frame_max"], int(tracklet.get("frame_max", tracklet.get("frame_id", 0)))))
    bmin = np.minimum(np.array(obj["bbox_3d"]["min"], dtype=np.float64), np.array(tracklet["bbox_3d"]["min"], dtype=np.float64))
    bmax = np.maximum(np.array(obj["bbox_3d"]["max"], dtype=np.float64), np.array(tracklet["bbox_3d"]["max"], dtype=np.float64))
    obj["bbox_3d"] = {"min": [float(x) for x in bmin], "max": [float(x) for x in bmax]}
    obj["centroid"] = [float(x) for x in ((np.array(obj["centroid"]) * old_count + np.array(tracklet["centroid"]) * new_count) / total)]
    color_sum = np.array(obj["color_sum"], dtype=np.float64) + np.array(tracklet["mean_color"], dtype=np.float64) * new_count
    obj["color_sum"] = [float(x) for x in color_sum]
    obj["mean_color"] = [float(x) for x in color_sum / total]
    obj["accepted_candidate_votes"].update({str(k): int(v) for k, v in tracklet.get("accepted_candidate_votes", {}).items()})
    obj["source_cluster_votes"].update({str(k): int(v) for k, v in tracklet.get("source_cluster_votes", {}).items()})
    obj["_tracklet_records"].append(tracklet)


def frame_gap(obj: dict, tracklet: dict) -> int:
    tmin = int(tracklet.get("frame_min", tracklet.get("frame_id", 0)))
    tmax = int(tracklet.get("frame_max", tracklet.get("frame_id", 0)))
    if tmin > obj["frame_max"]:
        return int(tmin - obj["frame_max"])
    if obj["frame_min"] > tmax:
        return int(obj["frame_min"] - tmax)
    return 0


def match_object(obj: dict, tracklet: dict, args: argparse.Namespace) -> tuple[bool, dict]:
    centroid_dist = float(np.linalg.norm(np.array(obj["centroid"]) - np.array(tracklet["centroid"])))
    bd = bbox_distance(obj, tracklet)
    color_dist = float(np.linalg.norm(np.array(obj["mean_color"]) - np.array(tracklet["mean_color"])))
    obj_candidate = dominant_vote(obj, "accepted_candidate_votes")
    trk_candidate = dominant_vote(tracklet, "accepted_candidate_votes")
    obj_source = dominant_vote(obj, "source_cluster_votes")
    trk_source = dominant_vote(tracklet, "source_cluster_votes")
    same_label = obj["label"] == tracklet["label"]
    same_candidate = bool(obj_candidate and trk_candidate and obj_candidate == trk_candidate)
    same_source = bool(obj_source and trk_source and obj_source == trk_source)
    gap = frame_gap(obj, tracklet)
    candidate_ok = (
        same_label
        and same_candidate
        and centroid_dist <= args.same_candidate_centroid_distance
        and bd <= args.same_candidate_bbox_distance
        and color_dist <= args.same_candidate_color_distance
    )
    source_ok = (
        same_label
        and same_source
        and gap <= args.source_frame_gap
        and centroid_dist <= args.source_centroid_distance
        and bd <= args.source_bbox_distance
        and color_dist <= args.source_color_distance
    )
    strict_cross_ok = (
        same_label
        and gap <= args.cross_frame_gap
        and centroid_dist <= args.cross_centroid_distance
        and bd <= args.cross_bbox_distance
        and color_dist <= args.cross_color_distance
    )
    ok = candidate_ok or source_ok or strict_cross_ok
    if candidate_ok:
        reason = "same_accepted_candidate"
    elif source_ok:
        reason = "same_source_cluster"
    elif strict_cross_ok:
        reason = "strict_cross_source"
    else:
        reason = "no_match"
    return ok, {
        "reason": reason,
        "frame_gap": int(gap),
        "centroid_distance": centroid_dist,
        "bbox_distance": bd,
        "color_distance": color_dist,
        "same_candidate": bool(same_candidate),
        "same_source": bool(same_source),
        "same_label": bool(same_label),
        "object_candidate": obj_candidate,
        "tracklet_candidate": trk_candidate,
    }


def finalize_object(obj: dict) -> dict:
    label, label_votes = obj["label_votes"].most_common(1)[0]
    total_votes = sum(obj["label_votes"].values())
    accepted_total = sum(obj["accepted_candidate_votes"].values())
    source_total = sum(obj["source_cluster_votes"].values())
    return {
        "long_object_id": obj["long_object_id"],
        "object_number": int(obj["object_number"]),
        "label": label,
        "label_vote_ratio": float(label_votes / max(total_votes, 1)),
        "tracklet_ids": obj["tracklet_ids"],
        "tracklet_count": int(obj["tracklet_count"]),
        "target_count": int(obj["target_count"]),
        "point_count": int(obj["point_count"]),
        "frame_min": int(obj["frame_min"]),
        "frame_max": int(obj["frame_max"]),
        "frame_count": int(len(obj["frames"])),
        "bbox_3d": obj["bbox_3d"],
        "centroid": obj["centroid"],
        "mean_color": obj["mean_color"],
        "accepted_candidate_votes": dict(obj["accepted_candidate_votes"]),
        "dominant_accepted_candidate": dominant_vote(obj, "accepted_candidate_votes"),
        "dominant_accepted_candidate_ratio": float(max(obj["accepted_candidate_votes"].values(), default=0) / max(accepted_total, 1)),
        "source_cluster_votes": dict(obj["source_cluster_votes"]),
        "dominant_source_cluster": dominant_vote(obj, "source_cluster_votes"),
        "dominant_source_cluster_ratio": float(max(obj["source_cluster_votes"].values(), default=0) / max(source_total, 1)),
        "status": "stable_long_object" if int(obj["tracklet_count"]) > 1 else "single_tracklet_object",
    }


def associate(tracklets: list[dict], args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    objects: list[dict] = []
    decisions: list[dict] = []
    for tracklet in tracklets:
        best_idx = None
        best_meta = None
        best_score = float("inf")
        for idx, obj in enumerate(objects):
            ok, meta = match_object(obj, tracklet, args)
            if not ok:
                continue
            reason_bonus = {"same_accepted_candidate": -2.0, "same_source_cluster": -1.0, "strict_cross_source": 0.0}.get(meta["reason"], 0.0)
            score = reason_bonus + meta["bbox_distance"] + meta["centroid_distance"] + meta["color_distance"] / 255.0
            if score < best_score:
                best_idx = idx
                best_meta = meta
                best_score = score
        if best_idx is None:
            obj = create_object(len(objects) + 1, tracklet)
            objects.append(obj)
            decisions.append({"tracklet_id": tracklet["tracklet_id"], "long_object_id": obj["long_object_id"], "action": "new_object"})
        else:
            obj = objects[best_idx]
            merge_object(obj, tracklet)
            decisions.append({"tracklet_id": tracklet["tracklet_id"], "long_object_id": obj["long_object_id"], "action": "merge", **(best_meta or {})})
    return objects, decisions


def object_color(index: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(index * 127 + 419)
    return tuple(int(x) for x in rng.integers(70, 245, 3))


def write_centroid_ply(path: Path, objects: list[dict]) -> int:
    points = []
    for obj in objects:
        color = object_color(int(obj["object_number"]))
        for trk in obj["_tracklet_records"]:
            points.append((trk["centroid"], color, int(obj["object_number"]), int(trk.get("frame_min", trk.get("frame_id", 0)))))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property int object\nproperty int frame\n")
        f.write("end_header\n")
        for centroid, color, object_number, frame in points:
            f.write(f"{centroid[0]:.6f} {centroid[1]:.6f} {centroid[2]:.6f} {color[0]} {color[1]} {color[2]} {object_number} {frame}\n")
    return len(points)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracklets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--same-candidate-centroid-distance", type=float, default=1.5)
    parser.add_argument("--same-candidate-bbox-distance", type=float, default=0.5)
    parser.add_argument("--same-candidate-color-distance", type=float, default=90.0)
    parser.add_argument("--source-frame-gap", type=int, default=240)
    parser.add_argument("--source-centroid-distance", type=float, default=0.8)
    parser.add_argument("--source-bbox-distance", type=float, default=0.25)
    parser.add_argument("--source-color-distance", type=float, default=60.0)
    parser.add_argument("--cross-frame-gap", type=int, default=80)
    parser.add_argument("--cross-centroid-distance", type=float, default=0.35)
    parser.add_argument("--cross-bbox-distance", type=float, default=0.08)
    parser.add_argument("--cross-color-distance", type=float, default=35.0)
    args = parser.parse_args()

    tracklets = load_tracklets(args.tracklets)
    objects, decisions = associate(tracklets, args)
    final_objects = [finalize_object(obj) for obj in objects]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "long_objects.jsonl").open("w", encoding="utf-8") as f:
        for obj in final_objects:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    with (args.output_dir / "long_association_decisions.jsonl").open("w", encoding="utf-8") as f:
        for row in decisions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    centroid_points = write_centroid_ply(args.output_dir / "long_object_tracklet_centroids.ply", objects)
    reason_counts = Counter(row.get("reason", "new_object") for row in decisions)
    status_counts = Counter(obj["status"] for obj in final_objects)
    params = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    report = {
        "tracklets": int(len(tracklets)),
        "objects": int(len(final_objects)),
        "merge_ratio": float(1.0 - len(final_objects) / max(len(tracklets), 1)),
        "status_counts": dict(status_counts),
        "reason_counts": dict(reason_counts),
        "centroid_ply": str(args.output_dir / "long_object_tracklet_centroids.ply"),
        "centroid_points": int(centroid_points),
        "params": params,
        "top_objects": sorted(final_objects, key=lambda row: row["point_count"], reverse=True)[:100],
    }
    (args.output_dir / "long_association_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["tracklets", "objects", "merge_ratio", "status_counts", "reason_counts"]}, indent=2))


if __name__ == "__main__":
    main()
