#!/usr/bin/env python3
"""Build short-window Tracklet records from frame-level Target JSONL files."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def load_targets(inputs: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in inputs:
        files = sorted(path.glob("targets_frame_*.jsonl")) if path.is_dir() else [path]
        for file_path in files:
            if file_path.name == "targets_all.jsonl":
                continue
            with file_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
    return sorted(rows, key=lambda r: (int(r.get("frame_id", 0)), str(r.get("target_id", ""))))


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


def create_tracklet(index: int, target: dict) -> dict:
    point_count = int(target.get("cluster_size", 1))
    accepted_votes = Counter({str(k): int(v) for k, v in target.get("accepted_candidate_votes", {}).items()})
    source_votes = Counter({str(k): int(v) for k, v in target.get("source_cluster_votes", {}).items()})
    subcluster_votes = Counter({str(k): int(v) for k, v in target.get("subcluster_votes", {}).items()})
    return {
        "tracklet_id": f"trk_{index:06d}",
        "target_id": f"trk_{index:06d}",
        "label": target["label"],
        "label_id": int(target.get("label_id", 0)),
        "parent_class": target.get("parent_class", "other"),
        "frame_id": int(target["frame_id"]),
        "frames": [int(target["frame_id"])],
        "target_ids": [target["target_id"]],
        "target_count": 1,
        "cluster_size": point_count,
        "point_count": point_count,
        "bbox_3d": target["bbox_3d"],
        "centroid": target["centroid"],
        "mean_color": target["mean_color"],
        "color_sum": (np.array(target["mean_color"], dtype=np.float64) * max(point_count, 1)).tolist(),
        "point_indices": list(target.get("point_indices", [])),
        "pca": target.get("pca", {"normal": [0, 0, 1], "linearity": 0.0, "planarity": 0.0}),
        "accepted_candidate_votes": dict(accepted_votes),
        "source_cluster_votes": dict(source_votes),
        "subcluster_votes": dict(subcluster_votes),
        "_target_records": [target],
    }


def update_tracklet(tracklet: dict, target: dict) -> None:
    old_count = max(int(tracklet["point_count"]), 1)
    new_count = max(int(target.get("cluster_size", 1)), 1)
    total = old_count + new_count
    tracklet["target_ids"].append(target["target_id"])
    tracklet["target_count"] = len(tracklet["target_ids"])
    tracklet["frames"] = sorted(set(tracklet["frames"] + [int(target["frame_id"])]))
    tracklet["cluster_size"] = int(total)
    tracklet["point_count"] = int(total)
    tracklet["point_indices"] = sorted(set(tracklet.get("point_indices", []) + target.get("point_indices", [])))
    bmin = np.minimum(np.array(tracklet["bbox_3d"]["min"], dtype=np.float64), np.array(target["bbox_3d"]["min"], dtype=np.float64))
    bmax = np.maximum(np.array(tracklet["bbox_3d"]["max"], dtype=np.float64), np.array(target["bbox_3d"]["max"], dtype=np.float64))
    tracklet["bbox_3d"] = {"min": [float(x) for x in bmin], "max": [float(x) for x in bmax]}
    centroid = (np.array(tracklet["centroid"]) * old_count + np.array(target["centroid"]) * new_count) / total
    tracklet["centroid"] = [float(x) for x in centroid]
    color_sum = np.array(tracklet["color_sum"], dtype=np.float64) + np.array(target["mean_color"], dtype=np.float64) * new_count
    tracklet["color_sum"] = [float(x) for x in color_sum]
    tracklet["mean_color"] = [float(x) for x in color_sum / total]
    tracklet["_target_records"].append(target)
    for key in ("accepted_candidate_votes", "source_cluster_votes", "subcluster_votes"):
        votes = Counter({str(k): int(v) for k, v in tracklet.get(key, {}).items()})
        votes.update({str(k): int(v) for k, v in target.get(key, {}).items()})
        tracklet[key] = dict(votes)
    normals = [np.array(t.get("pca", {}).get("normal", [0.0, 0.0, 1.0]), dtype=np.float64) for t in tracklet["_target_records"]]
    normal = np.mean(normals, axis=0)
    norm = np.linalg.norm(normal)
    tracklet["pca"] = {
        "normal": [float(x) for x in (normal / norm if norm > 1e-9 else normal)],
        "linearity": float(np.mean([t.get("pca", {}).get("linearity", 0.0) for t in tracklet["_target_records"]])),
        "planarity": float(np.mean([t.get("pca", {}).get("planarity", 0.0) for t in tracklet["_target_records"]])),
    }


def frame_gap(tracklet: dict, target: dict) -> int:
    return max(0, int(target["frame_id"]) - max(tracklet["frames"]))


def match_tracklet(tracklet: dict, target: dict, args: argparse.Namespace) -> tuple[bool, dict]:
    gap = frame_gap(tracklet, target)
    centroid_dist = float(np.linalg.norm(np.array(tracklet["centroid"]) - np.array(target["centroid"])))
    bd = bbox_distance(tracklet, target)
    color_dist = float(np.linalg.norm(np.array(tracklet["mean_color"]) - np.array(target["mean_color"])))
    normal_angle = angle_degrees(tracklet.get("pca", {}).get("normal", [0, 0, 1]), target.get("pca", {}).get("normal", [0, 0, 1]))
    same_label = tracklet["label"] == target["label"]
    near = centroid_dist <= args.centroid_distance or bd <= args.bbox_distance
    ok = same_label and gap <= args.max_frame_gap and near and color_dist <= args.color_distance and normal_angle <= args.normal_angle
    return ok, {
        "frame_gap": int(gap),
        "centroid_distance": centroid_dist,
        "bbox_distance": bd,
        "color_distance": color_dist,
        "normal_angle": normal_angle,
        "same_label": bool(same_label),
    }


def finalize_tracklet(tracklet: dict) -> dict:
    out = {k: v for k, v in tracklet.items() if not k.startswith("_") and k != "color_sum"}
    out["frame_id"] = int(min(out["frames"]))
    out["frame_min"] = int(min(out["frames"]))
    out["frame_max"] = int(max(out["frames"]))
    out["frame_count"] = int(len(out["frames"]))
    out["status"] = "stable_tracklet" if int(out["target_count"]) > 1 else "single_target_tracklet"
    return out


def build_tracklets(targets: list[dict], args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    tracklets: list[dict] = []
    decisions: list[dict] = []
    for target in targets:
        best_idx = None
        best_meta = None
        best_score = float("inf")
        for idx, tracklet in enumerate(tracklets):
            if frame_gap(tracklet, target) > args.max_frame_gap:
                continue
            ok, meta = match_tracklet(tracklet, target, args)
            if ok:
                score = meta["bbox_distance"] + meta["centroid_distance"] + meta["color_distance"] / 255.0
                if score < best_score:
                    best_idx = idx
                    best_meta = meta
                    best_score = score
        if best_idx is None:
            trk = create_tracklet(len(tracklets) + 1, target)
            tracklets.append(trk)
            decisions.append({"target_id": target["target_id"], "tracklet_id": trk["tracklet_id"], "action": "new_tracklet"})
        else:
            trk = tracklets[best_idx]
            update_tracklet(trk, target)
            decisions.append({"target_id": target["target_id"], "tracklet_id": trk["tracklet_id"], "action": "merge", **(best_meta or {})})
    return tracklets, decisions


def write_outputs(output_dir: Path, tracklets: list[dict], decisions: list[dict], targets: list[dict], args: argparse.Namespace) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    finalized = [finalize_tracklet(t) for t in tracklets]
    with (output_dir / "tracklets.jsonl").open("w", encoding="utf-8") as f:
        for row in finalized:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "tracklets_as_targets.jsonl").open("w", encoding="utf-8") as f:
        for row in finalized:
            as_target = dict(row)
            as_target["target_id"] = row["tracklet_id"]
            f.write(json.dumps(as_target, ensure_ascii=False) + "\n")
    with (output_dir / "tracklet_decisions.jsonl").open("w", encoding="utf-8") as f:
        for row in decisions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    statuses = Counter(row["status"] for row in finalized)
    report = {
        "targets": int(len(targets)),
        "tracklets": int(len(finalized)),
        "merge_ratio": float(1.0 - len(finalized) / max(len(targets), 1)),
        "status_counts": dict(statuses),
        "params": {
            "max_frame_gap": args.max_frame_gap,
            "centroid_distance": args.centroid_distance,
            "bbox_distance": args.bbox_distance,
            "color_distance": args.color_distance,
            "normal_angle": args.normal_angle,
        },
        "target_count_stats": {
            "min": int(min([row["target_count"] for row in finalized], default=0)),
            "max": int(max([row["target_count"] for row in finalized], default=0)),
            "mean": float(np.mean([row["target_count"] for row in finalized])) if finalized else 0.0,
        },
        "point_count_stats": {
            "min": int(min([row["point_count"] for row in finalized], default=0)),
            "max": int(max([row["point_count"] for row in finalized], default=0)),
            "mean": float(np.mean([row["point_count"] for row in finalized])) if finalized else 0.0,
        },
        "top_tracklets": sorted(finalized, key=lambda row: row["point_count"], reverse=True)[:100],
    }
    (output_dir / "tracklet_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frame-gap", type=int, default=15)
    parser.add_argument("--centroid-distance", type=float, default=0.35)
    parser.add_argument("--bbox-distance", type=float, default=0.08)
    parser.add_argument("--color-distance", type=float, default=45.0)
    parser.add_argument("--normal-angle", type=float, default=180.0)
    args = parser.parse_args()

    targets = load_targets(args.targets)
    tracklets, decisions = build_tracklets(targets, args)
    report = write_outputs(args.output_dir, tracklets, decisions, targets, args)
    print(json.dumps({k: report[k] for k in ["targets", "tracklets", "merge_ratio", "status_counts"]}, indent=2))


if __name__ == "__main__":
    main()
