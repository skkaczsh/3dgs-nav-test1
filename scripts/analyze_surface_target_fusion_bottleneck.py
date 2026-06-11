#!/usr/bin/env python3
"""Analyze surface target generation and object fusion bottlenecks.

This report is intentionally based on the target/object artifacts rather than
semantic point PLY colors. It answers whether surface failures come from
missing/fragmented targets, fusion fragmentation, or cross-label conflicts.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


SURFACE_LABELS = {"floor", "wall", "building"}
FINE_LABELS = {"equipment", "railing"}


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_target_files(targets_dir: Path):
    yield from sorted(targets_dir.glob("targets_frame_*.jsonl"))


def quantiles(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    vals = sorted(float(v) for v in values)

    def q(p: float) -> float:
        idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * p))))
        return vals[idx]

    return {
        "count": len(vals),
        "min": vals[0],
        "p25": q(0.25),
        "median": float(median(vals)),
        "p75": q(0.75),
        "p90": q(0.90),
        "p95": q(0.95),
        "max": vals[-1],
    }


def bbox_extent(target: dict) -> list[float]:
    bbox = target.get("bbox_3d", {})
    lo = bbox.get("min", [0, 0, 0])
    hi = bbox.get("max", [0, 0, 0])
    return [float(hi[i]) - float(lo[i]) for i in range(3)]


def normal_bucket(target: dict) -> str:
    normal = target.get("pca", {}).get("normal", [0.0, 0.0, 1.0])
    z = abs(float(normal[2])) if len(normal) >= 3 else 1.0
    if z >= 0.82:
        return "horizontal_like"
    if z <= 0.35:
        return "vertical_like"
    return "slanted_or_uncertain"


def object_vote_summary(obj: dict) -> tuple[str, float, str]:
    votes = obj.get("label_vote_weights") or obj.get("label_votes") or {}
    if not votes:
        return obj.get("semantic_label", "unknown"), 0.0, ""
    total = sum(float(v) for v in votes.values())
    items = sorted(votes.items(), key=lambda kv: float(kv[1]), reverse=True)
    winner, score = items[0]
    pair = "/".join(label for label, _ in items[:3])
    return str(winner), float(score) / max(total, 1e-9), pair


def add_counter(dst: dict, key: str, value: int | float) -> None:
    dst[key] = dst.get(key, 0) + value


def analyze_targets(targets_dir: Path) -> tuple[dict, dict]:
    by_label: dict[str, dict] = defaultdict(lambda: {
        "target_count": 0,
        "point_count": 0,
        "frames": set(),
        "cams": Counter(),
        "normal_buckets": Counter(),
        "cluster_sizes": [],
        "bbox_extents": [[], [], []],
        "components_per_mask": Counter(),
    })
    by_frame = defaultdict(lambda: Counter(target_count=0, point_count=0))
    mask_components = Counter()
    target_to_meta = {}

    file_count = 0
    for file_path in iter_target_files(targets_dir):
        file_count += 1
        for t in load_jsonl(file_path):
            label = str(t.get("label", "unknown"))
            size = int(t.get("cluster_size", len(t.get("point_indices", []))))
            frame = int(t.get("frame_id", -1))
            cam = int(t.get("cam_id", -1))
            mask_key = f"{frame:06d}:cam{cam}:m{int(t.get('mask_id', -1)):04d}:{label}"
            mask_components[mask_key] += 1
            item = by_label[label]
            item["target_count"] += 1
            item["point_count"] += size
            item["frames"].add(frame)
            item["cams"][cam] += 1
            item["normal_buckets"][normal_bucket(t)] += 1
            item["cluster_sizes"].append(size)
            extent = bbox_extent(t)
            for i in range(3):
                item["bbox_extents"][i].append(extent[i])
            by_frame[frame]["target_count"] += 1
            by_frame[frame]["point_count"] += size
            by_frame[frame][f"label:{label}:targets"] += 1
            by_frame[frame][f"label:{label}:points"] += size
            target_to_meta[str(t.get("target_id"))] = {
                "label": label,
                "cluster_size": size,
                "frame_id": frame,
                "mask_key": mask_key,
                "normal_bucket": normal_bucket(t),
            }

    for mask_key, count in mask_components.items():
        label = mask_key.rsplit(":", 1)[-1]
        by_label[label]["components_per_mask"][count] += 1

    target_summary = {}
    for label, item in by_label.items():
        ext = item["bbox_extents"]
        target_summary[label] = {
            "target_count": item["target_count"],
            "point_count": item["point_count"],
            "frame_count": len(item["frames"]),
            "cams": dict(item["cams"]),
            "normal_buckets": dict(item["normal_buckets"]),
            "cluster_size_quantiles": quantiles(item["cluster_sizes"]),
            "bbox_extent_quantiles_xyz": [quantiles(axis) for axis in ext],
            "components_per_mask_histogram": dict(sorted(item["components_per_mask"].items())),
        }

    top_fragmented_masks = [
        {"mask_key": key, "component_count": count}
        for key, count in mask_components.most_common(30)
        if count > 1
    ]
    frame_rows = [
        {"frame_id": frame, **dict(counter)}
        for frame, counter in sorted(by_frame.items())
    ]
    top_surface_sparse_frames = sorted(
        frame_rows,
        key=lambda r: (
            r.get("label:wall:points", 0) + r.get("label:building:points", 0),
            -r.get("label:floor:points", 0),
        ),
    )[:30]

    return {
        "target_file_count": file_count,
        "by_label": target_summary,
        "top_fragmented_masks": top_fragmented_masks,
        "top_surface_sparse_frames": top_surface_sparse_frames,
    }, target_to_meta


def analyze_objects(objects_path: Path, target_to_meta: dict) -> dict:
    by_label: dict[str, dict] = defaultdict(lambda: {
        "object_count": 0,
        "point_count": 0,
        "target_count": 0,
        "status_counts": Counter(),
        "purities": [],
        "vote_pairs": Counter(),
        "single_target_objects": 0,
        "single_target_points": 0,
    })
    ambiguous_pairs = Counter()
    target_assignment = {}
    largest_by_label = defaultdict(list)
    target_label_to_object_label = Counter()

    for obj in load_jsonl(objects_path):
        label = str(obj.get("semantic_label", "unknown"))
        points = int(obj.get("point_count", len(obj.get("merged_point_indices", []))))
        targets = list(obj.get("targets", []))
        target_count = int(obj.get("target_count", len(targets)))
        status = str(obj.get("status", "unknown"))
        winner, purity, vote_pair = object_vote_summary(obj)
        item = by_label[label]
        item["object_count"] += 1
        item["point_count"] += points
        item["target_count"] += target_count
        item["status_counts"][status] += 1
        item["purities"].append(purity)
        if vote_pair:
            item["vote_pairs"][vote_pair] += 1
        if target_count <= 1:
            item["single_target_objects"] += 1
            item["single_target_points"] += points
        if label == "ambiguous":
            ambiguous_pairs[vote_pair] += 1
        largest_by_label[label].append({
            "object_id": obj.get("object_id"),
            "semantic_label": label,
            "status": status,
            "point_count": points,
            "target_count": target_count,
            "vote_pair": vote_pair,
            "vote_purity": purity,
        })
        for tid in targets:
            meta = target_to_meta.get(str(tid), {})
            tlabel = str(meta.get("label", "unknown"))
            target_assignment[str(tid)] = label
            target_label_to_object_label[(tlabel, label)] += 1

    object_summary = {}
    for label, item in by_label.items():
        object_summary[label] = {
            "object_count": item["object_count"],
            "point_count": item["point_count"],
            "target_count": item["target_count"],
            "status_counts": dict(item["status_counts"]),
            "single_target_objects": item["single_target_objects"],
            "single_target_points": item["single_target_points"],
            "vote_purity_quantiles": quantiles(item["purities"]),
            "top_vote_pairs": [
                {"labels": pair, "object_count": count}
                for pair, count in item["vote_pairs"].most_common(15)
            ],
        }

    largest = {
        label: sorted(rows, key=lambda r: r["point_count"], reverse=True)[:20]
        for label, rows in largest_by_label.items()
    }
    target_flow = [
        {"target_label": src, "object_label": dst, "target_count": count}
        for (src, dst), count in target_label_to_object_label.most_common()
    ]
    return {
        "by_label": object_summary,
        "target_label_to_object_label": target_flow,
        "ambiguous_vote_pairs": [
            {"labels": pair, "object_count": count}
            for pair, count in ambiguous_pairs.most_common(30)
        ],
        "largest_objects_by_label": largest,
        "target_assignment_count": len(target_assignment),
    }


def analyze_decisions(decisions_path: Path | None, target_to_meta: dict) -> dict:
    if not decisions_path or not decisions_path.exists():
        return {"available": False}
    by_action = Counter()
    by_target_label_action = Counter()
    merge_reason_by_label = Counter()
    merge_metrics = defaultdict(list)
    new_object_by_label = Counter()
    for d in load_jsonl(decisions_path):
        action = str(d.get("action", "unknown"))
        tid = str(d.get("target_id", ""))
        label = str(target_to_meta.get(tid, {}).get("label", "unknown"))
        by_action[action] += 1
        by_target_label_action[(label, action)] += 1
        if action == "merge":
            reason = str(d.get("reason", "unknown"))
            merge_reason_by_label[(label, reason)] += 1
            for key in ("centroid_distance", "bbox_distance", "color_distance", "normal_angle"):
                if key in d:
                    merge_metrics[(label, key)].append(float(d[key]))
        elif action == "new_object":
            new_object_by_label[label] += 1

    metric_summary = defaultdict(dict)
    for (label, metric), values in merge_metrics.items():
        metric_summary[label][metric] = quantiles(values)
    return {
        "available": True,
        "action_counts": dict(by_action),
        "by_target_label_action": [
            {"target_label": label, "action": action, "count": count}
            for (label, action), count in by_target_label_action.most_common()
        ],
        "new_object_by_label": dict(new_object_by_label),
        "merge_reason_by_label": [
            {"target_label": label, "reason": reason, "count": count}
            for (label, reason), count in merge_reason_by_label.most_common(50)
        ],
        "merge_metric_quantiles_by_label": dict(metric_summary),
    }


def build_findings(targets: dict, objects: dict, decisions: dict) -> list[str]:
    findings = []
    t_by = targets.get("by_label", {})
    o_by = objects.get("by_label", {})
    for label in ("floor", "building", "wall"):
        t = t_by.get(label, {})
        o = o_by.get(label, {})
        targets_n = int(t.get("target_count", 0))
        target_points = int(t.get("point_count", 0))
        objects_n = int(o.get("object_count", 0))
        single_points = int(o.get("single_target_points", 0))
        point_count = int(o.get("point_count", 0))
        single_ratio = single_points / max(point_count, 1)
        findings.append(
            f"{label}: targets={targets_n}, target_points={target_points}, "
            f"objects={objects_n}, object_points={point_count}, "
            f"single_target_point_ratio={single_ratio:.3f}"
        )
    if decisions.get("available"):
        new_by_label = decisions.get("new_object_by_label", {})
        for label in ("floor", "building", "wall", "equipment", "railing"):
            t_total = sum(
                row["count"]
                for row in decisions.get("by_target_label_action", [])
                if row["target_label"] == label
            )
            new_n = int(new_by_label.get(label, 0))
            findings.append(f"{label}: new_object_targets={new_n}/{t_total} ({new_n / max(t_total, 1):.3f})")
    amb = objects.get("ambiguous_vote_pairs", [])[:5]
    if amb:
        findings.append("top ambiguous conflicts: " + "; ".join(f"{r['labels']}={r['object_count']}" for r in amb))
    findings.append(
        "Note: fusion_decisions records successful merges and new_object actions only; "
        "candidate rejection reasons are not available in current artifacts."
    )
    return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets-dir", type=Path, required=True)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--fusion-decisions-jsonl", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_summary, target_to_meta = analyze_targets(args.targets_dir)
    object_summary = analyze_objects(args.objects_jsonl, target_to_meta)
    decision_summary = analyze_decisions(args.fusion_decisions_jsonl, target_to_meta)
    report = {
        "inputs": {
            "targets_dir": str(args.targets_dir),
            "objects_jsonl": str(args.objects_jsonl),
            "fusion_decisions_jsonl": str(args.fusion_decisions_jsonl) if args.fusion_decisions_jsonl else None,
        },
        "summary": {
            "target_count": len(target_to_meta),
            "object_target_assignment_count": object_summary.get("target_assignment_count", 0),
        },
        "targets": target_summary,
        "objects": object_summary,
        "fusion_decisions": decision_summary,
        "findings": build_findings(target_summary, object_summary, decision_summary),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), **report["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
