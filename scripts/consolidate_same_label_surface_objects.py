#!/usr/bin/env python3
"""Consolidate same-label surface Objects without cross-label absorption.

This is a second-stage object-level pass for strict surface fusion outputs.
It only merges objects with the same semantic label, using 3D proximity,
normal consistency, plane compatibility, and mean RGB similarity.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SURFACE_LABELS = {"floor", "wall", "building"}


class DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def vec3(value: Any, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> list[float]:
    if isinstance(value, list) and len(value) >= 3:
        return [float(value[0]), float(value[1]), float(value[2])]
    return [float(x) for x in default]


def bbox_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    amin = vec3(a.get("bbox_3d", {}).get("min"))
    amax = vec3(a.get("bbox_3d", {}).get("max"))
    bmin = vec3(b.get("bbox_3d", {}).get("min"))
    bmax = vec3(b.get("bbox_3d", {}).get("max"))
    gap = [max(0.0, max(bmin[i] - amax[i], amin[i] - bmax[i])) for i in range(3)]
    return math.sqrt(sum(x * x for x in gap))


def dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    if n <= 1e-9:
        return [0.0, 0.0, 1.0]
    return [x / n for x in v]


def angle_degrees(a: list[float], b: list[float]) -> float:
    av = normalize(a)
    bv = normalize(b)
    dot = abs(sum(av[i] * bv[i] for i in range(3)))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def plane_distance(point: list[float], plane_centroid: list[float], plane_normal: list[float]) -> float:
    n = normalize(plane_normal)
    return abs(sum((point[i] - plane_centroid[i]) * n[i] for i in range(3)))


def compatible(a: dict[str, Any], b: dict[str, Any], args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    ca = vec3(a.get("centroid"))
    cb = vec3(b.get("centroid"))
    na = vec3(a.get("normal"), (0.0, 0.0, 1.0))
    nb = vec3(b.get("normal"), (0.0, 0.0, 1.0))
    ma = vec3(a.get("mean_color"))
    mb = vec3(b.get("mean_color"))
    centroid_distance = dist(ca, cb)
    bbox_gap = bbox_distance(a, b)
    normal_angle = angle_degrees(na, nb)
    color_distance = dist(ma, mb)
    plane_ab = plane_distance(cb, ca, na)
    plane_ba = plane_distance(ca, cb, nb)
    max_plane_distance = max(plane_ab, plane_ba)
    near = bbox_gap <= args.max_bbox_gap or centroid_distance <= args.max_centroid_distance
    ok = (
        near
        and normal_angle <= args.max_normal_angle
        and color_distance <= args.max_color_distance
        and max_plane_distance <= args.max_plane_distance
    )
    return ok, {
        "centroid_distance": centroid_distance,
        "bbox_gap": bbox_gap,
        "normal_angle": normal_angle,
        "color_distance": color_distance,
        "plane_distance": max_plane_distance,
    }


def merge_group(group_id: str, members: list[dict[str, Any]]) -> dict[str, Any]:
    label = members[0].get("semantic_label", "unknown")
    point_total = sum(int(o.get("point_count", 0)) for o in members)
    weight_total = max(point_total, 1)
    centroid = [0.0, 0.0, 0.0]
    color = [0.0, 0.0, 0.0]
    normal = [0.0, 0.0, 0.0]
    bbox_min = [float("inf"), float("inf"), float("inf")]
    bbox_max = [float("-inf"), float("-inf"), float("-inf")]
    targets = []
    frames = set()
    label_votes: Counter[str] = Counter()
    statuses = Counter()
    for obj in members:
        w = max(int(obj.get("point_count", 0)), 1)
        c = vec3(obj.get("centroid"))
        rgb = vec3(obj.get("mean_color"))
        n = normalize(vec3(obj.get("normal"), (0.0, 0.0, 1.0)))
        for i in range(3):
            centroid[i] += c[i] * w
            color[i] += rgb[i] * w
            normal[i] += n[i] * w
            bbox_min[i] = min(bbox_min[i], vec3(obj.get("bbox_3d", {}).get("min"))[i])
            bbox_max[i] = max(bbox_max[i], vec3(obj.get("bbox_3d", {}).get("max"))[i])
        targets.extend(obj.get("targets", []))
        frames.update(int(x) for x in obj.get("frames", []))
        statuses[str(obj.get("status", "unknown"))] += 1
        for k, v in (obj.get("label_votes") or {}).items():
            label_votes[str(k)] += int(v)
    centroid = [x / weight_total for x in centroid]
    color = [x / weight_total for x in color]
    normal = normalize(normal)
    return {
        "consolidated_object_id": group_id,
        "semantic_label": label,
        "status": "surface_consolidated" if len(members) > 1 else members[0].get("status", "single_target"),
        "source_object_count": len(members),
        "source_objects": [o.get("object_id") for o in members],
        "target_count": sum(int(o.get("target_count", len(o.get("targets", [])))) for o in members),
        "point_count": point_total,
        "targets": targets,
        "frames": sorted(frames),
        "bbox_3d": {"min": bbox_min, "max": bbox_max},
        "centroid": centroid,
        "mean_color": color,
        "normal": normal,
        "label_votes": dict(label_votes),
        "source_status_counts": dict(statuses),
    }


def consolidate(objects: list[dict[str, Any]], labels: set[str], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    mappings = []
    report_by_label = {}
    by_label: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    passthrough = []
    for idx, obj in enumerate(objects):
        label = str(obj.get("semantic_label", "unknown"))
        if label in labels and int(obj.get("point_count", 0)) >= args.min_points:
            by_label[label].append((idx, obj))
        else:
            passthrough.append(obj)

    for label, rows in sorted(by_label.items()):
        dsu = DSU(len(rows))
        edge_count = 0
        merge_examples = []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                ok, meta = compatible(rows[i][1], rows[j][1], args)
                if ok:
                    if dsu.union(i, j):
                        edge_count += 1
                        if len(merge_examples) < 25:
                            merge_examples.append({
                                "a": rows[i][1].get("object_id"),
                                "b": rows[j][1].get("object_id"),
                                **{k: round(v, 4) for k, v in meta.items()},
                            })
        groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for i, (_, obj) in enumerate(rows):
            groups[dsu.find(i)].append(obj)
        consolidated = []
        for group_number, members in enumerate(sorted(groups.values(), key=lambda g: -sum(int(o.get("point_count", 0)) for o in g)), start=1):
            group_id = f"surf_{label}_{group_number:05d}"
            row = merge_group(group_id, members)
            consolidated.append(row)
            for source in members:
                mappings.append({
                    "source_object_id": source.get("object_id"),
                    "consolidated_object_id": group_id,
                    "semantic_label": label,
                })
        out_rows.extend(consolidated)
        source_points = sum(int(obj.get("point_count", 0)) for _, obj in rows)
        report_by_label[label] = {
            "source_objects": len(rows),
            "consolidated_objects": len(consolidated),
            "merged_object_reduction": len(rows) - len(consolidated),
            "source_points": source_points,
            "edge_count": edge_count,
            "largest_group_source_objects": max((len(g) for g in groups.values()), default=0),
            "largest_group_points": max((sum(int(o.get("point_count", 0)) for o in g) for g in groups.values()), default=0),
            "merge_examples": merge_examples,
        }

    for obj in passthrough:
        out_rows.append({
            "consolidated_object_id": f"pass_{obj.get('object_id', '')}",
            "semantic_label": obj.get("semantic_label", "unknown"),
            "status": obj.get("status", "unknown"),
            "source_object_count": 1,
            "source_objects": [obj.get("object_id")],
            **{k: obj.get(k) for k in ["target_count", "point_count", "targets", "frames", "bbox_3d", "centroid", "mean_color", "normal", "label_votes"]},
        })
        mappings.append({
            "source_object_id": obj.get("object_id"),
            "consolidated_object_id": f"pass_{obj.get('object_id', '')}",
            "semantic_label": obj.get("semantic_label", "unknown"),
        })
    out_rows.sort(key=lambda r: (str(r.get("semantic_label")), -int(r.get("point_count", 0))))
    report = {
        "input_objects": len(objects),
        "output_objects": len(out_rows),
        "merged_object_reduction": len(objects) - len(out_rows),
        "labels": report_by_label,
        "params": {
            "labels": sorted(labels),
            "min_points": args.min_points,
            "max_bbox_gap": args.max_bbox_gap,
            "max_centroid_distance": args.max_centroid_distance,
            "max_normal_angle": args.max_normal_angle,
            "max_plane_distance": args.max_plane_distance,
            "max_color_distance": args.max_color_distance,
        },
    }
    return out_rows, {"report": report, "mappings": mappings}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-mapping", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", default=sorted(DEFAULT_SURFACE_LABELS))
    parser.add_argument("--min-points", type=int, default=100)
    parser.add_argument("--max-bbox-gap", type=float, default=0.35)
    parser.add_argument("--max-centroid-distance", type=float, default=1.0)
    parser.add_argument("--max-normal-angle", type=float, default=15.0)
    parser.add_argument("--max-plane-distance", type=float, default=0.20)
    parser.add_argument("--max-color-distance", type=float, default=65.0)
    args = parser.parse_args()

    objects = load_jsonl(args.objects_jsonl)
    rows, result = consolidate(objects, set(args.labels), args)
    write_jsonl(args.output_jsonl, rows)
    write_jsonl(args.output_mapping, result["mappings"])
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(result["report"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
