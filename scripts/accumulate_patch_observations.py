#!/usr/bin/env python3
"""Accumulate visual/scene evidence onto geometry-first GeoPatch records.

This stage intentionally summarizes evidence without producing final labels.
Downstream classification may use these votes, but geometry gates remain the
source of object boundaries.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apply_scene_prior_to_objects import vote_scene_context
from qa_viewer_candidate import LABELS


PRIORITY_LABELS = {
    0: "residual",
    1: "ground",
    2: "wall",
    3: "grass",
    4: "car",
    5: "railing",
    6: "sky",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def decode_votes(votes: dict[str, int], names: dict[int, str]) -> dict[str, int]:
    out: Counter[str] = Counter()
    for key, value in votes.items():
        try:
            label = names.get(int(float(key)), str(key))
        except (TypeError, ValueError):
            label = str(key)
        out[label] += int(value)
    return dict(out)


def weighted_winner(votes: dict[str, int | float]) -> tuple[str, float]:
    if not votes:
        return "unknown", 0.0
    counter = Counter({str(k): float(v) for k, v in votes.items()})
    label, value = counter.most_common(1)[0]
    return label, float(value / max(sum(counter.values()), 1.0))


def patch_frames(patch: dict[str, Any], max_expand: int = 60) -> list[int]:
    frame_votes = (patch.get("source_votes") or {}).get("frame") or {}
    frames = []
    for key in frame_votes.keys():
        try:
            frames.append(int(float(key)))
        except (TypeError, ValueError):
            pass
    if frames:
        return sorted(set(frames))
    span = patch.get("source_frame_span") if isinstance(patch.get("source_frame_span"), dict) else {}
    if "min" in span and "max" in span:
        lo, hi = int(span["min"]), int(span["max"])
        if hi - lo <= max_expand:
            return list(range(lo, hi + 1))
        return [lo, int(round((lo + hi) / 2)), hi]
    return []


def build_patch_evidence(patch: dict[str, Any], scene_segments: list[dict[str, Any]]) -> dict[str, Any]:
    source_votes = patch.get("source_votes") if isinstance(patch.get("source_votes"), dict) else {}
    semantic_votes = decode_votes(source_votes.get("semantic") or {}, LABELS)
    priority_votes = decode_votes(source_votes.get("priority") or {}, PRIORITY_LABELS)
    object_votes = source_votes.get("object") or {}
    frame_votes = source_votes.get("frame") or {}
    camera_votes = source_votes.get("camera") or {}

    semantic_label, semantic_ratio = weighted_winner(semantic_votes)
    priority_label, priority_ratio = weighted_winner(priority_votes)
    structural_label, structural_ratio = weighted_winner(patch.get("structural_region_votes") or {})
    scene_prior = vote_scene_context(
        {"semantic_label": semantic_label, "frames": patch_frames(patch)},
        scene_segments,
    ) if scene_segments else {"frames_with_scene_prior": 0, "scene_prior_status": "not_provided"}

    return {
        "semantic_votes": semantic_votes,
        "dominant_semantic_label": semantic_label,
        "dominant_semantic_ratio": semantic_ratio,
        "priority_votes": priority_votes,
        "dominant_priority_label": priority_label,
        "dominant_priority_ratio": priority_ratio,
        "source_object_votes": object_votes,
        "frame_votes": frame_votes,
        "camera_votes": camera_votes,
        "dominant_structural_region": structural_label,
        "dominant_structural_region_ratio": structural_ratio,
        "scene_prior": scene_prior,
        "evidence_point_count": int(patch.get("point_count") or 0),
    }


def enrich_patches(
    patches: list[dict[str, Any]],
    scene_prior: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scene_segments = scene_prior.get("segments", []) if isinstance(scene_prior, dict) else []
    out = []
    geometry_counts = Counter()
    semantic_counts = Counter()
    priority_counts = Counter()
    scene_counts = Counter()
    for patch in patches:
        row = dict(patch)
        evidence = build_patch_evidence(row, scene_segments)
        row["evidence"] = evidence
        out.append(row)
        geometry_counts[str(row.get("geometry_type") or "unknown")] += 1
        semantic_counts[evidence["dominant_semantic_label"]] += 1
        priority_counts[evidence["dominant_priority_label"]] += 1
        scene_counts[str(evidence["scene_prior"].get("dominant_scene_area_type") or "unknown")] += 1
    report = {
        "schema": "geo-patch-observations/v1",
        "patch_count": len(out),
        "scene_prior_schema": scene_prior.get("schema") if isinstance(scene_prior, dict) else None,
        "geometry_type_counts": dict(geometry_counts),
        "dominant_semantic_counts": dict(semantic_counts),
        "dominant_priority_counts": dict(priority_counts),
        "dominant_scene_area_counts": dict(scene_counts),
    }
    return out, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geo-patches", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--scene-prior", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene_prior = json.loads(args.scene_prior.read_text(encoding="utf-8")) if args.scene_prior else None
    rows, report = enrich_patches(read_jsonl(args.geo_patches), scene_prior)
    report.update({"geo_patches": str(args.geo_patches), "output_jsonl": str(args.output_jsonl)})
    write_jsonl(args.output_jsonl, rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
