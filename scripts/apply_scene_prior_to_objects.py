#!/usr/bin/env python3
"""Attach route-level scene priors to viewer object JSONL records.

This stage does not relabel points. It adds explicit, auditable context fields
that later relabel/split stages can use as weights instead of hard-coded scene
assumptions.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


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


def normalize_label(label: str) -> str:
    aliases = {
        "floor": "ground",
        "road": "ground",
        "roof": "ground",
        "glass": "glass_fence",
        "building": "wall",
        "building_part": "wall",
        "tree_or_shrub": "tree",
        "bush": "tree",
    }
    return aliases.get(label, label)


def segment_for_frame(segments: list[dict[str, Any]], frame_id: int) -> dict[str, Any] | None:
    for segment in segments:
        if int(segment.get("start_frame", -1)) <= frame_id <= int(segment.get("end_frame", -1)):
            return segment
    if not segments:
        return None
    return min(
        segments,
        key=lambda s: min(abs(frame_id - int(s.get("start_frame", 0))), abs(frame_id - int(s.get("end_frame", 0)))),
    )


def object_frames(obj: dict[str, Any]) -> list[int]:
    frames = obj.get("frames")
    if isinstance(frames, list):
        out = []
        for value in frames:
            try:
                out.append(int(value))
            except (TypeError, ValueError):
                pass
        return sorted(set(out))
    frame = obj.get("frame_id")
    if frame is not None:
        try:
            return [int(frame)]
        except (TypeError, ValueError):
            return []
    return []


def vote_scene_context(obj: dict[str, Any], segments: list[dict[str, Any]]) -> dict[str, Any]:
    frames = object_frames(obj)
    if not frames:
        return {"frames_with_scene_prior": 0, "scene_prior_status": "no_object_frames"}
    area_votes: Counter[str] = Counter()
    name_votes: Counter[str] = Counter()
    expected_votes: Counter[str] = Counter()
    unlikely_votes: Counter[str] = Counter()
    ground_votes: Counter[str] = Counter()
    segment_votes: Counter[str] = Counter()
    confidence_sum = 0.0
    matched = 0
    for frame_id in frames:
        segment = segment_for_frame(segments, frame_id)
        if not segment:
            continue
        weight = float(segment.get("confidence") or 1.0)
        matched += 1
        confidence_sum += weight
        segment_votes[str(segment.get("segment_id") or "")] += weight
        area_votes[str(segment.get("area_type") or "unknown")] += weight
        name_votes[str(segment.get("area_name_zh") or "unknown")] += weight
        for label in segment.get("expected_labels", []) if isinstance(segment.get("expected_labels"), list) else []:
            expected_votes[normalize_label(str(label))] += weight
        for label in segment.get("unlikely_labels", []) if isinstance(segment.get("unlikely_labels"), list) else []:
            unlikely_votes[normalize_label(str(label))] += weight
        for subtype in segment.get("ground_subtypes", []) if isinstance(segment.get("ground_subtypes"), list) else []:
            ground_votes[str(subtype)] += weight
    total = max(confidence_sum, 1e-9)
    current = normalize_label(str(obj.get("semantic_label") or "unknown"))
    return {
        "frames_with_scene_prior": matched,
        "scene_prior_status": "matched" if matched else "unmatched",
        "scene_area_type_votes": dict(area_votes),
        "scene_area_name_votes": dict(name_votes),
        "scene_segment_votes": dict(segment_votes),
        "scene_expected_label_weights": dict(expected_votes),
        "scene_unlikely_label_weights": dict(unlikely_votes),
        "scene_ground_subtype_weights": dict(ground_votes),
        "scene_expected_label_score": float(expected_votes.get(current, 0.0) / total),
        "scene_unlikely_label_score": float(unlikely_votes.get(current, 0.0) / total),
        "dominant_scene_area_type": area_votes.most_common(1)[0][0] if area_votes else "unknown",
        "dominant_scene_area_name": name_votes.most_common(1)[0][0] if name_votes else "unknown",
        "dominant_scene_ground_subtype": ground_votes.most_common(1)[0][0] if ground_votes else "",
        "scene_prior_confidence_mean": float(confidence_sum / matched) if matched else 0.0,
    }


def enrich_objects(objects: list[dict[str, Any]], scene_prior: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    segments = scene_prior.get("segments") if isinstance(scene_prior.get("segments"), list) else []
    enriched: list[dict[str, Any]] = []
    area_counts: Counter[str] = Counter()
    label_expected = Counter()
    label_unlikely = Counter()
    ground_subtypes = Counter()
    for obj in objects:
        out = dict(obj)
        context = vote_scene_context(out, segments)
        out["scene_prior"] = context
        enriched.append(out)
        area_counts[context.get("dominant_scene_area_type", "unknown")] += 1
        label = str(out.get("semantic_label") or "unknown")
        if float(context.get("scene_expected_label_score") or 0.0) > 0:
            label_expected[label] += 1
        if float(context.get("scene_unlikely_label_score") or 0.0) > 0:
            label_unlikely[label] += 1
        subtype = context.get("dominant_scene_ground_subtype") or ""
        if subtype:
            ground_subtypes[subtype] += 1
    report = {
        "schema": "scene-prior-object-enrichment/v1",
        "scene_prior_schema": scene_prior.get("schema"),
        "object_count": len(objects),
        "segment_count": len(segments),
        "area_type_object_counts": dict(area_counts),
        "labels_with_expected_scene_support": dict(label_expected),
        "labels_with_unlikely_scene_context": dict(label_unlikely),
        "ground_subtype_object_counts": dict(ground_subtypes),
    }
    return enriched, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--scene-prior", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    objects = read_jsonl(args.objects_jsonl)
    scene_prior = json.loads(args.scene_prior.read_text(encoding="utf-8"))
    enriched, report = enrich_objects(objects, scene_prior)
    report.update({
        "objects_jsonl": str(args.objects_jsonl),
        "scene_prior": str(args.scene_prior),
        "output_jsonl": str(args.output_jsonl),
    })
    write_jsonl(args.output_jsonl, enriched)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
