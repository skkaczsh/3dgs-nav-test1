#!/usr/bin/env python3
"""Create an augmented Objects JSONL with selected surface-seed candidates.

This is a diagnostic input generator. It does not modify the original fusion
outputs. Candidate Objects selected by analyze_surface_seed_candidates.py are
copied with `status=stable` and `semantic_label=<dominant_surface_label>` so
downstream residual coverage sweeps can estimate the value of promoting them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_candidate_map(path: Path, min_points: int, min_surface_vote_ratio: float) -> dict[str, dict]:
    report = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for row in report.get("top_candidates", []):
        if int(row.get("point_count", 0)) < min_points:
            continue
        if float(row.get("surface_vote_ratio", 0.0)) < min_surface_vote_ratio:
            continue
        label = row.get("dominant_surface_label")
        if not label:
            continue
        out[str(row["object_id"])] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--min-points", type=int, default=500)
    parser.add_argument("--min-surface-vote-ratio", type=float, default=0.8)
    args = parser.parse_args()

    candidates = load_candidate_map(args.candidates, args.min_points, args.min_surface_vote_ratio)
    promoted = []
    total = 0
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.objects_jsonl.open("r", encoding="utf-8") as src, args.output_jsonl.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            total += 1
            obj = json.loads(line)
            candidate = candidates.get(str(obj.get("object_id")))
            if candidate:
                original = {
                    "status": obj.get("status"),
                    "semantic_label": obj.get("semantic_label"),
                    "dominant_label": obj.get("dominant_label"),
                    "dominant_label_ratio": obj.get("dominant_label_ratio"),
                }
                obj["status"] = "stable"
                obj["semantic_label"] = candidate["dominant_surface_label"]
                obj["dominant_label"] = candidate["dominant_surface_label"]
                obj["dominant_label_ratio"] = candidate.get("dominant_surface_ratio", obj.get("dominant_label_ratio", 0.0))
                obj["surface_seed_promotion"] = {
                    "source": str(args.candidates),
                    "original": original,
                    "surface_votes": candidate.get("surface_votes", {}),
                    "surface_vote_ratio": candidate.get("surface_vote_ratio"),
                    "reason": candidate.get("reason"),
                }
                promoted.append(
                    {
                        "object_id": obj.get("object_id"),
                        "point_count": obj.get("point_count"),
                        "target_count": obj.get("target_count"),
                        "promoted_label": obj["semantic_label"],
                        "original": original,
                    }
                )
            dst.write(json.dumps(obj, ensure_ascii=False) + "\n")

    report = {
        "objects_jsonl": str(args.objects_jsonl),
        "candidates": str(args.candidates),
        "output_jsonl": str(args.output_jsonl),
        "total_objects": total,
        "candidate_count": len(candidates),
        "promoted_count": len(promoted),
        "promoted_points": int(sum(int(row.get("point_count", 0)) for row in promoted)),
        "params": {
            "min_points": args.min_points,
            "min_surface_vote_ratio": args.min_surface_vote_ratio,
        },
        "promoted": promoted,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["total_objects", "promoted_count", "promoted_points"]}, indent=2))


if __name__ == "__main__":
    main()
