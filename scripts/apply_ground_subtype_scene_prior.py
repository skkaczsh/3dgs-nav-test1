#!/usr/bin/env python3
"""Create a ground-subtype preview using attached scene priors."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import qa_viewer_candidate
from scripts import rewrite_viewer_ply_semantics
from scripts.apply_scene_prior_to_objects import read_jsonl, write_jsonl
from scripts.export_frame_target_objects_for_viewer import LABEL_TO_SEMANTIC


GROUND_LABELS = {"ground", "floor", "grass"}
SUBTYPE_TO_LABEL = {
    "ordinary_ground": "ground",
    "grass": "grass",
    "stair": "stair",
    "indoor_floor": "indoor_floor",
    "roof": "roof",
}


def ground_subtype(row: dict[str, Any]) -> str:
    prior = row.get("scene_prior") if isinstance(row.get("scene_prior"), dict) else {}
    subtype = str(prior.get("dominant_scene_ground_subtype") or "")
    return subtype if subtype in SUBTYPE_TO_LABEL else ""


def relabel_ground_subtypes(rows: list[dict[str, Any]], min_scene_score: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    changed = []
    subtype_counts = Counter()
    for row in rows:
        out = dict(row)
        label = str(out.get("semantic_label") or "unknown")
        prior = out.get("scene_prior") if isinstance(out.get("scene_prior"), dict) else {}
        subtype = ground_subtype(out)
        score = float(prior.get("scene_prior_confidence_mean") or 0.0)
        if label in GROUND_LABELS and subtype and score >= min_scene_score:
            new_label = SUBTYPE_TO_LABEL[subtype]
            old_label = label
            if new_label in LABEL_TO_SEMANTIC and new_label != old_label:
                out["semantic_label_original"] = out.get("semantic_label_original") or old_label
                out["semantic_label"] = new_label
                out["semantic_id"] = LABEL_TO_SEMANTIC[new_label]
                out["status"] = f"{out.get('status', 'object')}_scene_ground_subtype"
                out["scene_ground_subtype_applied"] = subtype
                changed.append({
                    "object_id": out.get("viewer_object_id", out.get("object_id")),
                    "source_object_id": out.get("object_id"),
                    "old_label": old_label,
                    "new_label": new_label,
                    "subtype": subtype,
                    "point_count": out.get("point_count"),
                    "dominant_scene_area_type": prior.get("dominant_scene_area_type"),
                })
        subtype_counts[subtype or "none"] += 1
        output.append(out)
    return output, {
        "schema": "ground-subtype-scene-prior-preview/v1",
        "input_object_count": len(rows),
        "changed_count": len(changed),
        "subtype_counts": dict(subtype_counts),
        "label_counts_after": dict(Counter(str(row.get("semantic_label") or "unknown") for row in output)),
        "changed": changed,
    }


def run_qa(output_dir: Path, ply_name: str, objects_name: str, top_n: int) -> dict[str, Any]:
    qa_args = argparse.Namespace(
        ply=output_dir / ply_name,
        objects_jsonl=output_dir / objects_name,
        output_json=output_dir / "viewer_candidate_qa.json",
        output_md=output_dir / "viewer_candidate_qa.md",
        top_n=top_n,
        ambiguous_report=None,
        consolidation_report=None,
    )
    report = qa_viewer_candidate.build_report(qa_args)
    qa_args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    qa_viewer_candidate.write_markdown(qa_args.output_md, report, top_n)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--source-ply", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ply-name", default="frame_object_points_stride10.ply")
    parser.add_argument("--objects-name", default="frame_objects_viewer.jsonl")
    parser.add_argument("--min-scene-score", type=float, default=0.5)
    parser.add_argument("--qa-top-n", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    objects, report = relabel_ground_subtypes(read_jsonl(args.objects_jsonl), args.min_scene_score)
    output_objects = args.output_dir / args.objects_name
    output_ply = args.output_dir / args.ply_name
    write_jsonl(output_objects, objects)
    rewrite_report = rewrite_viewer_ply_semantics.rewrite_ply(args.source_ply, output_objects, output_ply)
    qa_report = run_qa(args.output_dir, args.ply_name, args.objects_name, args.qa_top_n)
    report.update({
        "objects_jsonl": str(args.objects_jsonl),
        "source_ply": str(args.source_ply),
        "output_objects_jsonl": str(output_objects),
        "output_ply": str(output_ply),
        "rewrite": rewrite_report,
        "qa": {"status": qa_report["status"], "warnings": qa_report["warnings"], "errors": qa_report["errors"]},
    })
    (args.output_dir / "ground_subtype_scene_prior_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if qa_report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
