#!/usr/bin/env python3
"""Create the manual visual acceptance record for the 0-999 dataset package."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/Users/skkac/Work/SCAN")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def default_checks() -> list[dict[str, Any]]:
    return [
        {
            "id": "geometry_structure",
            "status": "pending",
            "required": True,
            "question": "Primary PLY geometry matches the known rooftop/Mid360 structure without collapsed or inverted projection artifacts.",
            "evidence": [],
            "notes": "",
        },
        {
            "id": "surface_coherence",
            "status": "pending",
            "required": True,
            "question": "Large rooftop floor/wall/building surfaces are coherent enough for the next 1000-frame increment.",
            "evidence": [],
            "notes": "",
        },
        {
            "id": "wall_floor_contamination",
            "status": "pending",
            "required": True,
            "question": "Wall/floor contamination is acceptable after strict surface fusion and hybrid consolidation.",
            "evidence": [],
            "notes": "",
        },
        {
            "id": "fine_object_reasonableness",
            "status": "pending",
            "required": True,
            "question": "Equipment/railing/pipe fine objects are not worse than the old/base object PLY and are acceptable as a review candidate.",
            "evidence": [],
            "notes": "",
        },
        {
            "id": "residual_risk_understood",
            "status": "pending",
            "required": True,
            "question": "Residual unassigned surface points are understood as a known bottleneck, not a projection-chain regression.",
            "evidence": [],
            "notes": "",
        },
        {
            "id": "conceptseg_not_promoted",
            "status": "accepted",
            "required": True,
            "question": "ConceptSeg outputs are treated only as review-only fine-object refinement proposals.",
            "evidence": ["conceptseg_3d_refinement_report", "conceptseg_integration_plan"],
            "notes": "Policy gate: do not use ConceptSeg as dense semantic source.",
        },
        {
            "id": "old_route_not_promoted",
            "status": "accepted",
            "required": True,
            "question": "Old route is used only as RGB/geometric visual reference.",
            "evidence": ["old_route_reference_validation"],
            "notes": "Policy gate: do not revive deprecated transforms.json + project_world_points semantic route.",
        },
    ]


def build_review(args: argparse.Namespace) -> dict[str, Any]:
    manifest = read_json(args.dataset_manifest)
    package = read_json(args.package_manifest)
    existing = read_json(args.output) if args.output.exists() else {}
    checks = existing.get("checks") or default_checks()
    required = [row for row in checks if row.get("required")]
    accepted_required = [row for row in required if row.get("status") == "accepted"]
    rejected_required = [row for row in required if row.get("status") in {"rejected", "blocked"}]
    all_required_accepted = len(required) == len(accepted_required)

    return {
        "review_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": manifest.get("dataset", {}).get("name", "new_route_dense_semantic_0000_0999"),
            "frame_range": manifest.get("dataset", {}).get("frame_range", [0, 999]),
            "semantic_combo": manifest.get("dataset", {}).get("semantic_combo", "sam2_prompt_v3_sky_label_merge_completion"),
            "projection_route": manifest.get("dataset", {}).get("projection_route", "img_pos.txt + cam_in_ex.txt + Tcl + Til"),
        },
        "status": "accepted" if all_required_accepted else ("blocked" if rejected_required else "pending"),
        "allow_next_increment": all_required_accepted,
        "reviewer": existing.get("reviewer", ""),
        "reviewed_at": existing.get("reviewed_at", ""),
        "summary": existing.get("summary", ""),
        "blockers": existing.get("blockers", []),
        "recommended_viewer_inputs": manifest.get("recommended_viewer_inputs") or package.get("recommended_viewer_inputs", []),
        "review_urls": [
            "http://127.0.0.1:8765/dataset_delivery_0000_0999/qa_index.html",
            "http://127.0.0.1:8765/new_route/tools/semantic_ply_viewer.html",
        ],
        "checks": checks,
        "instructions": [
            "Set each required check status to accepted, rejected, or blocked after visual review.",
            "Only status=accepted with allow_next_increment=true should unblock the 1000-1999 main-route increment.",
            "Attach screenshot paths or CloudCompare notes in evidence/notes for any rejected or blocked check.",
        ],
    }


def render_markdown(review: dict[str, Any]) -> str:
    lines = [
        "# Visual Acceptance Review",
        "",
        f"- status: `{review['status']}`",
        f"- allow next increment: `{review['allow_next_increment']}`",
        f"- reviewer: `{review.get('reviewer', '')}`",
        f"- reviewed at: `{review.get('reviewed_at', '')}`",
        f"- dataset: `{review['dataset']}`",
        "",
        "## Review URLs",
        "",
    ]
    lines.extend(f"- {url}" for url in review["review_urls"])
    lines.extend(["", "## Viewer Inputs", ""])
    lines.extend(f"- `{path}`" for path in review["recommended_viewer_inputs"])
    lines.extend(["", "## Checks", ""])
    for row in review["checks"]:
        lines.extend(
            [
                f"### {row['id']}",
                "",
                f"- status: `{row.get('status')}`",
                f"- required: `{row.get('required')}`",
                f"- question: {row.get('question')}",
                f"- evidence: `{row.get('evidence', [])}`",
                f"- notes: {row.get('notes', '')}",
                "",
            ]
        )
    lines.extend(["## Instructions", ""])
    lines.extend(f"- {item}" for item in review["instructions"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, default=ROOT / "route_status_20260610/dataset_delivery_manifest_0000_0999.json")
    parser.add_argument("--package-manifest", type=Path, default=ROOT / "dataset_delivery_0000_0999/package_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "route_status_20260610/visual_acceptance_review_20260611.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "route_status_20260610/visual_acceptance_review_20260611.md")
    args = parser.parse_args()

    review = build_review(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    args.markdown.write_text(render_markdown(review), encoding="utf-8")
    print(json.dumps({"json": str(args.output), "markdown": str(args.markdown), "status": review["status"]}, indent=2))


if __name__ == "__main__":
    main()
