#!/usr/bin/env python3
"""Create the manual visual acceptance record for the 0-999 dataset package."""

from __future__ import annotations

import argparse
import html
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


def viewer_href(path: str, mode: str = "semantic") -> str:
    if path.startswith("/Users/skkac/Work/SCAN/"):
        file_param = "/" + path.removeprefix("/Users/skkac/Work/SCAN/")
    else:
        file_param = path
    return f"/new_route/tools/semantic_ply_viewer.html?file={html.escape(file_param)}&mode={mode}&stride=1&pointSize=2"


def render_html(review: dict[str, Any]) -> str:
    def esc(value: Any) -> str:
        return html.escape(str(value))

    status_class = "accepted" if review["status"] == "accepted" else ("blocked" if review["status"] == "blocked" else "pending")
    viewer_rows = []
    for path in review["recommended_viewer_inputs"]:
        label = Path(path).name
        viewer_rows.append(
            f"<li><code>{esc(label)}</code> <a href=\"{viewer_href(path)}\">open in viewer</a><br><span>{esc(path)}</span></li>"
        )
    check_rows = []
    for row in review["checks"]:
        check_rows.append(
            f"""
            <article class="check {esc(row.get('status'))}">
              <header>
                <h3>{esc(row.get('id'))}</h3>
                <span>{esc(row.get('status'))}</span>
              </header>
              <p>{esc(row.get('question'))}</p>
              <dl>
                <dt>required</dt><dd><code>{esc(row.get('required'))}</code></dd>
                <dt>evidence</dt><dd><code>{esc(row.get('evidence', []))}</code></dd>
                <dt>notes</dt><dd>{esc(row.get('notes', ''))}</dd>
              </dl>
            </article>
            """
        )
    payload = json.dumps(review, ensure_ascii=False, indent=2)
    css = """
    body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;background:#101418;color:#e8edf2}
    h1{font-size:28px;margin-bottom:8px} h2{font-size:18px;margin-top:0} h3{font-size:16px;margin:0}
    a{color:#83c5ff} code,pre{background:#202833;border-radius:4px} code{padding:2px 4px} pre{padding:12px;overflow:auto}
    section,.check{border:1px solid #2b3440;border-radius:8px;padding:16px;margin:16px 0;background:#151b22}
    .status{display:inline-block;padding:4px 8px;border-radius:999px;font-weight:600}
    .status.pending{background:#3a2f14;color:#ffd37a}.status.accepted{background:#16351e;color:#8ee69d}.status.blocked{background:#401d1d;color:#ff9b9b}
    .check header{display:flex;justify-content:space-between;gap:16px;align-items:center}
    .check header span{font-size:12px;text-transform:uppercase;color:#9aa3af}
    .check.pending{border-color:#6a541d}.check.accepted{border-color:#255f35}.check.blocked,.check.rejected{border-color:#7a3030}
    dl{display:grid;grid-template-columns:120px 1fr;gap:8px;margin:0} dt{color:#9aa3af} dd{margin:0}
    textarea{width:100%;min-height:280px;background:#0b0f14;color:#e8edf2;border:1px solid #2b3440;border-radius:8px;padding:12px;font-family:ui-monospace,Menlo,monospace}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
    """
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Visual Acceptance Review</title><style>{css}</style></head>
<body>
<h1>Visual Acceptance Review</h1>
<p><span class="status {status_class}">{esc(review['status'])}</span> allow next increment: <code>{esc(review['allow_next_increment'])}</code></p>
<div class="grid">
<section>
  <h2>Dataset</h2>
  <p><code>{esc(review['dataset'])}</code></p>
  <p>Reviewer: <code>{esc(review.get('reviewer',''))}</code></p>
  <p>Reviewed at: <code>{esc(review.get('reviewed_at',''))}</code></p>
</section>
<section>
  <h2>Commands</h2>
  <pre>cd /Users/skkac/Work/SCAN/new_route
python3 scripts/validate_visual_acceptance_review.py
python3 scripts/validate_visual_acceptance_review.py --require-accepted</pre>
  <p>The second command is expected to fail until every required check is accepted.</p>
</section>
</div>
<section>
  <h2>Viewer Inputs</h2>
  <ol>{''.join(viewer_rows)}</ol>
</section>
<section>
  <h2>Checks</h2>
  {''.join(check_rows)}
</section>
<section>
  <h2>JSON To Edit</h2>
  <p>Edit <code>/Users/skkac/Work/SCAN/route_status_20260610/visual_acceptance_review_20260611.json</code>, then rerun validation and packaging.</p>
  <textarea spellcheck="false">{esc(payload)}</textarea>
</section>
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, default=ROOT / "route_status_20260610/dataset_delivery_manifest_0000_0999.json")
    parser.add_argument("--package-manifest", type=Path, default=ROOT / "dataset_delivery_0000_0999/package_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "route_status_20260610/visual_acceptance_review_20260611.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "route_status_20260610/visual_acceptance_review_20260611.md")
    parser.add_argument("--html", type=Path, default=ROOT / "route_status_20260610/visual_acceptance_review_20260611.html")
    args = parser.parse_args()

    review = build_review(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    args.markdown.write_text(render_markdown(review), encoding="utf-8")
    args.html.write_text(render_html(review), encoding="utf-8")
    print(json.dumps({"json": str(args.output), "markdown": str(args.markdown), "html": str(args.html), "status": review["status"]}, indent=2))


if __name__ == "__main__":
    main()
