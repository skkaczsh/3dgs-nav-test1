#!/usr/bin/env python3
"""Build a lightweight human QA index for the dataset delivery package."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(package_dir: Path, path: Path) -> str:
    try:
        return str(path.relative_to(package_dir))
    except ValueError:
        return str(path)


def artifact_by_role(package: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["role"]: row for row in package.get("files", [])}


def link(label: str, href: str) -> str:
    return f'<a href="{html.escape(href)}">{html.escape(label)}</a>'


def row_link(package_dir: Path, role_map: dict[str, dict[str, Any]], role: str) -> str:
    row = role_map.get(role, {})
    if row.get("packaged"):
        return rel(package_dir, package_dir / row["package_path"])
    if row.get("path"):
        return row["path"]
    return ""


def browser_href(package_dir: Path, href: str) -> str:
    if not href:
        return ""
    path = Path(href)
    root = package_dir.parent
    if path.is_absolute():
        try:
            return "/" + str(path.relative_to(root))
        except ValueError:
            return href
    return f"/{package_dir.name}/{href}"


def viewer_link(package_dir: Path, href: str, mode: str = "semantic") -> str:
    if not href:
        return ""
    return f"/new_route/tools/semantic_ply_viewer.html?file={browser_href(package_dir, href)}&mode={mode}&stride=1&pointSize=2"


def render_markdown(package_dir: Path, package: dict[str, Any], validation: dict[str, Any]) -> str:
    metrics = package.get("metrics", {})
    role_map = artifact_by_role(package)
    lines = [
        "# Dataset QA Index",
        "",
        "## Gates",
        "",
        f"- package passed: `{package.get('passed')}`",
        f"- package validation passed: `{validation.get('passed')}`",
        f"- route decision: `{metrics.get('route_decision')}`",
        f"- release status: `{metrics.get('release_status')}`",
        f"- release manual gate: `{metrics.get('release_manual_gate')}`",
        f"- visual acceptance status: `{metrics.get('visual_acceptance_status')}`",
        f"- visual acceptance allow next increment: `{metrics.get('visual_acceptance_allow_next_increment')}`",
        f"- ConceptSeg decision: `{metrics.get('conceptseg_decision')}`",
        f"- old route decision: `{metrics.get('old_route_decision')}`",
        "",
        "## Visual Review Order",
        "",
        "1. Open the semantic PLY viewer: `tools/semantic_ply_viewer.html` from the repository root.",
        f"2. Load primary hybrid strict-surface PLY: `{row_link(package_dir, role_map, 'surface_hybrid_consolidated_preview_ply')}`.",
        f"3. Compare balanced consolidated PLY: `{row_link(package_dir, role_map, 'surface_consolidated_preview_ply')}`.",
        f"4. Compare strict raw surface PLY: `{row_link(package_dir, role_map, 'strict_surface_fusion_preview_ply')}`.",
        f"5. Compare old/base object PLY: `{row_link(package_dir, role_map, 'target_object_preview_ply')}`.",
        f"6. Inspect ConceptSeg 3D refinement PLY: `{row_link(package_dir, role_map, 'conceptseg_3d_components_ply')}`.",
        f"7. Inspect XY previews: `{row_link(package_dir, role_map, 'surface_hybrid_consolidated_xy_preview')}`, `{row_link(package_dir, role_map, 'surface_consolidated_xy_preview')}`, `{row_link(package_dir, role_map, 'strict_surface_fusion_xy_preview')}`.",
        f"8. Inspect surface-first XY preview: `{row_link(package_dir, role_map, 'surface_first_subcluster_xy_preview')}`.",
        f"9. Inspect residual surface-assignment preview: `{row_link(package_dir, role_map, 'residual_surface_assignment_xy_preview')}`.",
        f"10. Inspect ConceptSeg accepted sheet: `{row_link(package_dir, role_map, 'conceptseg_instance_accepted_sheet')}`.",
        f"11. Inspect old-route color preview: `{row_link(package_dir, role_map, 'old_route_color_smoke_preview')}`.",
        "",
        "## Local Review Server",
        "",
        "```bash",
        "cd /Users/skkac/Work/SCAN/new_route",
        "python3 scripts/serve_review_package.py --root /Users/skkac/Work/SCAN --host 127.0.0.1 --port 8765",
        "```",
        "",
        "- QA index: `http://127.0.0.1:8765/dataset_delivery_0000_0999/qa_index.html`",
        "- PLY viewer: `http://127.0.0.1:8765/new_route/tools/semantic_ply_viewer.html`",
        "",
        "## Key Metrics",
        "",
        f"- target count: `{metrics.get('target_count')}`",
        f"- object count: `{metrics.get('object_count')}`",
        f"- object ambiguous ratio: `{metrics.get('object_ambiguous_ratio')}`",
        f"- surface-first changed ratio: `{metrics.get('surface_first_changed_ratio')}`",
        f"- residual surface assigned ratio: `{metrics.get('residual_surface_assigned_ratio')}`",
        f"- residual surface unassigned points: `{metrics.get('residual_surface_unassigned_points')}`",
        f"- residual absorption sweep best ratio: `{metrics.get('residual_absorption_sweep_best_ratio')}`",
        f"- residual absorption sweep best unassigned: `{metrics.get('residual_absorption_sweep_best_unassigned')}`",
        f"- residual miss reasons: `{metrics.get('residual_surface_miss_reason_counts')}`",
        f"- residual candidate coverage best ratio: `{metrics.get('residual_candidate_coverage_best_ratio')}`",
        f"- residual candidate coverage best reasons: `{metrics.get('residual_candidate_coverage_best_reasons')}`",
        f"- surface seed candidates: `{metrics.get('surface_seed_candidate_count')}` / `{metrics.get('surface_seed_candidate_points')}` points",
        f"- surface seed augmented best ratio: `{metrics.get('residual_candidate_coverage_augmented_best_ratio')}`",
        f"- ConceptSeg accepted intersections: `{metrics.get('conceptseg_instance_accepted_candidates')}`",
        f"- ConceptSeg target status: `{metrics.get('conceptseg_instance_target_status_counts')}`",
        f"- ConceptSeg 3D refinement components/points: `{metrics.get('conceptseg_3d_refinement_components')}` / `{metrics.get('conceptseg_3d_refinement_component_points')}`",
        f"- ConceptSeg 3D refinement status: `{metrics.get('conceptseg_3d_refinement_status_counts')}`",
        f"- old-route colored ratio: `{metrics.get('old_route_colored_ratio')}`",
        "",
        "## Packaged Reports",
        "",
    ]
    for role in [
        "dense_semantic_route_decision_markdown",
        "dense_semantic_release_status_markdown",
        "visual_acceptance_review_markdown",
        "visual_acceptance_review_validation",
        "strict_output_validation",
        "target_object_qa",
        "object_pipeline_qa_summary",
        "surface_first_subcluster_report",
        "residual_surface_assignment_report",
        "residual_absorption_sweep_report",
        "residual_surface_miss_reasons_report",
        "residual_candidate_surface_coverage_report",
        "surface_seed_candidates_report",
        "surface_seed_promotion_report",
        "residual_candidate_surface_coverage_augmented_report",
        "conceptseg_fine_object_alignment",
        "conceptseg_instance_intersection",
        "conceptseg_integration_plan_markdown",
        "conceptseg_3d_refinement_report",
        "old_route_reference_validation",
        "v008_reviewed_merge_qa",
    ]:
        href = row_link(package_dir, role_map, role)
        if href:
            lines.append(f"- `{role}`: `{href}`")
    lines.extend(
        [
            "",
            "## No-Go Conditions",
            "",
            "- Do not treat ConceptSeg results as dense semantic production.",
            "- Do not revive the deprecated `transforms.json + project_world_points` route.",
            "- Do not extend beyond 0-999 frames until the current package is visually accepted.",
            "",
        ]
    )
    return "\n".join(lines)


def render_html(package_dir: Path, package: dict[str, Any], validation: dict[str, Any]) -> str:
    metrics = package.get("metrics", {})
    role_map = artifact_by_role(package)

    def card(title: str, body: str) -> str:
        return f"<section><h2>{html.escape(title)}</h2>{body}</section>"

    def visual_li(label: str, href: str, load_in_viewer: bool = False) -> str:
        if not href:
            return f"<li>{html.escape(label)}</li>"
        primary = link(label, browser_href(package_dir, href))
        if load_in_viewer and (href.endswith(".ply") or href.endswith(".jsonl") or href.endswith(".json")):
            return f"<li>{primary} <span class=\"muted\">({link('open in semantic viewer', viewer_link(package_dir, href, 'semantic'))})</span></li>"
        return f"<li>{primary}</li>"

    visual_items = [
        ("Primary hybrid strict-surface PLY", row_link(package_dir, role_map, "surface_hybrid_consolidated_preview_ply"), True),
        ("Balanced consolidated PLY", row_link(package_dir, role_map, "surface_consolidated_preview_ply"), True),
        ("Strict raw surface PLY", row_link(package_dir, role_map, "strict_surface_fusion_preview_ply"), True),
        ("Old/base object PLY", row_link(package_dir, role_map, "target_object_preview_ply"), True),
        ("ConceptSeg 3D refinement PLY", row_link(package_dir, role_map, "conceptseg_3d_components_ply"), True),
        ("Hybrid strict-surface XY preview", row_link(package_dir, role_map, "surface_hybrid_consolidated_xy_preview"), False),
        ("Balanced consolidated XY preview", row_link(package_dir, role_map, "surface_consolidated_xy_preview"), False),
        ("Strict raw surface XY preview", row_link(package_dir, role_map, "strict_surface_fusion_xy_preview"), False),
        ("Surface-first preview PLY", row_link(package_dir, role_map, "surface_first_subcluster_preview_ply"), True),
        ("Surface-first XY preview", row_link(package_dir, role_map, "surface_first_subcluster_xy_preview"), False),
        ("Residual surface-assignment XY preview", row_link(package_dir, role_map, "residual_surface_assignment_xy_preview"), False),
        ("ConceptSeg 3D refinement XY preview", row_link(package_dir, role_map, "conceptseg_3d_components_xy_preview"), False),
        ("ConceptSeg accepted sheet", row_link(package_dir, role_map, "conceptseg_instance_accepted_sheet"), False),
        ("Old-route color preview", row_link(package_dir, role_map, "old_route_color_smoke_preview"), False),
        ("Large file index", "large_files.json", False),
        ("Semantic PLY viewer", "/new_route/tools/semantic_ply_viewer.html", False),
    ]
    report_items = [
        ("Route decision", row_link(package_dir, role_map, "dense_semantic_route_decision_markdown")),
        ("Release status", row_link(package_dir, role_map, "dense_semantic_release_status_markdown")),
        ("Visual acceptance review", row_link(package_dir, role_map, "visual_acceptance_review_markdown")),
        ("Visual acceptance validation", row_link(package_dir, role_map, "visual_acceptance_review_validation")),
        ("Strict output validation", row_link(package_dir, role_map, "strict_output_validation")),
        ("Target/object QA", row_link(package_dir, role_map, "target_object_qa")),
        ("Object pipeline QA summary", row_link(package_dir, role_map, "object_pipeline_qa_summary")),
        ("Surface-first report", row_link(package_dir, role_map, "surface_first_subcluster_report")),
        ("Residual surface assignment", row_link(package_dir, role_map, "residual_surface_assignment_report")),
        ("Residual absorption sweep", row_link(package_dir, role_map, "residual_absorption_sweep_report")),
        ("Residual surface miss reasons", row_link(package_dir, role_map, "residual_surface_miss_reasons_report")),
        ("Residual candidate surface coverage", row_link(package_dir, role_map, "residual_candidate_surface_coverage_report")),
        ("Surface seed candidates", row_link(package_dir, role_map, "surface_seed_candidates_report")),
        ("Surface seed promotion", row_link(package_dir, role_map, "surface_seed_promotion_report")),
        ("Residual candidate coverage augmented", row_link(package_dir, role_map, "residual_candidate_surface_coverage_augmented_report")),
        ("ConceptSeg alignment", row_link(package_dir, role_map, "conceptseg_fine_object_alignment")),
        ("ConceptSeg intersection", row_link(package_dir, role_map, "conceptseg_instance_intersection")),
        ("ConceptSeg integration plan", row_link(package_dir, role_map, "conceptseg_integration_plan_markdown")),
        ("ConceptSeg 3D refinement", row_link(package_dir, role_map, "conceptseg_3d_refinement_report")),
        ("Old-route validation", row_link(package_dir, role_map, "old_route_reference_validation")),
    ]
    metric_rows = [
        ("Package passed", package.get("passed")),
        ("Validation passed", validation.get("passed")),
        ("Route decision", metrics.get("route_decision")),
        ("Release status", metrics.get("release_status")),
        ("Release manual gate", metrics.get("release_manual_gate")),
        ("Visual acceptance status", metrics.get("visual_acceptance_status")),
        ("Visual acceptance allow next increment", metrics.get("visual_acceptance_allow_next_increment")),
        ("ConceptSeg decision", metrics.get("conceptseg_decision")),
        ("Old route decision", metrics.get("old_route_decision")),
        ("Targets", metrics.get("target_count")),
        ("Objects", metrics.get("object_count")),
        ("Object ambiguous ratio", metrics.get("object_ambiguous_ratio")),
        ("Surface-first changed ratio", metrics.get("surface_first_changed_ratio")),
        ("Residual surface assigned ratio", metrics.get("residual_surface_assigned_ratio")),
        ("Residual surface unassigned points", metrics.get("residual_surface_unassigned_points")),
        ("Residual absorption sweep best ratio", metrics.get("residual_absorption_sweep_best_ratio")),
        ("Residual miss reasons", metrics.get("residual_surface_miss_reason_counts")),
        ("Residual candidate coverage best ratio", metrics.get("residual_candidate_coverage_best_ratio")),
        ("Surface seed candidate count", metrics.get("surface_seed_candidate_count")),
        ("Surface seed augmented best ratio", metrics.get("residual_candidate_coverage_augmented_best_ratio")),
        ("ConceptSeg accepted intersections", metrics.get("conceptseg_instance_accepted_candidates")),
        ("ConceptSeg 3D refinement components", metrics.get("conceptseg_3d_refinement_components")),
        ("ConceptSeg 3D refinement points", metrics.get("conceptseg_3d_refinement_component_points")),
        ("Old-route colored ratio", metrics.get("old_route_colored_ratio")),
    ]
    css = """
    body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;background:#101418;color:#e8edf2}
    h1{font-size:28px} h2{font-size:18px;margin-top:0}
    section{border:1px solid #2b3440;border-radius:8px;padding:16px;margin:16px 0;background:#151b22}
    a{color:#83c5ff} table{border-collapse:collapse;width:100%} td,th{border-bottom:1px solid #2b3440;padding:8px;text-align:left}
    code{background:#202833;padding:2px 4px;border-radius:4px}
    .muted{color:#9aa3af}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}
    """
    metrics_html = "<table>" + "".join(
        f"<tr><th>{html.escape(str(k))}</th><td><code>{html.escape(str(v))}</code></td></tr>" for k, v in metric_rows
    ) + "</table>"
    visual_html = "<ul>" + "".join(visual_li(label, href, load_in_viewer) for label, href, load_in_viewer in visual_items) + "</ul>"
    report_html = "<ul>" + "".join(
        f"<li>{link(label, href) if href else html.escape(label)}</li>" for label, href in report_items
    ) + "</ul>"
    review_html = """
    <ol>
      <li>Open the semantic PLY viewer and load the primary hybrid strict-surface PLY.</li>
      <li>Compare hybrid, balanced, strict raw, and old/base PLYs in that order.</li>
      <li>Check whether floor/wall/building regions are coherent and not fragmented into fine-object colors.</li>
      <li>Use residual surface-assignment evidence to distinguish surface-noise cleanup from unresolved fine-object residuals.</li>
      <li>Open the ConceptSeg 3D refinement PLY and accepted sheet; use them only as evidence for local fine-object refinements.</li>
      <li>Use the old-route color preview only as RGB sanity reference.</li>
    </ol>
    <p><code>python3 scripts/serve_review_package.py --root /Users/skkac/Work/SCAN --host 127.0.0.1 --port 8765</code></p>
    <p><a href="/new_route/tools/semantic_ply_viewer.html">Open PLY viewer</a></p>
    """
    nogo_html = """
    <ul>
      <li>Do not use ConceptSeg for dense semantic generation.</li>
      <li>Do not revive deprecated transforms.json/project_world_points projection.</li>
      <li>Do not extend beyond 0-999 frames before visual acceptance.</li>
    </ul>
    """
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Dense Semantic QA Index</title><style>{css}</style></head>
<body>
<h1>Dense Semantic QA Index</h1>
<div class="grid">
{card("Gates and Metrics", metrics_html)}
{card("Visual Inputs", visual_html)}
{card("Reports", report_html)}
{card("Review Order", review_html)}
{card("No-Go Conditions", nogo_html)}
</div>
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path("/Users/skkac/Work/SCAN")
    parser.add_argument("--package-dir", type=Path, default=root / "dataset_delivery_0000_0999")
    parser.add_argument("--validation", type=Path, default=root / "dataset_delivery_0000_0999_validation.json")
    args = parser.parse_args()

    package = read_json(args.package_dir / "package_manifest.json")
    validation = read_json(args.validation)
    md = render_markdown(args.package_dir, package, validation)
    html_text = render_html(args.package_dir, package, validation)
    md_path = args.package_dir / "qa_index.md"
    html_path = args.package_dir / "qa_index.html"
    md_path.write_text(md, encoding="utf-8")
    html_path.write_text(html_text, encoding="utf-8")
    print(json.dumps({"markdown": str(md_path), "html": str(html_path)}, indent=2))


if __name__ == "__main__":
    main()
