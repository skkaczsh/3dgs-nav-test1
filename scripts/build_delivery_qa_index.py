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
        f"- ConceptSeg decision: `{metrics.get('conceptseg_decision')}`",
        f"- old route decision: `{metrics.get('old_route_decision')}`",
        "",
        "## Visual Review Order",
        "",
        "1. Open the semantic PLY viewer: `tools/semantic_ply_viewer.html` from the repository root.",
        f"2. Load surface-first preview PLY: `{row_link(package_dir, role_map, 'surface_first_subcluster_preview_ply')}`.",
        "3. Compare against the full/stride object PLY listed in `large_files.json`.",
        f"4. Inspect surface-first XY preview: `{row_link(package_dir, role_map, 'surface_first_subcluster_xy_preview')}`.",
        f"5. Inspect residual surface-assignment preview: `{row_link(package_dir, role_map, 'residual_surface_assignment_xy_preview')}`.",
        f"6. Inspect ConceptSeg accepted sheet: `{row_link(package_dir, role_map, 'conceptseg_instance_accepted_sheet')}`.",
        f"7. Inspect old-route color preview: `{row_link(package_dir, role_map, 'old_route_color_smoke_preview')}`.",
        "",
        "## Key Metrics",
        "",
        f"- target count: `{metrics.get('target_count')}`",
        f"- object count: `{metrics.get('object_count')}`",
        f"- object ambiguous ratio: `{metrics.get('object_ambiguous_ratio')}`",
        f"- surface-first changed ratio: `{metrics.get('surface_first_changed_ratio')}`",
        f"- residual surface assigned ratio: `{metrics.get('residual_surface_assigned_ratio')}`",
        f"- residual surface unassigned points: `{metrics.get('residual_surface_unassigned_points')}`",
        f"- ConceptSeg accepted intersections: `{metrics.get('conceptseg_instance_accepted_candidates')}`",
        f"- ConceptSeg target status: `{metrics.get('conceptseg_instance_target_status_counts')}`",
        f"- old-route colored ratio: `{metrics.get('old_route_colored_ratio')}`",
        "",
        "## Packaged Reports",
        "",
    ]
    for role in [
        "dense_semantic_route_decision_markdown",
        "strict_output_validation",
        "target_object_qa",
        "object_pipeline_qa_summary",
        "surface_first_subcluster_report",
        "residual_surface_assignment_report",
        "conceptseg_fine_object_alignment",
        "conceptseg_instance_intersection",
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

    visual_items = [
        ("Surface-first preview PLY", row_link(package_dir, role_map, "surface_first_subcluster_preview_ply")),
        ("Surface-first XY preview", row_link(package_dir, role_map, "surface_first_subcluster_xy_preview")),
        ("Residual surface-assignment XY preview", row_link(package_dir, role_map, "residual_surface_assignment_xy_preview")),
        ("ConceptSeg accepted sheet", row_link(package_dir, role_map, "conceptseg_instance_accepted_sheet")),
        ("Old-route color preview", row_link(package_dir, role_map, "old_route_color_smoke_preview")),
        ("Large file index", "large_files.json"),
        ("Semantic PLY viewer", "../new_route/tools/semantic_ply_viewer.html"),
    ]
    report_items = [
        ("Route decision", row_link(package_dir, role_map, "dense_semantic_route_decision_markdown")),
        ("Strict output validation", row_link(package_dir, role_map, "strict_output_validation")),
        ("Target/object QA", row_link(package_dir, role_map, "target_object_qa")),
        ("Object pipeline QA summary", row_link(package_dir, role_map, "object_pipeline_qa_summary")),
        ("Surface-first report", row_link(package_dir, role_map, "surface_first_subcluster_report")),
        ("Residual surface assignment", row_link(package_dir, role_map, "residual_surface_assignment_report")),
        ("ConceptSeg alignment", row_link(package_dir, role_map, "conceptseg_fine_object_alignment")),
        ("ConceptSeg intersection", row_link(package_dir, role_map, "conceptseg_instance_intersection")),
        ("Old-route validation", row_link(package_dir, role_map, "old_route_reference_validation")),
    ]
    metric_rows = [
        ("Package passed", package.get("passed")),
        ("Validation passed", validation.get("passed")),
        ("Route decision", metrics.get("route_decision")),
        ("ConceptSeg decision", metrics.get("conceptseg_decision")),
        ("Old route decision", metrics.get("old_route_decision")),
        ("Targets", metrics.get("target_count")),
        ("Objects", metrics.get("object_count")),
        ("Object ambiguous ratio", metrics.get("object_ambiguous_ratio")),
        ("Surface-first changed ratio", metrics.get("surface_first_changed_ratio")),
        ("Residual surface assigned ratio", metrics.get("residual_surface_assigned_ratio")),
        ("Residual surface unassigned points", metrics.get("residual_surface_unassigned_points")),
        ("ConceptSeg accepted intersections", metrics.get("conceptseg_instance_accepted_candidates")),
        ("Old-route colored ratio", metrics.get("old_route_colored_ratio")),
    ]
    css = """
    body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;background:#101418;color:#e8edf2}
    h1{font-size:28px} h2{font-size:18px;margin-top:0}
    section{border:1px solid #2b3440;border-radius:8px;padding:16px;margin:16px 0;background:#151b22}
    a{color:#83c5ff} table{border-collapse:collapse;width:100%} td,th{border-bottom:1px solid #2b3440;padding:8px;text-align:left}
    code{background:#202833;padding:2px 4px;border-radius:4px}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}
    """
    metrics_html = "<table>" + "".join(
        f"<tr><th>{html.escape(str(k))}</th><td><code>{html.escape(str(v))}</code></td></tr>" for k, v in metric_rows
    ) + "</table>"
    visual_html = "<ul>" + "".join(
        f"<li>{link(label, href) if href else html.escape(label)}</li>" for label, href in visual_items
    ) + "</ul>"
    report_html = "<ul>" + "".join(
        f"<li>{link(label, href) if href else html.escape(label)}</li>" for label, href in report_items
    ) + "</ul>"
    review_html = """
    <ol>
      <li>Open the semantic PLY viewer and load the surface-first preview PLY.</li>
      <li>Check whether floor/wall/building regions are coherent and not fragmented into fine-object colors.</li>
      <li>Use residual surface-assignment evidence to distinguish surface-noise cleanup from unresolved fine-object residuals.</li>
      <li>Open the ConceptSeg accepted sheet; use it only as evidence for local fine-object refinements.</li>
      <li>Use the old-route color preview only as RGB sanity reference.</li>
    </ol>
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
