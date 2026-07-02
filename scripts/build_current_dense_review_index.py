#!/usr/bin/env python3
"""Build a small review index for the current dense mainline artifacts."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.current_mainline_contract import FORBIDDEN_ARTIFACT_SUBSTRINGS, forbidden_artifact_match

DEFAULT_QA = REPO_ROOT / "docs" / "current_dense_mainline_qa.json"
DEFAULT_VISUAL_ACCEPTANCE = REPO_ROOT / "docs" / "current_dense_visual_acceptance.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "current_dense_review_index.html"


ARTIFACTS = [
    {
        "id": "v7_object_refinement",
        "title": "v7 Object Refinement",
        "role": "conservative object baseline",
        "mode": "object",
        "point_size": 1.2,
        "ply": "/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/dense_patch_object_refinement_v7_r4_attach_v4_20260624_170126/objects_v7_structural_multimaterial/geo_patch_objects_v7_structural_multimaterial_stride10.ply",
        "objects": "/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/dense_patch_object_refinement_v7_r4_attach_v4_20260624_170126/objects_v7_structural_multimaterial/geo_patch_objects_v7_structural_multimaterial.jsonl",
        "note": "Lower recall; useful as over-merge guard.",
    },
    {
        "id": "v8_object_refinement",
        "title": "v8 Object Refinement",
        "role": "high-recall object candidate",
        "mode": "object",
        "point_size": 1.2,
        "ply": "/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/dense_patch_object_refinement_v8_tiny_attach_20260624_170619/objects_v7_structural_multimaterial/geo_patch_objects_v7_structural_multimaterial_stride10.ply",
        "objects": "/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/dense_patch_object_refinement_v8_tiny_attach_20260624_170619/objects_v7_structural_multimaterial/geo_patch_objects_v7_structural_multimaterial.jsonl",
        "note": "More merges; requires visual QA before promotion.",
    },
    {
        "id": "v9_teacher_semantic",
        "title": "v9 Teacher Semantic",
        "role": "safe semantic diagnostic",
        "mode": "semantic",
        "point_size": 1.2,
        "ply": "/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/objects_v9_teacher_v20_semantic/objects_v9_teacher_v20_semantic.ply",
        "objects": "/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/objects_v9_teacher_v20_semantic/objects_v9_teacher_v20_semantic.jsonl",
        "note": "Teacher transfer without the surface-preserve guard wrapper.",
    },
    {
        "id": "v17_surface_preserve_guard",
        "title": "v17 Surface Preserve Guard",
        "role": "unknown-regression guard reference",
        "mode": "semantic",
        "point_size": 1.2,
        "ply": "/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/objects_v17_teacher_v20_surface_preserve_guard/objects_v17_teacher_v20_surface_preserve_guard.ply",
        "objects": "/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/objects_v17_teacher_v20_surface_preserve_guard/objects_v17_teacher_v20_surface_preserve_guard.jsonl",
        "note": "Same point labels as v9; proves guard does not create unknown spike.",
    },
]

def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def viewer_url(artifact: dict[str, Any], mode: str | None = None) -> str:
    chosen_mode = mode or str(artifact["mode"])
    url = (
        "/tools/semantic_ply_viewer.html"
        f"?file={artifact['ply']}"
        f"&objects={artifact['objects']}"
        f"&mode={chosen_mode}"
        "&stride=1"
        f"&pointSize={artifact['point_size']}"
    )
    return url


def local_artifact_path(path: str) -> Path:
    if path.startswith("/"):
        return REPO_ROOT / path.lstrip("/")
    return REPO_ROOT / path


def validate_artifact_files(artifacts: list[dict[str, Any]] = ARTIFACTS) -> list[str]:
    errors: list[str] = []
    for item in artifacts:
        artifact_id = str(item.get("id", "<unknown>"))
        ply_path = local_artifact_path(str(item.get("ply", "")))
        objects_path = local_artifact_path(str(item.get("objects", "")))
        if not ply_path.is_file():
            errors.append(f"artifact_ply_missing={artifact_id}:{ply_path}")
        else:
            with ply_path.open("rb") as fh:
                if fh.readline().strip() != b"ply":
                    errors.append(f"artifact_ply_bad_header={artifact_id}:{ply_path}")
        if not objects_path.is_file():
            errors.append(f"artifact_objects_missing={artifact_id}:{objects_path}")
        elif objects_path.stat().st_size <= 0:
            errors.append(f"artifact_objects_empty={artifact_id}:{objects_path}")
    return errors


def validate_artifact_allowlist(
    artifacts: list[dict[str, Any]] = ARTIFACTS,
    *,
    check_files: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    artifact_ids = [str(item.get("id", "")) for item in artifacts]
    if len(artifact_ids) != len(set(artifact_ids)):
        errors.append("duplicate_artifact_ids")
    expected_ids = {
        "v7_object_refinement",
        "v8_object_refinement",
        "v9_teacher_semantic",
        "v17_surface_preserve_guard",
    }
    extra_ids = sorted(set(artifact_ids) - expected_ids)
    missing_ids = sorted(expected_ids - set(artifact_ids))
    if extra_ids:
        errors.append(f"unexpected_artifact_ids={extra_ids}")
    if missing_ids:
        errors.append(f"missing_artifact_ids={missing_ids}")
    for item in artifacts:
        haystack = "\n".join(
            str(item.get(key, ""))
            for key in ("id", "title", "role", "ply", "objects", "note")
        )
        forbidden = forbidden_artifact_match(haystack)
        if forbidden:
            errors.append(f"forbidden_artifact_reference={item.get('id')}:{forbidden}")
    if check_files:
        errors.extend(validate_artifact_files(artifacts))
    return {
        "schema": "current-dense-review-artifact-allowlist/v1",
        "passed": not errors,
        "artifact_ids": artifact_ids,
        "forbidden_substrings": list(FORBIDDEN_ARTIFACT_SUBSTRINGS),
        "errors": errors,
    }


def metric_rows(qa: dict[str, Any]) -> str:
    metrics = qa["object_refinement"]["metrics"]
    lines = []
    for key in [
        "candidate_count",
        "accepted_candidate_rows",
        "output_object_count",
        "mixed_object_voxel_ratio_020",
        "object_count_in_overlap_preview",
    ]:
        lines.append(
            "<tr>"
            f"<td>{html.escape(key)}</td>"
            f"<td>{metrics['v7'][key]}</td>"
            f"<td>{metrics['v8'][key]}</td>"
            f"<td>{metrics['delta_v8_minus_v7'][key]}</td>"
            "</tr>"
        )
    return "\n".join(lines)


def label_rows(qa: dict[str, Any]) -> str:
    counts = qa["surface_guard"]["label_point_counts"]
    labels = sorted(set(counts["v9"]) | set(counts["v17"]))
    lines = []
    for label in labels:
        lines.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{counts['v9'].get(label, 0)}</td>"
            f"<td>{counts['v17'].get(label, 0)}</td>"
            f"<td>{counts['delta_v17_minus_v9'].get(label, 0)}</td>"
            "</tr>"
        )
    return "\n".join(lines)


def visual_check_rows(visual: dict[str, Any] | None) -> str:
    if not visual:
        return '<tr><td colspan="4">No visual acceptance record provided.</td></tr>'
    lines = []
    for row in visual.get("checks", []):
        if not isinstance(row, dict):
            continue
        required = "yes" if row.get("required") else "no"
        lines.append(
            "<tr>"
            f"<td><code>{html.escape(str(row.get('id', '')))}</code></td>"
            f"<td>{html.escape(required)}</td>"
            f"<td>{html.escape(str(row.get('status', '')))}</td>"
            f"<td>{html.escape(str(row.get('question', '')))}</td>"
            "</tr>"
        )
    return "\n".join(lines)


def artifact_cards() -> str:
    cards = []
    for item in ARTIFACTS:
        cards.append(
            f"""
      <article class="card">
        <h2>{html.escape(item["title"])}</h2>
        <p class="role">{html.escape(item["role"])}</p>
        <p>{html.escape(item["note"])}</p>
        <div class="actions">
          <a href="{html.escape(viewer_url(item, item["mode"]))}" target="_blank">Open default</a>
          <a href="{html.escape(viewer_url(item, "object"))}" target="_blank">Object mode</a>
          <a href="{html.escape(viewer_url(item, "semantic"))}" target="_blank">Semantic mode</a>
        </div>
        <code>{html.escape(item["ply"])}</code>
      </article>
"""
        )
    return "\n".join(cards)


def build_html(qa: dict[str, Any], visual: dict[str, Any] | None = None) -> str:
    visual_status = visual.get("status", "missing") if visual else "missing"
    accepted_candidate = visual.get("accepted_candidate", "unknown") if visual else "unknown"
    update_command = (
        "python3 scripts/update_current_dense_visual_acceptance.py "
        "--check-id <check_id> --status accepted --reviewer <name> "
        "--notes <brief_evidence> --run-gate"
    )
    gate_command = (
        "python3 scripts/gate_current_dense_mainline_promotion.py "
        "--qa-json docs/current_dense_mainline_qa.json "
        "--visual-acceptance docs/current_dense_visual_acceptance.json "
        "--output docs/current_dense_promotion_gate.json"
    )
    plan_command = "python3 scripts/plan_current_dense_promotion.py"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Current Dense Mainline Review</title>
  <style>
    :root {{ color-scheme: dark; --bg:#111318; --panel:#181c23; --line:#303641; --text:#eef1f5; --muted:#a3abb7; --accent:#63c5dd; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    header {{ padding: 18px 22px; border-bottom: 1px solid var(--line); }}
    h1 {{ margin: 0 0 6px; font-size: 20px; }}
    main {{ display: grid; gap: 16px; padding: 16px; }}
    section, .card {{ border: 1px solid var(--line); border-radius: 8px; background: var(--panel); padding: 14px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }}
    h2 {{ margin: 0 0 8px; font-size: 15px; }}
    .role, .muted {{ color: var(--muted); }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 7px 8px; border-bottom: 1px solid var(--line); text-align: left; }}
    th {{ color: var(--muted); }}
    a {{ color: var(--accent); text-decoration: none; }}
    .actions {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0; }}
    .actions a {{ border: 1px solid var(--line); border-radius: 6px; padding: 7px 9px; background: #202631; }}
    code {{ display: block; color: var(--muted); word-break: break-all; font-size: 12px; }}
    .status {{ display: inline-block; border: 1px solid var(--line); border-radius: 999px; padding: 3px 8px; color: var(--muted); }}
  </style>
</head>
<body>
  <header>
    <h1>Current Dense Mainline Review</h1>
    <div class="muted">Only current approved comparison artifacts are linked here. Rejected diagnostic runs are intentionally excluded.</div>
  </header>
  <main>
    <section>
      <h2>Promotion Review Checklist</h2>
      <p class="muted">Candidate <code>{html.escape(str(accepted_candidate))}</code> is not promoted until all required visual checks are accepted. Current status: <span class="status">{html.escape(str(visual_status))}</span></p>
      <table>
        <thead><tr><th>check id</th><th>required</th><th>status</th><th>question</th></tr></thead>
        <tbody>
{visual_check_rows(visual)}
        </tbody>
      </table>
      <p class="muted">Update one accepted check after reviewing the fixed viewer links:</p>
      <code>{html.escape(update_command)}</code>
      <p class="muted">Re-run gate explicitly if needed:</p>
      <code>{html.escape(gate_command)}</code>
      <p class="muted">After the gate passes, generate the exact state-change plan:</p>
      <code>{html.escape(plan_command)}</code>
    </section>
    <section>
      <h2>Object Refinement QA</h2>
      <table>
        <thead><tr><th>metric</th><th>v7</th><th>v8</th><th>delta</th></tr></thead>
        <tbody>
{metric_rows(qa)}
        </tbody>
      </table>
    </section>
    <section>
      <h2>Surface Guard QA</h2>
      <table>
        <thead><tr><th>label</th><th>v9 points</th><th>v17 points</th><th>delta</th></tr></thead>
        <tbody>
{label_rows(qa)}
        </tbody>
      </table>
    </section>
    <section>
      <h2>Viewer Links</h2>
      <div class="grid">
{artifact_cards()}
      </div>
    </section>
  </main>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-json", type=Path, default=DEFAULT_QA)
    parser.add_argument("--visual-acceptance", type=Path, default=DEFAULT_VISUAL_ACCEPTANCE)
    parser.add_argument("--output-html", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    qa = read_json(args.qa_json)
    visual = read_json(args.visual_acceptance) if args.visual_acceptance.exists() else None
    allowlist = validate_artifact_allowlist()
    if not allowlist["passed"]:
        raise SystemExit(json.dumps(allowlist, ensure_ascii=False, indent=2))
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(build_html(qa, visual), encoding="utf-8")
    print(
        json.dumps(
            {"output_html": str(args.output_html), "artifact_count": len(ARTIFACTS), "allowlist": allowlist},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
