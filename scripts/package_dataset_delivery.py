#!/usr/bin/env python3
"""Create a lightweight delivery package from the dataset manifest.

Large point-cloud/JSONL artifacts are referenced, not copied. The package is for
handoff, review, and reproducibility: it contains manifests, reports, previews,
and a large-file index that points to the authoritative local/remote artifacts.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from build_delivery_qa_index import render_html as render_qa_html
from build_delivery_qa_index import render_markdown as render_qa_markdown


DEFAULT_COPY_LIMIT = 32 * 1024 * 1024


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_name(role: str, path: Path) -> str:
    suffix = "".join(path.suffixes)
    stem = path.name[: -len(suffix)] if suffix else path.name
    return f"{role}{suffix or '_' + stem}"


def copy_or_reference(row: dict, package_dir: Path, copy_limit: int) -> tuple[dict, dict | None]:
    src = Path(row["path"])
    entry = dict(row)
    if not src.exists():
        entry["packaged"] = False
        entry["package_path"] = ""
        return entry, None
    size = src.stat().st_size
    if size > copy_limit:
        entry["packaged"] = False
        entry["package_path"] = ""
        return entry, {
            "role": row["role"],
            "path": str(src),
            "remote_path": row.get("remote_path", ""),
            "bytes": size,
            "reason": f"larger_than_copy_limit_{copy_limit}",
        }
    out_dir = package_dir / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / safe_name(row["role"], src)
    shutil.copy2(src, dst)
    entry["packaged"] = True
    entry["package_path"] = str(dst.relative_to(package_dir))
    return entry, None


def build_package(args: argparse.Namespace) -> dict:
    manifest = load_json(args.manifest)
    if args.output_dir.exists() and args.clean:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    copied_files = []
    large_files = []
    for row in manifest.get("files", []):
        entry, large = copy_or_reference(row, args.output_dir, args.copy_limit_bytes)
        copied_files.append(entry)
        if large:
            large_files.append(large)

    manifest_copy = args.output_dir / "dataset_delivery_manifest_0000_0999.json"
    manifest_copy.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    large_path = args.output_dir / "large_files.json"
    large_path.write_text(json.dumps({"files": large_files}, ensure_ascii=False, indent=2), encoding="utf-8")

    package_manifest = {
        "package_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(args.manifest),
        "dataset": manifest.get("dataset", {}),
        "metrics": manifest.get("metrics", {}),
        "checks": manifest.get("checks", []),
        "passed": bool(manifest.get("passed")) and not any(row.get("required") and not row.get("exists") for row in manifest.get("files", [])),
        "copy_limit_bytes": args.copy_limit_bytes,
        "files": copied_files,
        "large_files": large_files,
        "recommended_viewer_inputs": manifest.get("recommended_viewer_inputs", []),
    }
    package_manifest_path = args.output_dir / "package_manifest.json"
    package_manifest_path.write_text(json.dumps(package_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = args.output_dir / "README.md"
    readme.write_text(render_readme(package_manifest), encoding="utf-8")
    validation_stub = {"passed": package_manifest["passed"], "errors": []}
    (args.output_dir / "qa_index.md").write_text(render_qa_markdown(args.output_dir, package_manifest, validation_stub), encoding="utf-8")
    (args.output_dir / "qa_index.html").write_text(render_qa_html(args.output_dir, package_manifest, validation_stub), encoding="utf-8")

    if args.tgz:
        with tarfile.open(args.tgz, "w:gz") as tf:
            tf.add(args.output_dir, arcname=args.output_dir.name)
        package_manifest["tgz"] = str(args.tgz)
        package_manifest_path.write_text(json.dumps(package_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return package_manifest


def render_readme(package: dict) -> str:
    dataset = package.get("dataset", {})
    metrics = package.get("metrics", {})
    lines = [
        "# Dense Semantic Dataset Package",
        "",
        f"- dataset: `{dataset.get('name')}`",
        f"- status: `{dataset.get('status')}`",
        f"- passed: `{package.get('passed')}`",
        f"- frame range: `{dataset.get('frame_range')}`",
        f"- semantic combo: `{dataset.get('semantic_combo')}`",
        "",
        "## Core Metrics",
        "",
        f"- targets: `{metrics.get('target_count')}`",
        f"- target points: `{metrics.get('target_points')}`",
        f"- objects: `{metrics.get('object_count')}`",
        f"- surface-first changed ratio: `{metrics.get('surface_first_changed_ratio')}`",
        f"- residual surface assigned ratio: `{metrics.get('residual_surface_assigned_ratio')}`",
        f"- residual surface unassigned points: `{metrics.get('residual_surface_unassigned_points')}`",
        f"- residual absorption sweep best ratio: `{metrics.get('residual_absorption_sweep_best_ratio')}`",
        f"- residual miss reasons: `{metrics.get('residual_surface_miss_reason_counts')}`",
        f"- residual candidate coverage best ratio: `{metrics.get('residual_candidate_coverage_best_ratio')}`",
        f"- surface seed candidates/promoted points: `{metrics.get('surface_seed_candidate_points')}` / `{metrics.get('surface_seed_promoted_points')}`",
        f"- surface seed augmented best ratio: `{metrics.get('residual_candidate_coverage_augmented_best_ratio')}`",
        f"- surface fusion wall points base/strict: `{metrics.get('surface_fusion_wall_points_base')}` / `{metrics.get('surface_fusion_wall_points_strict')}`",
        f"- surface fusion ambiguous points base/strict: `{metrics.get('surface_fusion_ambiguous_points_base')}` / `{metrics.get('surface_fusion_ambiguous_points_strict')}`",
        f"- surface consolidation objects input/output/reduced: `{metrics.get('surface_consolidation_input_objects')}` / `{metrics.get('surface_consolidation_output_objects')}` / `{metrics.get('surface_consolidation_merged_reduction')}`",
        f"- hybrid surface consolidation objects input/output/reduced: `{metrics.get('surface_hybrid_consolidation_input_objects')}` / `{metrics.get('surface_hybrid_consolidation_output_objects')}` / `{metrics.get('surface_hybrid_consolidation_merged_reduction')}`",
        f"- fine targets / tracklets / reviewed objects: `{metrics.get('fine_targets')}` / `{metrics.get('fine_tracklets')}` / `{metrics.get('reviewed_output_objects')}`",
        f"- route decision: `{metrics.get('route_decision')}`",
        f"- release status: `{metrics.get('release_status')}`",
        f"- release manual gate: `{metrics.get('release_manual_gate')}`",
        f"- ConceptSeg decision: `{metrics.get('conceptseg_decision')}`",
        f"- side-track ConceptSeg decision: `{metrics.get('side_track_conceptseg_decision')}`",
        f"- side-track ConceptSeg accepted target ratio: `{metrics.get('side_track_conceptseg_accepted_target_ratio')}`",
        f"- ConceptSeg accepted intersection candidates: `{metrics.get('conceptseg_instance_accepted_candidates')}`",
        f"- ConceptSeg target status: `{metrics.get('conceptseg_instance_target_status_counts')}`",
        f"- ConceptSeg integration decision: `{metrics.get('conceptseg_integration_decision')}`",
        f"- ConceptSeg integration accepted targets/objects: `{metrics.get('conceptseg_integration_accepted_targets')}` / `{metrics.get('conceptseg_integration_accepted_objects')}`",
        f"- ConceptSeg 3D refinement components/points: `{metrics.get('conceptseg_3d_refinement_components')}` / `{metrics.get('conceptseg_3d_refinement_component_points')}`",
        f"- side-track old route decision: `{metrics.get('side_track_old_route_decision')}`",
        f"- old route decision: `{metrics.get('old_route_decision')}`",
        f"- old route reference passed: `{metrics.get('old_route_reference_passed')}`",
        f"- delivery acceptance passed: `{metrics.get('delivery_acceptance_passed')}`",
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
        "## Viewer Inputs",
        "",
    ]
    lines.extend(f"- `{path}`" for path in package.get("recommended_viewer_inputs", []))
    lines.extend(["", "## Large Files", ""])
    lines.extend(f"- `{row['role']}` `{row['bytes']}` bytes: `{row['path']}`" for row in package.get("large_files", []))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path("/Users/skkac/Work/SCAN")
    parser.add_argument("--manifest", type=Path, default=root / "route_status_20260610/dataset_delivery_manifest_0000_0999.json")
    parser.add_argument("--output-dir", type=Path, default=root / "dataset_delivery_0000_0999")
    parser.add_argument("--tgz", type=Path, default=root / "dataset_delivery_0000_0999.tgz")
    parser.add_argument("--copy-limit-bytes", type=int, default=DEFAULT_COPY_LIMIT)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    package = build_package(args)
    print(json.dumps({"output_dir": str(args.output_dir), "tgz": str(args.tgz), "passed": package["passed"], "large_files": len(package["large_files"])}, indent=2))
    if not package["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
