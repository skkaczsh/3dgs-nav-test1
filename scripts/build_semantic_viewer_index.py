#!/usr/bin/env python3
"""Build a static index for semantic PLY viewer artifacts.

The viewer artifacts are intentionally large and versioned by directory names.
This script scans those directories, extracts lightweight QA/report metadata,
and emits a JSON index consumed by tools/semantic_viewer_index.html.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.current_mainline_contract import forbidden_artifact_match


PLY_NAMES = (
    "frame_object_points_stride10.ply",
    "frame_object_points_local_geometry.ply",
    "frame_object_points.ply",
    "semantic_review_candidates_ascii.ply",
    "full_scene_objects_ascii.ply",
)
PLY_GLOB_PATTERNS = (
    "*_stride10.ply",
    "geo_patches_*.ply",
    "geo_patch_objects_*.ply",
    "objects_v*.ply",
)
OBJECT_NAMES = (
    "frame_objects_viewer.jsonl",
    "full_scene_objects_enriched.jsonl",
    "full_scene_objects_geometry_relabel.jsonl",
    "semantic_review_candidates.jsonl",
)
QA_NAMES = (
    "viewer_candidate_qa.json",
    "frame_object_viewer_export_report.json",
    "frame_object_points_local_geometry_report.json",
    "local_geometry_split_candidates_report.json",
)
REVIEW_INDEX_NAME = "semantic_object_review_index.json"
REVIEW_HTML_NAME = "semantic_object_review_index.html"
REVIEW_DECISION_NAME = "manual_object_review_decisions.csv"
REVIEW_NORMALIZED_NAME = "manual_object_review_decisions.normalized.jsonl"
REVIEW_NORMALIZE_REPORT_NAME = "manual_object_review_decisions.report.json"
REVIEW_APPLY_REPORT_NAME = "manual_object_review_apply_report.json"
REVIEW_EXPORT_REPORT_NAME = "manual_object_review_export_report.json"


@dataclass(frozen=True)
class ViewerArtifact:
    directory: Path
    ply: Path
    objects: Path | None


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def safe_stat(path: Path) -> os.stat_result | None:
    try:
        return path.stat()
    except OSError:
        return None


def iso_from_mtime(mtime: float) -> str:
    return datetime.fromtimestamp(mtime, timezone.utc).isoformat().replace("+00:00", "Z")


def first_existing(directory: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def companion_object_candidates(ply: Path) -> list[Path]:
    stem = ply.stem
    stems = [stem]
    for suffix in ("_stride10", "_stride5", "_stride3", "_stride2"):
        if stem.endswith(suffix):
            stems.append(stem[: -len(suffix)])
    out: list[Path] = []
    for candidate_stem in dict.fromkeys(stems):
        out.append(ply.with_name(f"{candidate_stem}.jsonl"))
    return out


def find_companion_objects(directory: Path, ply: Path) -> Path | None:
    exact = first_existing(directory, OBJECT_NAMES)
    if exact:
        return exact
    for candidate in companion_object_candidates(ply):
        if candidate.exists():
            return candidate
    return None


def find_generic_report(directory: Path, ply: Path) -> Path | None:
    stems = [candidate.stem for candidate in companion_object_candidates(ply)]
    stems.append(ply.stem)
    for stem in dict.fromkeys(stems):
        candidate = directory / f"{stem}_report.json"
        if candidate.exists():
            return candidate
    reports = sorted(directory.glob("*_report.json"))
    return reports[0] if len(reports) == 1 else None


def iter_viewer_artifacts(artifact_root: Path) -> list[ViewerArtifact]:
    artifacts: list[ViewerArtifact] = []
    ply_paths: set[Path] = set()
    for name in PLY_NAMES:
        ply_paths.update(artifact_root.rglob(name))
    for pattern in PLY_GLOB_PATTERNS:
        ply_paths.update(artifact_root.rglob(pattern))
    for ply in sorted(ply_paths):
        if forbidden_artifact_match(ply):
            continue
        directory = ply.parent
        objects = find_companion_objects(directory, ply)
        if objects and forbidden_artifact_match(objects):
            continue
        artifacts.append(ViewerArtifact(directory=directory, ply=ply, objects=objects))
    return artifacts


def web_visible_path(path: Path, web_root: Path) -> Path:
    """Map a real artifact path back through a symlink under the HTTP root."""
    path_abs = path.absolute()
    try:
        path_abs.relative_to(web_root.absolute())
        return path_abs
    except ValueError:
        pass

    path_resolved = path.resolve()
    web_root_resolved = web_root.resolve()
    try:
        rel = path_resolved.relative_to(web_root_resolved)
        return web_root / rel
    except ValueError:
        pass

    for child in web_root.iterdir():
        if not child.is_symlink():
            continue
        try:
            rel = path_resolved.relative_to(child.resolve())
        except ValueError:
            continue
        return child / rel

    return path_abs


def rel_url(path: Path, web_root: Path) -> str:
    visible = web_visible_path(path, web_root)
    try:
        rel = visible.absolute().relative_to(web_root.absolute())
    except ValueError:
        rel = visible.resolve().relative_to(web_root.resolve())
    return "/" + rel.as_posix()


def collect_review_indexes(web_root: Path, artifact_root: Path | None = None) -> dict[tuple[str, str], dict[str, Any]]:
    reviews: dict[tuple[str, str], dict[str, Any]] = {}
    roots = [web_root]
    if artifact_root is not None and artifact_root != web_root:
        roots.append(artifact_root)
    seen: set[Path] = set()
    review_jsons: list[Path] = []
    for root in roots:
        for review_json in root.rglob(REVIEW_INDEX_NAME):
            resolved = review_json.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            review_jsons.append(review_json)
    for review_json in review_jsons:
        report = read_json(review_json)
        if not report:
            continue
        objects = report.get("objects")
        if not isinstance(objects, list):
            objects = []
        first_object = objects[0] if objects else {}
        ply_url = first_object.get("semantic_url", "") if isinstance(first_object, dict) else ""
        objects_url = first_object.get("semantic_url", "") if isinstance(first_object, dict) else ""
        if isinstance(ply_url, str) and "file=" in ply_url:
            from urllib.parse import parse_qs, urlparse

            query = parse_qs(urlparse(ply_url).query)
            ply_url = (query.get("file") or [""])[0]
            objects_url = (query.get("objects") or [""])[0]
        key = (str(ply_url), str(objects_url))
        if not key[0]:
            continue
        directory = review_json.parent
        normalize_report = read_json(directory / REVIEW_NORMALIZE_REPORT_NAME)
        apply_report = read_json(directory / REVIEW_APPLY_REPORT_NAME)
        export_report = read_json(directory / REVIEW_EXPORT_REPORT_NAME)
        decision_csv = directory / REVIEW_DECISION_NAME
        normalized = directory / REVIEW_NORMALIZED_NAME
        review = {
            "review_json": rel_url(review_json, web_root),
            "review_html": rel_url(directory / REVIEW_HTML_NAME, web_root) if (directory / REVIEW_HTML_NAME).exists() else "",
            "decision_csv": rel_url(decision_csv, web_root) if decision_csv.exists() else "",
            "normalized_jsonl": rel_url(normalized, web_root) if normalized.exists() else "",
            "normalize_report": rel_url(directory / REVIEW_NORMALIZE_REPORT_NAME, web_root) if normalize_report else "",
            "apply_report": rel_url(directory / REVIEW_APPLY_REPORT_NAME, web_root) if apply_report else "",
            "export_report": rel_url(directory / REVIEW_EXPORT_REPORT_NAME, web_root) if export_report else "",
            "object_count": len(objects),
            "normalize": {
                "accepted_count": normalize_report.get("accepted_count"),
                "error_count": normalize_report.get("error_count"),
            } if normalize_report else None,
            "apply": {
                "applied_count": apply_report.get("applied_count"),
                "error_count": apply_report.get("error_count"),
            } if apply_report else None,
            "export": {
                "output_dir": export_report.get("output_dir"),
                "qa": export_report.get("qa"),
            } if export_report else None,
        }
        reviews[key] = review
    return reviews


def extract_counts(
    qa: dict[str, Any],
    export_report: dict[str, Any],
    localgeom_report: dict[str, Any],
    generic_report: dict[str, Any],
) -> dict[str, Any]:
    ply_qa = qa.get("ply") if isinstance(qa.get("ply"), dict) else {}
    semantic_counts = ply_qa.get("semantic_point_counts")
    if not isinstance(semantic_counts, dict):
        semantic_counts = (
            localgeom_report.get("point_label_counts")
            if isinstance(localgeom_report.get("point_label_counts"), dict)
            else export_report.get("label_counts")
        )
    if not isinstance(semantic_counts, dict):
        semantic_counts = {}

    object_counts = localgeom_report.get("object_label_counts")
    if not isinstance(object_counts, dict):
        objects_qa = qa.get("objects") if isinstance(qa.get("objects"), dict) else {}
        object_counts = objects_qa.get("label_counts") if isinstance(objects_qa.get("label_counts"), dict) else {}

    vertex_count = ply_qa.get("vertex_count")
    if vertex_count is None:
        vertex_count = localgeom_report.get("input_vertex_count")
    if vertex_count is None:
        vertex_count = export_report.get("output_vertices")
    if vertex_count is None:
        vertex_count = generic_report.get("preview_points") or generic_report.get("input_point_count")

    object_count = localgeom_report.get("output_object_count")
    if object_count is None:
        object_count = export_report.get("object_records") or export_report.get("object_count_with_points")
    if object_count is None:
        object_count = generic_report.get("output_patch_count") or generic_report.get("jsonl_patch_count")
    if object_count is None and isinstance(object_counts, dict):
        object_count = sum(v for v in object_counts.values() if isinstance(v, int))

    point_source_support_counts = export_report.get("point_source_support_counts")
    if not isinstance(point_source_support_counts, dict):
        point_source_support_counts = {}
    object_source_support_counts = export_report.get("object_source_support_counts")
    if not isinstance(object_source_support_counts, dict):
        object_source_support_counts = {}
    fusion_status_counts = export_report.get("fusion_status_counts")
    if not isinstance(fusion_status_counts, dict):
        fusion_status_counts = {}
    conflict_flag_counts = export_report.get("conflict_flag_counts")
    if not isinstance(conflict_flag_counts, dict):
        conflict_flag_counts = {}

    return {
        "vertex_count": vertex_count,
        "object_count": object_count,
        "semantic_point_counts": semantic_counts,
        "object_label_counts": object_counts if isinstance(object_counts, dict) else {},
        "point_source_support_counts": point_source_support_counts,
        "object_source_support_counts": object_source_support_counts,
        "fusion_status_counts": fusion_status_counts,
        "conflict_flag_counts": conflict_flag_counts,
    }


def count_total(counts: dict[str, Any]) -> int:
    total = 0
    for value in counts.values():
        try:
            total += int(value)
        except (TypeError, ValueError):
            continue
    return total


def ratio_for_keys(counts: dict[str, Any], keys: tuple[str, ...]) -> float:
    total = count_total(counts)
    if total <= 0:
        return 0.0
    matched = 0
    for key in keys:
        try:
            matched += int(counts.get(key, 0) or 0)
        except (TypeError, ValueError):
            continue
    return matched / total


def evidence_risk_warnings(counts: dict[str, Any]) -> list[str]:
    point_sources = counts.get("point_source_support_counts")
    if not isinstance(point_sources, dict):
        point_sources = {}
    object_sources = counts.get("object_source_support_counts")
    if not isinstance(object_sources, dict):
        object_sources = {}
    conflict_flags = counts.get("conflict_flag_counts")
    if not isinstance(conflict_flags, dict):
        conflict_flags = {}

    warnings: list[str] = []
    missing_point_ratio = ratio_for_keys(point_sources, ("missing_object", "missing_source_scores", "no_label_source_support"))
    if missing_point_ratio >= 0.01:
        warnings.append(f"evidence provenance missing/unsupported for {missing_point_ratio:.1%} of visible points")
    missing_object_ratio = ratio_for_keys(object_sources, ("missing_object", "missing_source_scores", "no_label_source_support"))
    if missing_object_ratio >= 0.01:
        warnings.append(f"evidence provenance missing/unsupported for {missing_object_ratio:.1%} of visible objects")

    scene_point_ratio = ratio_for_keys(point_sources, ("scene",))
    if scene_point_ratio >= 0.05:
        warnings.append(f"scene-only support covers {scene_point_ratio:.1%} of visible points")
    scene_object_ratio = ratio_for_keys(object_sources, ("scene",))
    if scene_object_ratio >= 0.05:
        warnings.append(f"scene-only support covers {scene_object_ratio:.1%} of visible objects")

    try:
        geometry_veto_count = int(conflict_flags.get("geometry_vetoed_some_evidence", 0) or 0)
    except (TypeError, ValueError):
        geometry_veto_count = 0
    object_total = count_total(object_sources)
    if object_total > 0 and geometry_veto_count / object_total >= 0.10:
        warnings.append(f"geometry veto evidence is dense: {geometry_veto_count} flags over {object_total} visible objects")
    return warnings


def build_entry(
    artifact: ViewerArtifact,
    web_root: Path,
    artifact_root: Path,
    reviews: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    directory = artifact.directory
    qa = read_json(directory / "viewer_candidate_qa.json")
    export_report = read_json(directory / "frame_object_viewer_export_report.json")
    localgeom_report = read_json(directory / "frame_object_points_local_geometry_report.json")
    split_report = read_json(directory / "local_geometry_split_candidates_report.json")
    generic_report_path = find_generic_report(directory, artifact.ply)
    generic_report = read_json(generic_report_path) if generic_report_path else {}

    watched_paths = [artifact.ply]
    if artifact.objects:
        watched_paths.append(artifact.objects)
    watched_paths.extend(directory / name for name in QA_NAMES if (directory / name).exists())
    if generic_report_path:
        watched_paths.append(generic_report_path)
    stats = [safe_stat(path) for path in watched_paths]
    mtimes = [stat.st_mtime for stat in stats if stat is not None]
    sizes = {path.name: stat.st_size for path, stat in zip(watched_paths, stats) if stat is not None}
    updated_at_ts = max(mtimes) if mtimes else 0.0

    counts = extract_counts(qa, export_report, localgeom_report, generic_report)
    warnings = qa.get("warnings") if isinstance(qa.get("warnings"), list) else []
    evidence_warnings = evidence_risk_warnings(counts)
    warnings = list(warnings) + evidence_warnings
    errors = qa.get("errors") if isinstance(qa.get("errors"), list) else []
    status = qa.get("status") or ("report_only" if generic_report else ("missing_qa" if not qa else "unknown"))

    file_url = rel_url(artifact.ply, web_root)
    objects_url = rel_url(artifact.objects, web_root) if artifact.objects else ""
    semantic_viewer = f"/tools/semantic_ply_viewer.html?file={file_url}&mode=semantic&stride=1&pointSize=1.5"
    object_viewer = f"/tools/semantic_ply_viewer.html?file={file_url}&mode=object&stride=1&pointSize=1.5"
    if objects_url:
        semantic_viewer += f"&objects={objects_url}"
        object_viewer += f"&objects={objects_url}"

    try:
        rel_dir = directory.absolute().relative_to(artifact_root.absolute()).as_posix()
    except ValueError:
        rel_dir = directory.resolve().relative_to(artifact_root.resolve()).as_posix()
    review = (reviews or {}).get((file_url, objects_url)) or (reviews or {}).get((file_url, ""))
    return {
        "name": directory.name,
        "relative_dir": rel_dir,
        "updated_at": iso_from_mtime(updated_at_ts),
        "updated_at_ts": updated_at_ts,
        "status": status,
        "warnings": warnings,
        "evidence_warnings": evidence_warnings,
        "errors": errors,
        "ply": file_url,
        "objects": objects_url,
        "viewer_urls": {
            "semantic": semantic_viewer,
            "object": object_viewer,
            "rgb": semantic_viewer.replace("mode=semantic", "mode=rgb"),
        },
        "sizes": sizes,
        "counts": counts,
        "reports": {
            "qa": qa if qa else None,
            "export": export_report if export_report else None,
            "local_geometry": localgeom_report if localgeom_report else None,
            "split_candidates": split_report if split_report else None,
            "generic": generic_report if generic_report else None,
        },
        "review": review,
    }


def build_index(web_root: Path, artifact_root: Path) -> dict[str, Any]:
    reviews = collect_review_indexes(web_root, artifact_root)
    entries = [build_entry(artifact, web_root, artifact_root, reviews) for artifact in iter_viewer_artifacts(artifact_root)]
    entries.sort(key=lambda item: item["updated_at_ts"], reverse=True)
    return {
        "schema": "semantic-viewer-index/v1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "web_root": str(web_root),
        "artifact_root": str(artifact_root),
        "artifact_count": len(entries),
        "entries": entries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web-root", type=Path, default=Path.cwd(), help="HTTP server root for URL generation.")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("server_parking_priority_s10"),
        help="Directory to scan for viewer artifacts, relative to --web-root unless absolute.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tools/semantic_viewer_index.json"),
        help="Output JSON path, relative to --web-root unless absolute.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    web_root = args.web_root.resolve()
    artifact_root = args.artifact_root if args.artifact_root.is_absolute() else web_root / args.artifact_root
    output = args.output if args.output.is_absolute() else web_root / args.output

    index = build_index(web_root=web_root, artifact_root=artifact_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(index, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "artifact_count": index["artifact_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
