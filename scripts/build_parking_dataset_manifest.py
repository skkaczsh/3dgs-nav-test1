#!/usr/bin/env python3
"""Build a reusable manifest for the parking priority semantic dataset.

This manifest is intentionally separate from the older indoor 0-999 delivery
manifest.  The parking route is now surface-first:

1. trusted ground/wall/grass/ceiling surfaces
2. point-level surface-trust guard from drivability_cpp
3. remaining fine candidates for later DINO / detector review

The script is read-only for source artifacts and writes JSON + Markdown
manifests next to the current output bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SEMANTIC_NAMES = {
    0: "unknown",
    1: "other",
    2: "wall",
    3: "floor",
    4: "ceiling",
    5: "grass",
    6: "tree",
    7: "person",
    8: "car",
    9: "railing",
    10: "building",
    11: "sky",
    12: "road",
    13: "water",
    14: "furniture",
    15: "pipe",
    16: "equipment",
    17: "fine_candidate",
    255: "ignore",
}

TRUSTED_SURFACE_LABELS = {"floor", "wall", "ceiling", "grass"}
FINE_REVIEW_LABELS = {"unknown", "fine_candidate", "car", "railing"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_entry(path: Path, role: str, remote_path: str = "", required: bool = True, include_hash: bool = False) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "role": role,
        "path": str(path),
        "remote_path": remote_path,
        "required": required,
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
    }
    if include_hash and path.exists():
        entry["sha256"] = sha256_file(path)
    return entry


def remote_file_entry(host: str, remote_path: str) -> dict[str, Any]:
    if not host or not remote_path:
        return {"host": host, "path": remote_path, "checked": False}
    script = (
        "python3 - <<'PY'\n"
        "from pathlib import Path\n"
        "import json\n"
        f"p = Path({remote_path!r})\n"
        "print(json.dumps({'exists': p.exists(), 'bytes': p.stat().st_size if p.exists() else 0}))\n"
        "PY"
    )
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, script],
        text=True,
        capture_output=True,
        check=False,
    )
    out = {"host": host, "path": remote_path, "checked": True, "returncode": proc.returncode}
    if proc.returncode == 0:
        try:
            out.update(json.loads(proc.stdout))
        except json.JSONDecodeError:
            out.update({"exists": False, "parse_error": proc.stdout[-500:]})
    else:
        out.update({"exists": False, "stderr": proc.stderr[-500:]})
    return out


def parse_ply_semantics(path: Path) -> dict[str, Any]:
    props: list[str] = []
    vertex_count = 0
    header_lines = 0
    in_vertex = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            header_lines += 1
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] == "format" and parts[1] != "ascii":
                raise ValueError(f"Only ascii PLY is supported: {path}")
            if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
                vertex_count = int(parts[2])
                in_vertex = True
            elif len(parts) >= 2 and parts[0] == "element":
                in_vertex = False
            elif in_vertex and len(parts) >= 3 and parts[0] == "property":
                props.append(parts[-1])
            elif line.strip() == "end_header":
                break
    idx = {name: i for i, name in enumerate(props)}
    if "semantic" not in idx:
        raise ValueError(f"PLY missing semantic field: {path}")
    object_col = idx.get("object", idx.get("object_id"))
    if object_col is None:
        raise ValueError(f"PLY missing object/object_id field: {path}")

    semantic_counts: Counter[str] = Counter()
    semantic_id_counts: Counter[int] = Counter()
    object_counts: Counter[int] = Counter()
    rows = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for _ in range(header_lines):
            next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) <= max(idx["semantic"], object_col):
                continue
            sem_id = int(round(float(parts[idx["semantic"]])))
            object_id = int(round(float(parts[object_col])))
            semantic_id_counts[sem_id] += 1
            semantic_counts[SEMANTIC_NAMES.get(sem_id, f"semantic_{sem_id}")] += 1
            object_counts[object_id] += 1
            rows += 1
    return {
        "header_vertex_count": vertex_count,
        "parsed_vertex_count": rows,
        "semantic_id_counts": {str(k): int(v) for k, v in sorted(semantic_id_counts.items())},
        "semantic_counts": dict(semantic_counts),
        "object_count_in_ply": len(object_counts),
        "top_objects_by_point_count": [
            {"object_id": int(k), "points": int(v)}
            for k, v in object_counts.most_common(20)
        ],
    }


def summarize_objects(objects: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts = Counter(str(row.get("semantic_label") or "unknown") for row in objects)
    stage_counts = Counter(str(row.get("downstream_stage") or "") for row in objects)
    status_counts = Counter(str(row.get("surface_trust_guard_status") or "") for row in objects)
    review_status_counts = Counter(str(row.get("review_status") or "") for row in objects)
    geometry_counts = Counter(str(row.get("geometry_class") or "") for row in objects)
    scene_context_counts = Counter(str(row.get("scene_context") or "") for row in objects)
    for counter in (stage_counts, status_counts, review_status_counts, geometry_counts, scene_context_counts):
        counter.pop("", None)

    stable_surfaces = [row for row in objects if str(row.get("semantic_label") or "") in TRUSTED_SURFACE_LABELS]
    fine_candidates = [row for row in objects if str(row.get("semantic_label") or "") in FINE_REVIEW_LABELS]
    return {
        "object_count": len(objects),
        "semantic_label_counts": dict(label_counts),
        "downstream_stage_counts": dict(stage_counts),
        "surface_trust_guard_status_counts": dict(status_counts),
        "review_status_counts": dict(review_status_counts),
        "geometry_class_counts": dict(geometry_counts),
        "scene_context_counts_top": dict(scene_context_counts.most_common(20)),
        "trusted_surface_object_count": len(stable_surfaces),
        "fine_review_object_count": len(fine_candidates),
        "trusted_surface_point_count_from_objects": int(sum(int(row.get("point_count") or 0) for row in stable_surfaces)),
        "fine_review_point_count_from_objects": int(sum(int(row.get("point_count") or 0) for row in fine_candidates)),
    }


def check(name: str, passed: bool, detail: str, value: Any = None) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail, "value": value}


def render_markdown(manifest: dict[str, Any]) -> str:
    metrics = manifest["metrics"]
    files = manifest["files"]
    lines = [
        "# Parking Priority Dataset Manifest",
        "",
        f"- generated at: `{manifest['generated_at']}`",
        f"- status: `{manifest['status']}`",
        f"- current stage: `{manifest['dataset']['stage']}`",
        f"- viewer: {manifest['viewer']['url']}",
        "",
        "## Point Metrics",
        "",
    ]
    point_counts = metrics["ply"]["semantic_counts"]
    for label in ["floor", "wall", "ceiling", "grass", "car", "railing", "fine_candidate", "unknown"]:
        lines.append(f"- {label}: `{point_counts.get(label, 0)}`")
    lines.extend(
        [
            f"- trusted surface points: `{metrics['trusted_surface_points']}` (`{metrics['trusted_surface_point_ratio']:.4f}`)",
            f"- fine/review points: `{metrics['fine_review_points']}` (`{metrics['fine_review_point_ratio']:.4f}`)",
            f"- v19 changed points: `{metrics['surface_trust_guard']['changed_points']}`",
            "",
            "## Object Metrics",
            "",
            f"- objects: `{metrics['objects']['object_count']}`",
            f"- trusted surface objects: `{metrics['objects']['trusted_surface_object_count']}`",
            f"- fine/review objects: `{metrics['objects']['fine_review_object_count']}`",
            f"- downstream stages: `{metrics['objects']['downstream_stage_counts']}`",
            "",
            "## Files",
            "",
        ]
    )
    for row in files:
        lines.append(f"- {row['role']}: `{row['path']}` bytes `{row['bytes']}`")
    lines.extend(["", "## Checks", ""])
    for row in manifest["checks"]:
        mark = "PASS" if row["passed"] else "FAIL"
        lines.append(f"- {mark} `{row['name']}`: {row['detail']}")
    return "\n".join(lines) + "\n"


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    report = read_json(args.report)
    objects = read_jsonl(args.objects_jsonl)
    ply = parse_ply_semantics(args.ply)
    object_summary = summarize_objects(objects)

    files = [
        file_entry(args.ply, "v19_surface_trust_guard_ply", args.remote_ply, include_hash=args.include_hash),
        file_entry(args.objects_jsonl, "v19_surface_trust_guard_objects_jsonl", args.remote_objects_jsonl, include_hash=args.include_hash),
        file_entry(args.report, "v19_surface_trust_guard_report", args.remote_report, include_hash=args.include_hash),
        file_entry(args.viewer_html, "local_viewer_entry", required=True, include_hash=False),
        file_entry(args.route_doc, "route_documentation", required=True, include_hash=False),
    ]
    remote_files = [
        remote_file_entry(args.remote_host, args.remote_ply),
        remote_file_entry(args.remote_host, args.remote_objects_jsonl),
        remote_file_entry(args.remote_host, args.remote_report),
    ] if args.remote_host else []

    semantic_counts = ply["semantic_counts"]
    total_points = int(ply["parsed_vertex_count"])
    trusted_surface_points = sum(int(semantic_counts.get(label, 0)) for label in TRUSTED_SURFACE_LABELS)
    fine_review_points = sum(int(semantic_counts.get(label, 0)) for label in FINE_REVIEW_LABELS)
    person_points = int(ply["semantic_id_counts"].get("7", 0))
    report_after = report.get("semantic_counts_after", {})
    mismatches = {
        key: {"ply": int(semantic_counts.get(key, 0)), "report": int(report_after.get(key, 0))}
        for key in sorted(set(semantic_counts) | set(report_after))
        if int(semantic_counts.get(key, 0)) != int(report_after.get(key, 0))
    }

    checks = [
        check("required_files_exist", all(row["exists"] and row["bytes"] > 0 for row in files if row["required"]), "all required local files exist"),
        check("remote_files_exist", all(row.get("exists") and int(row.get("bytes") or 0) > 0 for row in remote_files), "all required remote files exist" if remote_files else "remote check skipped", remote_files),
        check("ply_vertex_count_matches_header", ply["parsed_vertex_count"] == ply["header_vertex_count"], "parsed PLY rows match header vertex count"),
        check("no_person_semantic_id", person_points == 0, "semantic id 7 is reserved for person and must not appear in this route", person_points),
        check("report_matches_ply_counts", not mismatches, "v19 report semantic_counts_after matches parsed PLY counts", mismatches),
        check("surface_guard_changed_points", int(report.get("changed_points") or 0) > 0, "surface-trust guard restored some points to trusted surfaces", report.get("changed_points")),
        check("trusted_surface_ratio", trusted_surface_points / max(total_points, 1) >= args.min_trusted_surface_ratio, "trusted surfaces cover expected share of the parking scene", trusted_surface_points / max(total_points, 1)),
        check("car_points_present", int(semantic_counts.get("car", 0)) > 0, "car points remain after surface guard", semantic_counts.get("car", 0)),
        check("fine_candidates_present", int(semantic_counts.get("fine_candidate", 0)) > 0, "fine candidates remain for later DINO/detector review", semantic_counts.get("fine_candidate", 0)),
    ]
    passed = all(row["passed"] for row in checks)

    return {
        "manifest_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready_for_visual_qa_and_fine_object_stage" if passed else "not_ready",
        "passed": passed,
        "dataset": {
            "name": "MT20260616-175807_parking_priority_surface_guard",
            "raw_dataset_local": str(args.raw_dataset_local),
            "raw_dataset_remote": args.raw_dataset_remote,
            "stage": "v19_surface_trust_guard",
            "route": "priority surfaces + drivability_cpp trusted surface guard + fine-object candidates",
            "trusted_surface_source": "drivability_cpp full-point wallbfs prior",
            "fine_object_next_stage": "DINOv3/GroundingDINO detector review over remaining fine candidates",
        },
        "files": files,
        "remote_files": remote_files,
        "viewer": {
            "entry": str(args.viewer_html),
            "url": args.viewer_url,
            "ply": str(args.ply),
            "objects_jsonl": str(args.objects_jsonl),
        },
        "metrics": {
            "ply": ply,
            "objects": object_summary,
            "surface_trust_guard": report,
            "trusted_surface_points": trusted_surface_points,
            "trusted_surface_point_ratio": trusted_surface_points / max(total_points, 1),
            "fine_review_points": fine_review_points,
            "fine_review_point_ratio": fine_review_points / max(total_points, 1),
        },
        "checks": checks,
        "next_actions": [
            "Use v19 as the canonical viewer/dataset entry for human QA.",
            "Run DINOv3/GroundingDINO only on remaining fine_candidate/car/railing/unknown candidates, not on trusted floor/wall/grass surfaces.",
            "Keep full-point drivability wallbfs prior for future surface guard runs; voxel-only prior lacks wall labels for this dataset.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("/Users/skkac/Work/SCAN")
    repo = root / "new_route"
    out_dir = repo / "server_parking_priority_s10/full_scene_surface_trust_guard_v19"
    parser.add_argument("--ply", type=Path, default=out_dir / "full_scene_surface_trust_guard_v19.ply")
    parser.add_argument("--objects-jsonl", type=Path, default=out_dir / "full_scene_surface_trust_guard_v19.jsonl")
    parser.add_argument("--report", type=Path, default=out_dir / "full_scene_surface_trust_guard_v19_report.json")
    parser.add_argument("--output", type=Path, default=out_dir / "parking_dataset_manifest_v19.json")
    parser.add_argument("--markdown", type=Path, default=out_dir / "parking_dataset_manifest_v19.md")
    parser.add_argument("--viewer-html", type=Path, default=repo / "tools/parking_full_scene_viewer.html")
    parser.add_argument("--viewer-url", default="http://127.0.0.1:8765/tools/parking_full_scene_viewer.html")
    parser.add_argument("--route-doc", type=Path, default=repo / "docs/parking_priority_route_20260617.md")
    parser.add_argument("--raw-dataset-local", type=Path, default=root / "MT20260616-175807")
    parser.add_argument("--raw-dataset-remote", default="/root/epfs/datasets/MT20260616-175807")
    parser.add_argument("--remote-host", default="scan-train")
    parser.add_argument("--remote-root", default="/root/epfs/work_MT20260616-175807/full_scene_surface_trust_guard_v19")
    parser.add_argument("--remote-ply", default="")
    parser.add_argument("--remote-objects-jsonl", default="")
    parser.add_argument("--remote-report", default="")
    parser.add_argument("--min-trusted-surface-ratio", type=float, default=0.60)
    parser.add_argument("--include-hash", action="store_true")
    args = parser.parse_args()

    if not args.remote_ply:
        args.remote_ply = f"{args.remote_root}/full_scene_surface_trust_guard_v19.ply"
    if not args.remote_objects_jsonl:
        args.remote_objects_jsonl = f"{args.remote_root}/full_scene_surface_trust_guard_v19.jsonl"
    if not args.remote_report:
        args.remote_report = f"{args.remote_root}/full_scene_surface_trust_guard_v19_report.json"

    manifest = build_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    args.markdown.write_text(render_markdown(manifest), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "markdown": str(args.markdown),
        "passed": manifest["passed"],
        "status": manifest["status"],
        "trusted_surface_point_ratio": manifest["metrics"]["trusted_surface_point_ratio"],
        "fine_review_point_ratio": manifest["metrics"]["fine_review_point_ratio"],
    }, ensure_ascii=False, indent=2))
    if not manifest["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
