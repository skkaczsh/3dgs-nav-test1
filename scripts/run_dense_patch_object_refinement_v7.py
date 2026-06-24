#!/usr/bin/env python3
"""Run or print the dense Patch object-refinement v7 command chain.

v7 starts from the current dense v6 patch labels.  It does not create patch
boundaries from semantic labels; it only proposes and accepts object-level
unions from geometric/contact evidence.  The default mode is dry-run so the
operator can inspect paths before launching a large run.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any


FORBIDDEN_INPUT_SUBSTRINGS = (
    "frame_object_points_stride10.ply",
    "objects_v12_teacher_v20_grid6_unknown_absorb",
    "objects_v14_teacher_v20_grid6_geometry_guard_wall_recall",
    "objects_v15_teacher_v20_grid6_geometry_guard_no_wall_to_floor",
    "objects_v16_teacher_v20_grid6_geometry_guard_surface_recall",
)


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def reject_forbidden_path(path: Path) -> None:
    value = str(path)
    for forbidden in FORBIDDEN_INPUT_SUBSTRINGS:
        if forbidden in value:
            raise ValueError(f"forbidden input path contains {forbidden}: {value}")


def existing_file(path: Path, name: str) -> None:
    reject_forbidden_path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name} missing: {path}")
    if not path.is_file():
        raise ValueError(f"{name} is not a file: {path}")


def read_state(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("schema") != "current-dense-patch-state/v1":
        raise ValueError(f"unexpected dense patch state schema: {data.get('schema')!r}")
    return data


def default_patch_labels(state: dict[str, Any]) -> Path:
    for item in state.get("current_patch_baseline", {}).get("local_paths", []):
        p = Path(item)
        if p.name.endswith("_labels.bin"):
            return p
    raise ValueError("current_patch_baseline does not contain a local labels.bin path")


def build_commands(args: argparse.Namespace) -> dict[str, Any]:
    candidates_dir = args.output_dir / "object_merge_candidates_v7_structural_multimaterial"
    objects_dir = args.output_dir / "objects_v7_structural_multimaterial"
    candidates_jsonl = candidates_dir / "geo_patch_object_merge_candidates.jsonl"

    propose = [
        args.python,
        "scripts/propose_geo_patch_object_merges.py",
        "--region-input",
        str(args.region_input),
        "--labels",
        str(args.patch_labels),
        "--output-dir",
        str(candidates_dir),
        "--edge-source",
        args.edge_source,
        "--min-patch-voxels",
        str(args.min_patch_voxels),
        "--min-shared-edges",
        str(args.min_shared_edges),
        "--min-contact-ratio",
        str(args.min_contact_ratio),
        "--max-bbox-gap",
        str(args.max_bbox_gap),
        "--max-color-distance",
        str(args.max_color_distance),
        "--min-normal-score",
        str(args.min_normal_score),
        "--min-bucket-score",
        str(args.min_bucket_score),
        "--min-score",
        str(args.min_score),
        "--contact-ratio-norm",
        str(args.contact_ratio_norm),
        "--max-candidates",
        str(args.max_candidates),
        "--enable-structural-multimaterial",
        "--min-structural-score",
        str(args.min_structural_score),
        "--structural-min-contact-ratio",
        str(args.structural_min_contact_ratio),
        "--structural-min-shared-edges",
        str(args.structural_min_shared_edges),
        "--structural-min-normal-score",
        str(args.structural_min_normal_score),
        "--structural-max-bbox-gap",
        str(args.structural_max_bbox_gap),
    ]
    if args.edge_source == "grid6":
        propose.extend(["--grid-voxel-size", str(args.grid_voxel_size)])

    build = [
        args.python,
        "scripts/build_geo_patch_objects_from_candidates.py",
        "--region-input",
        str(args.region_input),
        "--patch-labels",
        str(args.patch_labels),
        "--candidates-jsonl",
        str(candidates_jsonl),
        "--output-dir",
        str(objects_dir),
        "--output-stem",
        "geo_patch_objects_v7_structural_multimaterial",
        "--preview-stride",
        str(args.preview_stride),
        "--min-score",
        str(args.accept_min_score),
        "--min-contact-ratio",
        str(args.accept_min_contact_ratio),
        "--min-shared-edges",
        str(args.accept_min_shared_edges),
        "--max-color-distance",
        str(args.accept_max_color_distance),
        "--max-bbox-gap",
        str(args.accept_max_bbox_gap),
        "--min-normal-score",
        str(args.accept_min_normal_score),
        "--enable-attachment-model",
        "--attachment-min-score",
        str(args.attachment_min_score),
        "--attachment-min-contact-ratio",
        str(args.attachment_min_contact_ratio),
        "--attachment-min-shared-edges",
        str(args.attachment_min_shared_edges),
        "--attachment-max-color-distance",
        str(args.attachment_max_color_distance),
        "--attachment-min-normal-score",
        str(args.attachment_min_normal_score),
        "--attachment-max-bbox-gap",
        str(args.attachment_max_bbox_gap),
        "--attachment-max-fragment-voxels",
        str(args.attachment_max_fragment_voxels),
        "--attachment-min-anchor-voxels",
        str(args.attachment_min_anchor_voxels),
        "--attachment-min-size-ratio",
        str(args.attachment_min_size_ratio),
        "--enable-structural-multimaterial",
        "--min-structural-score",
        str(args.accept_min_structural_score),
        "--structural-min-contact-ratio",
        str(args.accept_structural_min_contact_ratio),
        "--structural-min-shared-edges",
        str(args.accept_structural_min_shared_edges),
        "--structural-min-normal-score",
        str(args.accept_structural_min_normal_score),
        "--structural-max-bbox-gap",
        str(args.accept_structural_max_bbox_gap),
    ]

    return {
        "schema": "dense-patch-object-refinement-v7-plan/v1",
        "output_dir": str(args.output_dir),
        "candidates_dir": str(candidates_dir),
        "objects_dir": str(objects_dir),
        "commands": [
            {"name": "propose_candidates", "argv": propose, "shell": shell_join(propose)},
            {"name": "build_objects", "argv": build, "shell": shell_join(build)},
        ],
    }


def run_command(argv: list[str], cwd: Path) -> None:
    subprocess.run(argv, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=Path("docs/current_dense_patch_state.json"))
    parser.add_argument("--region-input", type=Path, required=True)
    parser.add_argument("--patch-labels", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--run", action="store_true", help="Execute commands; default only writes command plan")
    parser.add_argument("--plan-json", type=Path, help="Optional path for the generated command plan")

    parser.add_argument("--edge-source", choices=("region", "grid6"), default="region")
    parser.add_argument("--grid-voxel-size", type=float, default=0.03)
    parser.add_argument("--min-patch-voxels", type=int, default=40)
    parser.add_argument("--min-shared-edges", type=int, default=3)
    parser.add_argument("--min-contact-ratio", type=float, default=0.006)
    parser.add_argument("--max-bbox-gap", type=float, default=0.20)
    parser.add_argument("--max-color-distance", type=float, default=105.0)
    parser.add_argument("--min-normal-score", type=float, default=0.42)
    parser.add_argument("--min-bucket-score", type=float, default=0.42)
    parser.add_argument("--min-score", type=float, default=0.54)
    parser.add_argument("--contact-ratio-norm", type=float, default=0.18)
    parser.add_argument("--max-candidates", type=int, default=50000)
    parser.add_argument("--min-structural-score", type=float, default=0.70)
    parser.add_argument("--structural-min-contact-ratio", type=float, default=0.025)
    parser.add_argument("--structural-min-shared-edges", type=int, default=12)
    parser.add_argument("--structural-min-normal-score", type=float, default=0.56)
    parser.add_argument("--structural-max-bbox-gap", type=float, default=0.10)

    parser.add_argument("--preview-stride", type=int, default=10)
    parser.add_argument("--accept-min-score", type=float, default=0.80)
    parser.add_argument("--accept-min-contact-ratio", type=float, default=0.08)
    parser.add_argument("--accept-min-shared-edges", type=int, default=32)
    parser.add_argument("--accept-max-color-distance", type=float, default=55.0)
    parser.add_argument("--accept-max-bbox-gap", type=float, default=0.08)
    parser.add_argument("--accept-min-normal-score", type=float, default=0.65)
    parser.add_argument("--accept-min-structural-score", type=float, default=0.74)
    parser.add_argument("--accept-structural-min-contact-ratio", type=float, default=0.035)
    parser.add_argument("--accept-structural-min-shared-edges", type=int, default=24)
    parser.add_argument("--accept-structural-min-normal-score", type=float, default=0.58)
    parser.add_argument("--accept-structural-max-bbox-gap", type=float, default=0.08)
    parser.add_argument("--attachment-min-score", type=float, default=0.82)
    parser.add_argument("--attachment-min-contact-ratio", type=float, default=0.16)
    parser.add_argument("--attachment-min-shared-edges", type=int, default=48)
    parser.add_argument("--attachment-max-color-distance", type=float, default=38.0)
    parser.add_argument("--attachment-min-normal-score", type=float, default=0.65)
    parser.add_argument("--attachment-max-bbox-gap", type=float, default=0.06)
    parser.add_argument("--attachment-max-fragment-voxels", type=int, default=1200)
    parser.add_argument("--attachment-min-anchor-voxels", type=int, default=100000)
    parser.add_argument("--attachment-min-size-ratio", type=float, default=500.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.patch_labels is None:
        state = read_state(args.state)
        args.patch_labels = default_patch_labels(state)

    existing_file(args.region_input, "region input")
    existing_file(args.patch_labels, "patch labels")
    reject_forbidden_path(args.output_dir)

    plan = build_commands(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.plan_json or (args.output_dir / "dense_patch_object_refinement_v7_plan.json")
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan, ensure_ascii=False, indent=2))

    if args.run:
        cwd = Path.cwd()
        for item in plan["commands"]:
            run_command([str(part) for part in item["argv"]], cwd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
