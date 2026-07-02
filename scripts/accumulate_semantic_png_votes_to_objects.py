#!/usr/bin/env python3
"""Accumulate SKYMASK/SAM semantic PNG votes onto geometry-first objects.

This stage does not change point/object ownership.  It projects an existing
viewer PLY into calibrated camera poses, samples semantic PNGs after z-buffer
visibility filtering, aggregates votes per object id, applies simple geometry
guards, and writes a relabeled object JSONL plus semantic-color viewer PLY.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ModuleNotFoundError as exc:
    raise SystemExit("OpenCV is required for semantic PNG voting") from exc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

try:
    from scripts.geometry_input_contract import is_geometry_only_row
    from scripts.current_mainline_contract import reject_forbidden_production_input
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    from geometry_input_contract import is_geometry_only_row
    from current_mainline_contract import reject_forbidden_production_input

LABEL_NAMES = {
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
    18: "stair",
    19: "indoor_floor",
    20: "roof",
    255: "ignore",
}
LABEL_IDS = {v: k for k, v in LABEL_NAMES.items()}

SEMANTIC_COLORS = {
    0: (150, 150, 150),
    1: (180, 180, 180),
    2: (120, 150, 180),
    3: (196, 168, 112),
    4: (170, 170, 210),
    5: (80, 160, 80),
    6: (50, 130, 70),
    7: (235, 80, 80),
    8: (235, 90, 80),
    9: (240, 210, 60),
    10: (145, 145, 160),
    12: (120, 120, 120),
    14: (150, 110, 80),
    15: (220, 160, 60),
    16: (210, 90, 210),
    17: (245, 150, 40),
    18: (245, 125, 60),
    19: (105, 180, 210),
    20: (165, 145, 210),
}

SKIP_LABELS = {"unknown", "sky", "ignore", "water"}
HORIZONTAL_LABELS = {"floor", "ground", "road", "grass", "stair", "indoor_floor", "roof", "other"}
VERTICAL_LABELS = {"wall", "building", "railing", "pipe", "equipment", "other"}
THIN_LABELS = {"railing", "pipe", "equipment", "tree", "other"}
ROUGH_LABELS = {"car", "tree", "grass", "railing", "pipe", "equipment", "wall", "building", "other", "furniture"}
GEOMETRY_LABELS = {"horizontal", "vertical", "thin_linear", "rough_mixed", "mixed"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def semantic_path(base: Path, combo: str, cam_id: int, frame_id: int) -> Path:
    return base / "images" / f"cam{cam_id}_{frame_id:06d}" / combo / "semantic.png"


def frames_with_semantic(base: Path, combo: str, cams: list[int]) -> list[int]:
    frames = set()
    images_dir = base / "images"
    for cam_id in cams:
        for sem_path in images_dir.glob(f"cam{cam_id}_*/{combo}/semantic.png"):
            try:
                frames.add(int(sem_path.parent.parent.name.rsplit("_", 1)[1]))
            except (IndexError, ValueError):
                continue
    return sorted(frames)


def read_ply(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    header: list[str] = []
    props: list[str] = []
    in_vertex = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            header.append(line)
            parts = line.strip().split()
            if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
                in_vertex = True
            elif len(parts) >= 2 and parts[0] == "element":
                in_vertex = False
            elif in_vertex and len(parts) >= 3 and parts[0] == "property":
                props.append(parts[-1])
            elif line.strip() == "end_header":
                break
    data = np.loadtxt(path, skiprows=len(header), dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return header, props, data


def object_key(row: dict[str, Any]) -> int | None:
    for key in ("viewer_object_id", "object_id"):
        value = row.get(key)
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def object_geometry(row: dict[str, Any]) -> str:
    return str(row.get("geometry_type") or row.get("object_type_geometry") or "unknown")


def label_allowed(label: str, geometry_type: str) -> bool:
    if label in SKIP_LABELS:
        return False
    if geometry_type == "horizontal":
        return label in HORIZONTAL_LABELS
    if geometry_type == "vertical":
        return label in VERTICAL_LABELS
    if geometry_type == "thin_linear":
        return label in THIN_LABELS
    if geometry_type in {"rough_mixed", "mixed", "unknown"}:
        return label in ROUGH_LABELS or label in HORIZONTAL_LABELS or label in VERTICAL_LABELS
    return True


def fallback_label(geometry_type: str) -> str:
    if geometry_type == "horizontal":
        return "floor"
    if geometry_type == "vertical":
        return "wall"
    if geometry_type == "thin_linear":
        return "railing"
    return "unknown"


def normalized_original_label(row: dict[str, Any]) -> str:
    geometry_type = object_geometry(row)
    original = str(row.get("semantic_label") or "").strip()
    if is_geometry_only_row(row):
        return "unknown"
    if not original or original in GEOMETRY_LABELS:
        return fallback_label(geometry_type)
    return original


def transform_world_to_camera(points: np.ndarray, pose: dict[str, Any], cam_id: int) -> tuple[np.ndarray, np.ndarray]:
    T = pose["T_world_robot"]
    R_rw = T[:3, :3]
    t_rw = T[:3, 3]
    R_wr = R_rw.T
    t_wr = (-R_wr @ t_rw).reshape(3)
    R_li = config.Til[:3, :3].T
    t_li = (-R_li @ config.Til[:3, 3]).reshape(3)
    p_robot = (R_wr @ points.T + t_wr.reshape(3, 1)).T
    p_lidar = (R_li @ p_robot.T + t_li.reshape(3, 1)).T
    t_cl = config.Tcl[cam_id]
    p_cam = (t_cl[:3, :3] @ p_lidar.T + t_cl[:3, 3:]).T
    z = p_cam[:, 2]
    uv_h = (config.CAMERA_PARAMS[cam_id]["K"] @ p_cam.T).T
    uv = np.column_stack([uv_h[:, 0] / np.maximum(uv_h[:, 2], 1e-9), uv_h[:, 1] / np.maximum(uv_h[:, 2], 1e-9)])
    return uv.astype(np.float32), z.astype(np.float32)


def zbuffer_visible(point_indices: np.ndarray, uu: np.ndarray, vv: np.ndarray, depth: np.ndarray, width: int) -> np.ndarray:
    pixel_idx = vv.astype(np.int64) * int(width) + uu.astype(np.int64)
    order = np.lexsort((depth, pixel_idx))
    sorted_pixels = pixel_idx[order]
    first = np.r_[True, sorted_pixels[1:] != sorted_pixels[:-1]]
    keep_order = order[first]
    keep = np.zeros(len(point_indices), dtype=bool)
    keep[keep_order] = True
    return keep


def accumulate_votes(points: np.ndarray, object_ids: np.ndarray, object_rows: dict[int, dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    frame_ids = args.frames if args.frames else frames_with_semantic(args.semantic_eval_dir, args.combo, args.cams)
    if args.max_frames and len(frame_ids) > args.max_frames:
        step = max(len(frame_ids) / float(args.max_frames), 1.0)
        frame_ids = [frame_ids[int(round(i * step))] for i in range(args.max_frames) if int(round(i * step)) < len(frame_ids)]
        frame_ids = sorted(set(frame_ids))

    pose_map = {int(p["frame_id"]): p for p in config.load_img_pos(min(frame_ids or [0]), max(frame_ids or [0]))}
    votes: dict[int, Counter[str]] = defaultdict(Counter)
    vetoes: dict[int, Counter[str]] = defaultdict(Counter)
    frame_reports = []

    for frame_id in frame_ids:
        pose = pose_map.get(int(frame_id))
        if pose is None:
            frame_reports.append({"frame_id": frame_id, "status": "missing_pose"})
            continue
        for cam_id in args.cams:
            sem_path = semantic_path(args.semantic_eval_dir, args.combo, cam_id, int(frame_id))
            if not sem_path.exists():
                continue
            sem = cv2.imread(str(sem_path), cv2.IMREAD_GRAYSCALE)
            if sem is None:
                continue
            h, w = sem.shape[:2]
            uv, depth = transform_world_to_camera(points, pose, cam_id)
            valid = (
                (depth > args.min_depth)
                & (uv[:, 0] >= 0)
                & (uv[:, 0] < w)
                & (uv[:, 1] >= 0)
                & (uv[:, 1] < h)
            )
            if not np.any(valid):
                frame_reports.append({"frame_id": frame_id, "cam_id": cam_id, "status": "no_projection"})
                continue
            idx = np.where(valid)[0]
            uu = np.clip(np.rint(uv[idx, 0]).astype(np.int32), 0, w - 1)
            vv = np.clip(np.rint(uv[idx, 1]).astype(np.int32), 0, h - 1)
            d = depth[idx]
            if args.zbuffer:
                keep = zbuffer_visible(idx, uu, vv, d, w)
                idx, uu, vv, d = idx[keep], uu[keep], vv[keep], d[keep]
            sampled = sem[vv, uu].astype(np.uint8)
            used = 0
            for oid, sid in zip(object_ids[idx], sampled):
                label = LABEL_NAMES.get(int(sid), "unknown")
                if label in SKIP_LABELS:
                    continue
                row = object_rows.get(int(oid))
                if row is None:
                    continue
                geometry_type = object_geometry(row)
                if label_allowed(label, geometry_type):
                    votes[int(oid)][label] += 1
                    used += 1
                else:
                    vetoes[int(oid)][label] += 1
            frame_reports.append(
                {
                    "frame_id": int(frame_id),
                    "cam_id": int(cam_id),
                    "status": "ok",
                    "projected": int(len(idx)),
                    "used_votes": int(used),
                }
            )

    return {
        "frame_ids": frame_ids,
        "votes": votes,
        "vetoes": vetoes,
        "frame_reports": frame_reports,
    }


def apply_votes(objects: list[dict[str, Any]], vote_data: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    votes: dict[int, Counter[str]] = vote_data["votes"]
    vetoes: dict[int, Counter[str]] = vote_data["vetoes"]
    updated = []
    counts = Counter()
    changed = 0
    for row in objects:
        out = dict(row)
        oid = object_key(out)
        geometry_type = object_geometry(out)
        obj_votes = votes.get(int(oid), Counter()) if oid is not None else Counter()
        total = sum(obj_votes.values())
        original = normalized_original_label(out)
        if total >= args.min_votes:
            label, count = obj_votes.most_common(1)[0]
            ratio = count / max(total, 1)
            if ratio >= args.min_vote_ratio:
                out["semantic_label_original"] = original
                out["semantic_label"] = label
                out["semantic_vote_status"] = "sam_vote_applied"
                out["semantic_vote_ratio"] = float(ratio)
                if label != original:
                    changed += 1
            else:
                out["semantic_label"] = original
                out["semantic_vote_status"] = "sam_vote_ambiguous"
                out["semantic_vote_ratio"] = float(ratio)
        else:
            out["semantic_label"] = original if original not in {"rough_mixed", "mixed", "horizontal", "vertical", "thin_linear"} else fallback_label(geometry_type)
            out["semantic_vote_status"] = "insufficient_sam_votes"
            out["semantic_vote_ratio"] = 0.0
        out["semantic_votes"] = dict(obj_votes.most_common(12))
        out["semantic_veto_votes"] = dict(vetoes.get(int(oid), Counter()).most_common(12)) if oid is not None else {}
        counts[str(out.get("semantic_label") or "unknown")] += 1
        updated.append(out)
    report = {
        "object_count": len(objects),
        "changed_object_count": int(changed),
        "label_counts": dict(counts),
    }
    return updated, report


def write_semantic_ply(source_ply: Path, output_ply: Path, labels_by_object: dict[int, str]) -> dict[str, Any]:
    header, props, _data = read_ply(source_ply)
    idx = {name: i for i, name in enumerate(props)}
    required = {"red", "green", "blue", "object", "semantic"}
    missing = required - set(idx)
    if missing:
        raise ValueError(f"PLY missing required fields: {sorted(missing)}")
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    label_counts = Counter()
    rows = 0
    with source_ply.open("r", encoding="utf-8", errors="replace") as src, output_ply.open("w", encoding="utf-8") as dst:
        for line in header:
            dst.write(line)
        for _ in header:
            next(src)
        for line in src:
            if not line.strip():
                continue
            parts = line.strip().split()
            oid = int(float(parts[idx["object"]]))
            label = labels_by_object.get(oid, "unknown")
            sid = LABEL_IDS.get(label, 0)
            color = SEMANTIC_COLORS.get(sid, SEMANTIC_COLORS[0])
            parts[idx["red"]] = str(color[0])
            parts[idx["green"]] = str(color[1])
            parts[idx["blue"]] = str(color[2])
            parts[idx["semantic"]] = str(sid)
            dst.write(" ".join(parts) + "\n")
            label_counts[label] += 1
            rows += 1
    return {"rows": rows, "point_label_counts": dict(label_counts)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ply", type=Path, required=True)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--semantic-eval-dir", type=Path, required=True)
    parser.add_argument("--combo", default="sam2_prompt_v3_sky_label_merge_completion")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", default="objects_sam_semantic")
    parser.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--frames", type=int, nargs="*", default=[])
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--min-depth", type=float, default=0.3)
    parser.add_argument("--zbuffer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-votes", type=int, default=12)
    parser.add_argument("--min-vote-ratio", type=float, default=0.55)
    args = parser.parse_args()

    reject_forbidden_production_input(args.source_ply)
    reject_forbidden_production_input(args.objects_jsonl)
    reject_forbidden_production_input(args.output_dir)
    _header, props, data = read_ply(args.source_ply)
    idx = {name: i for i, name in enumerate(props)}
    for field in ("x", "y", "z", "object"):
        if field not in idx:
            raise ValueError(f"PLY missing {field}: {args.source_ply}")
    points = data[:, [idx["x"], idx["y"], idx["z"]]].astype(np.float64)
    object_ids = data[:, idx["object"]].astype(np.int64)
    objects = read_jsonl(args.objects_jsonl)
    object_rows = {int(k): row for row in objects if (k := object_key(row)) is not None}

    vote_data = accumulate_votes(points, object_ids, object_rows, args)
    updated, object_report = apply_votes(objects, vote_data, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    objects_out = args.output_dir / f"{args.output_stem}.jsonl"
    ply_out = args.output_dir / f"{args.output_stem}.ply"
    report_out = args.output_dir / f"{args.output_stem}_report.json"
    write_jsonl(objects_out, updated)
    labels_by_object = {int(k): str(row.get("semantic_label") or "unknown") for row in updated if (k := object_key(row)) is not None}
    ply_report = write_semantic_ply(args.source_ply, ply_out, labels_by_object)
    report = {
        "schema": "semantic-png-object-votes/v1",
        "source_ply": str(args.source_ply),
        "objects_jsonl": str(args.objects_jsonl),
        "semantic_eval_dir": str(args.semantic_eval_dir),
        "combo": args.combo,
        "output_objects_jsonl": str(objects_out),
        "output_ply": str(ply_out),
        "frame_count": len(vote_data["frame_ids"]),
        "frame_ids": vote_data["frame_ids"],
        "object_report": object_report,
        "ply_report": ply_report,
        "frame_reports": vote_data["frame_reports"],
        "params": vars(args),
    }
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
