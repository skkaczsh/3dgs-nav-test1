#!/usr/bin/env python3
"""Build depth / edge / semantic-prior guidance maps for camera frames.

This is the image-side counterpart of the surface-first 3D route.  It projects
the per-frame `.lx` section into each undistorted camera image using the
validated MANIFOLD calibration chain, then writes compact guidance artifacts:

- depth map with z-buffer nearest point
- local point-index map
- depth-edge map
- semantic-prior map, queried from a trusted semantic PLY such as v19

The generated maps are intended to constrain SAM/DINO masks: model masks may
suggest candidates, but depth discontinuities and trusted surface priors decide
where boundaries can safely pass.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from project_priority_masks_to_lx import (
    read_lx_points,
    read_lx_sections,
    transform_world_to_lidar,
    zbuffer_visible,
)


LABEL_COLORS = {
    0: (90, 90, 90),
    2: (160, 170, 180),
    3: (190, 172, 135),
    4: (180, 180, 210),
    5: (70, 150, 80),
    8: (235, 90, 80),
    9: (245, 200, 35),
    17: (230, 55, 220),
}


def frame_path(base: Path, cam_id: int, frame_id: int) -> Path:
    return base / f"cam{cam_id}" / f"frame_{frame_id:06d}.jpg"


def parse_ascii_ply_header(path: Path) -> tuple[list[str], int]:
    props: list[str] = []
    vertex_count = 0
    in_vertex = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
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
    return props, vertex_count


def voxel_key(point: np.ndarray, voxel_size: float) -> tuple[int, int, int]:
    return tuple(np.floor(point / voxel_size).astype(np.int32).tolist())


def build_semantic_prior(ply_path: Path | None, voxel_size: float) -> dict[tuple[int, int, int], int]:
    if ply_path is None:
        return {}
    props, _vertex_count = parse_ascii_ply_header(ply_path)
    idx = {name: i for i, name in enumerate(props)}
    for name in ("x", "y", "z", "semantic"):
        if name not in idx:
            raise ValueError(f"PLY missing {name}: {ply_path}")
    votes: dict[tuple[int, int, int], Counter[int]] = defaultdict(Counter)
    with ply_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip() == "end_header":
                break
        for line in f:
            parts = line.strip().split()
            if len(parts) <= idx["semantic"]:
                continue
            p = np.array([float(parts[idx["x"]]), float(parts[idx["y"]]), float(parts[idx["z"]])], dtype=np.float32)
            sem = int(round(float(parts[idx["semantic"]])))
            votes[voxel_key(p, voxel_size)][sem] += 1
    prior = {key: counter.most_common(1)[0][0] for key, counter in votes.items()}
    return prior


def query_semantic_prior(
    points: np.ndarray,
    prior: dict[tuple[int, int, int], int],
    voxel_size: float,
    radius: int,
) -> np.ndarray:
    if not prior:
        return np.zeros(len(points), dtype=np.uint8)
    out = np.zeros(len(points), dtype=np.uint8)
    offsets = [
        (dx, dy, dz)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        for dz in range(-radius, radius + 1)
    ]
    offsets.sort(key=lambda o: o[0] * o[0] + o[1] * o[1] + o[2] * o[2])
    coords = np.floor(points / voxel_size).astype(np.int32)
    for i, base in enumerate(coords):
        counter: Counter[int] = Counter()
        bx, by, bz = [int(x) for x in base]
        for dx, dy, dz in offsets:
            sem = prior.get((bx + dx, by + dy, bz + dz))
            if sem is not None:
                # Near voxels count more than diagonal neighbors.
                weight = max(1, 4 - (dx * dx + dy * dy + dz * dz))
                counter[int(sem)] += weight
        if counter:
            out[i] = int(counter.most_common(1)[0][0])
    return out


def compute_depth_edges(depth: np.ndarray, valid: np.ndarray, threshold: float, mark_invalid_boundary: bool) -> np.ndarray:
    edge = np.zeros(valid.shape, dtype=np.uint8)
    for axis in (0, 1):
        a = [slice(None), slice(None)]
        b = [slice(None), slice(None)]
        a[axis] = slice(1, None)
        b[axis] = slice(None, -1)
        a_t = tuple(a)
        b_t = tuple(b)
        both = valid[a_t] & valid[b_t]
        diff = np.zeros_like(both, dtype=bool)
        diff[both] = np.abs(depth[a_t][both] - depth[b_t][both]) > threshold
        edge[a_t][diff] = 255
        edge[b_t][diff] = 255
        if mark_invalid_boundary:
            boundary = valid[a_t] ^ valid[b_t]
            edge[a_t][boundary & valid[a_t]] = 255
            edge[b_t][boundary & valid[b_t]] = 255
    return edge


def depth_to_viz(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    vals = depth[valid]
    lo = float(np.percentile(vals, 2))
    hi = float(np.percentile(vals, 98))
    if hi <= lo:
        hi = lo + 1.0
    norm = np.zeros(depth.shape, dtype=np.uint8)
    clipped = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    norm[valid] = (255.0 * (1.0 - clipped[valid])).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)


def semantic_to_rgb(semantic: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*semantic.shape, 3), dtype=np.uint8)
    for sem, color in LABEL_COLORS.items():
        rgb[semantic == sem] = color
    return rgb


def project_one_camera(
    points_world: np.ndarray,
    p_lidar: np.ndarray,
    semantic_for_point: np.ndarray,
    cam_id: int,
    frame_id: int,
    image: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    h, w = image.shape[:2]
    depth = np.zeros((h, w), dtype=np.float32)
    local_point_index = np.full((h, w), -1, dtype=np.int32)
    semantic = np.zeros((h, w), dtype=np.uint8)
    valid_map = np.zeros((h, w), dtype=np.uint8)

    t_cl = config.Tcl[cam_id]
    p_cam = (t_cl[:3, :3] @ p_lidar.T + t_cl[:3, 3:]).T
    z = p_cam[:, 2]
    valid = z > args.min_depth
    if not np.any(valid):
        edge = np.zeros((h, w), dtype=np.uint8)
        return {"depth": depth, "point_index": local_point_index, "semantic": semantic, "edge": edge, "valid": valid_map, "visible": 0}

    valid_idx = np.where(valid)[0]
    uv_h = (config.CAMERA_PARAMS[cam_id]["K"] @ p_cam[valid].T).T
    u = uv_h[:, 0] / uv_h[:, 2]
    v = uv_h[:, 1] / uv_h[:, 2]
    in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not np.any(in_img):
        edge = np.zeros((h, w), dtype=np.uint8)
        return {"depth": depth, "point_index": local_point_index, "semantic": semantic, "edge": edge, "valid": valid_map, "visible": 0}

    idx = valid_idx[in_img]
    uu = np.clip(np.rint(u[in_img]).astype(np.int32), 0, w - 1)
    vv = np.clip(np.rint(v[in_img]).astype(np.int32), 0, h - 1)
    depths = z[valid][in_img].astype(np.float32)
    keep = zbuffer_visible(idx, uu, vv, depths, w)
    idx, uu, vv, depths = idx[keep], uu[keep], vv[keep], depths[keep]
    depth[vv, uu] = depths
    local_point_index[vv, uu] = idx.astype(np.int32)
    semantic[vv, uu] = semantic_for_point[idx]
    valid_map[vv, uu] = 255
    edge = compute_depth_edges(depth, valid_map > 0, args.edge_depth_threshold, args.mark_invalid_boundary)
    return {
        "depth": depth,
        "point_index": local_point_index,
        "semantic": semantic,
        "edge": edge,
        "valid": valid_map,
        "visible": int(len(idx)),
    }


def write_contact_sheet(paths: list[Path], output: Path, cols: int = 4) -> None:
    images = [cv2.imread(str(path)) for path in paths if path.exists()]
    images = [img for img in images if img is not None]
    if not images:
        return
    thumb_w = 360
    thumbs = []
    for img in images:
        scale = thumb_w / img.shape[1]
        thumbs.append(cv2.resize(img, (thumb_w, max(1, int(img.shape[0] * scale)))))
    max_h = max(img.shape[0] for img in thumbs)
    padded = []
    for img in thumbs:
        if img.shape[0] < max_h:
            pad = np.zeros((max_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
            img = np.vstack([img, pad])
        padded.append(img)
    rows = []
    for i in range(0, len(padded), cols):
        row = padded[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(padded[0]))
        rows.append(np.hstack(row))
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.vstack(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lx", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--semantic-prior-ply", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--edge-depth-threshold", type=float, default=0.35)
    parser.add_argument("--mark-invalid-boundary", action="store_true")
    parser.add_argument("--prior-voxel-size", type=float, default=0.20)
    parser.add_argument("--prior-neighbor-radius", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--save-npz", action="store_true", default=True)
    args = parser.parse_args()

    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("maps", "depth_viz", "depth_edge", "semantic_prior"):
        (args.output_dir / name).mkdir(exist_ok=True)

    prior = build_semantic_prior(args.semantic_prior_ply, args.prior_voxel_size)
    sections = read_lx_sections(args.lx)
    poses = {row["frame_id"]: row for row in config.load_img_pos(args.start, args.end)}
    frame_ids = [i for i in range(args.start, args.end + 1, max(args.stride, 1)) if i < len(sections) and i in poses]
    if args.max_frames:
        frame_ids = frame_ids[: args.max_frames]
    if not frame_ids:
        raise SystemExit("No overlapping .lx sections, img_pos rows, and frame range.")

    rows: list[dict[str, Any]] = []
    contact_paths: list[Path] = []
    with args.lx.open("rb") as lx_f:
        for frame_id in frame_ids:
            points = read_lx_points(lx_f, sections[frame_id])
            semantic_for_point = query_semantic_prior(points, prior, args.prior_voxel_size, args.prior_neighbor_radius)
            pose = poses[frame_id]
            p_lidar = transform_world_to_lidar(points, pose)
            for cam_id in args.cams:
                img_path = frame_path(args.frame_root, cam_id, frame_id)
                image = cv2.imread(str(img_path))
                if image is None:
                    rows.append({"frame_id": frame_id, "cam_id": cam_id, "status": "missing_image", "image_path": str(img_path)})
                    continue
                out = project_one_camera(points, p_lidar, semantic_for_point, cam_id, frame_id, image, args)
                image_id = f"cam{cam_id}_{frame_id:06d}"
                npz_path = args.output_dir / "maps" / f"{image_id}_geometry.npz"
                depth_viz_path = args.output_dir / "depth_viz" / f"{image_id}_depth.jpg"
                edge_path = args.output_dir / "depth_edge" / f"{image_id}_edge.png"
                semantic_path = args.output_dir / "semantic_prior" / f"{image_id}_semantic_prior.png"
                if args.save_npz:
                    np.savez_compressed(
                        npz_path,
                        depth=out["depth"],
                        point_index=out["point_index"],
                        semantic=out["semantic"],
                        edge=out["edge"],
                        valid=out["valid"],
                    )
                cv2.imwrite(str(depth_viz_path), depth_to_viz(out["depth"], out["valid"] > 0))
                cv2.imwrite(str(edge_path), out["edge"])
                cv2.imwrite(str(semantic_path), semantic_to_rgb(out["semantic"])[:, :, ::-1])
                if len(contact_paths) < 48:
                    contact_paths.extend([depth_viz_path, edge_path, semantic_path])
                hist = Counter(int(x) for x in out["semantic"][out["valid"] > 0].tolist())
                rows.append({
                    "frame_id": frame_id,
                    "cam_id": cam_id,
                    "status": "ok",
                    "image_path": str(img_path),
                    "npz_path": str(npz_path),
                    "depth_viz_path": str(depth_viz_path),
                    "edge_path": str(edge_path),
                    "semantic_prior_path": str(semantic_path),
                    "raw_points": int(len(points)),
                    "visible_pixels": int(out["visible"]),
                    "semantic_prior_counts": {str(k): int(v) for k, v in sorted(hist.items())},
                })

    status_counts = Counter(row["status"] for row in rows)
    report = {
        "lx": str(args.lx),
        "frame_root": str(args.frame_root),
        "semantic_prior_ply": str(args.semantic_prior_ply) if args.semantic_prior_ply else "",
        "output_dir": str(args.output_dir),
        "start": args.start,
        "end": args.end,
        "stride": args.stride,
        "cams": args.cams,
        "prior_voxel_size": args.prior_voxel_size,
        "prior_neighbor_radius": args.prior_neighbor_radius,
        "prior_voxel_count": len(prior),
        "image_count": len(rows),
        "status_counts": dict(status_counts),
        "elapsed_sec": time.time() - t0,
        "items": rows,
    }
    (args.output_dir / "geometry_guidance_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_contact_sheet(contact_paths[:48], args.output_dir / "geometry_guidance_contact.jpg")
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "image_count": len(rows),
        "status_counts": dict(status_counts),
        "prior_voxel_count": len(prior),
        "elapsed_sec": report["elapsed_sec"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
