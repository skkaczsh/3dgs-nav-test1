#!/usr/bin/env python3
"""Build depth / edge / semantic-prior guidance maps for camera frames.

This is the image-side counterpart of the surface-first 3D route.  It projects
either the per-frame `.lx` section or a fused global colored PLY into each
undistorted camera image using the validated MANIFOLD calibration chain, then
writes compact guidance artifacts:

- depth map with z-buffer nearest point
- rendered global point-cloud RGB map when a colored PLY is provided
- local point-index map
- depth-edge map
- color-edge map when a rendered RGB map is available
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


PLY_DTYPE = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def frame_path(base: Path, cam_id: int, frame_id: int) -> Path:
    return base / f"cam{cam_id}" / f"frame_{frame_id:06d}.jpg"


def parse_ply_header(path: Path) -> tuple[str, list[tuple[str, str]], int, int]:
    fmt = "ascii"
    props: list[tuple[str, str]] = []
    vertex_count = 0
    header_bytes = 0
    in_vertex = False
    with path.open("rb") as f:
        while True:
            raw = f.readline()
            if not raw:
                break
            header_bytes += len(raw)
            line = raw.decode("utf-8", errors="replace").strip()
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] == "format":
                fmt = parts[1]
            if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
                vertex_count = int(parts[2])
                in_vertex = True
            elif len(parts) >= 2 and parts[0] == "element":
                in_vertex = False
            elif in_vertex and len(parts) >= 3 and parts[0] == "property":
                props.append((parts[-2], parts[-1]))
            elif line.strip() == "end_header":
                break
    return fmt, props, vertex_count, header_bytes


def parse_ascii_ply_header(path: Path) -> tuple[list[str], int]:
    fmt, typed_props, vertex_count, _header_bytes = parse_ply_header(path)
    if fmt != "ascii":
        raise ValueError(f"Only ascii PLY is supported: {path}")
    props = [name for _ptype, name in typed_props]
    return props, vertex_count


def read_xyzrgb_ply_with_metadata(
    path: Path,
    max_points: int = 0,
    point_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    fmt, typed_props, vertex_count, header_bytes = parse_ply_header(path)
    names = [name for _ptype, name in typed_props]
    for name in ("x", "y", "z"):
        if name not in names:
            raise ValueError(f"PLY missing {name}: {path}")
    rgb_names = ("red", "green", "blue")
    has_rgb = all(name in names for name in rgb_names)
    metadata_names = ("frame_min", "frame_max", "frame_mean", "frame_count")
    has_frame_metadata = all(name in names for name in metadata_names)
    stride = max(int(point_stride), 1)
    if fmt == "ascii":
        rows = []
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip() == "end_header":
                    break
            for i, line in enumerate(f):
                if i % stride:
                    continue
                parts = line.strip().split()
                if len(parts) >= len(names):
                    rows.append(parts)
                    if max_points and len(rows) >= max_points:
                        break
        if not rows:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
        data = np.asarray(rows, dtype=np.float64)
        points = data[:, [names.index("x"), names.index("y"), names.index("z")]].astype(np.float32)
        if has_rgb:
            colors = np.clip(data[:, [names.index("red"), names.index("green"), names.index("blue")]], 0, 255).astype(np.uint8)
        else:
            colors = np.zeros((len(points), 3), dtype=np.uint8)
        metadata: dict[str, np.ndarray] = {}
        if has_frame_metadata:
            metadata = {
                "frame_min": data[:, names.index("frame_min")].astype(np.int32),
                "frame_max": data[:, names.index("frame_max")].astype(np.int32),
                "frame_mean": data[:, names.index("frame_mean")].astype(np.float32),
                "frame_count": data[:, names.index("frame_count")].astype(np.uint32),
            }
        return points, colors, metadata
    if fmt == "binary_little_endian":
        dtype = np.dtype([(name, PLY_DTYPE.get(ptype, "<f4")) for ptype, name in typed_props])
        with path.open("rb") as f:
            f.seek(header_bytes)
            data = np.fromfile(f, dtype=dtype, count=vertex_count)
        if stride > 1:
            data = data[::stride]
        if max_points:
            data = data[:max_points]
        points = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
        if has_rgb:
            colors = np.column_stack([data["red"], data["green"], data["blue"]]).astype(np.uint8)
        else:
            colors = np.zeros((len(points), 3), dtype=np.uint8)
        metadata = {}
        if has_frame_metadata:
            metadata = {
                "frame_min": data["frame_min"].astype(np.int32),
                "frame_max": data["frame_max"].astype(np.int32),
                "frame_mean": data["frame_mean"].astype(np.float32),
                "frame_count": data["frame_count"].astype(np.uint32),
            }
        return points, colors, metadata
    raise ValueError(f"Unsupported PLY format {fmt}: {path}")


def read_xyzrgb_ply(path: Path, max_points: int = 0, point_stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    points, colors, _metadata = read_xyzrgb_ply_with_metadata(path, max_points, point_stride)
    return points, colors


def source_frame_mask(metadata: dict[str, np.ndarray], frame_id: int, window: int, mode: str) -> np.ndarray | None:
    if window < 0 or mode == "none":
        return None
    if mode == "mean":
        frame_mean = metadata.get("frame_mean")
        if frame_mean is None:
            return None
        return np.abs(frame_mean.astype(np.float32) - float(frame_id)) <= float(window)
    frame_min = metadata.get("frame_min")
    frame_max = metadata.get("frame_max")
    if frame_min is None or frame_max is None:
        return None
    return (frame_min.astype(np.int32) <= frame_id + window) & (frame_max.astype(np.int32) >= frame_id - window)


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


def compute_color_edges(color_rgb: np.ndarray, valid: np.ndarray, threshold: float) -> np.ndarray:
    edge = np.zeros(valid.shape, dtype=np.uint8)
    if color_rgb.size == 0 or not np.any(valid):
        return edge
    lab = cv2.cvtColor(color_rgb[:, :, ::-1], cv2.COLOR_BGR2LAB).astype(np.float32)
    for axis in (0, 1):
        a = [slice(None), slice(None)]
        b = [slice(None), slice(None)]
        a[axis] = slice(1, None)
        b[axis] = slice(None, -1)
        a_t = tuple(a)
        b_t = tuple(b)
        both = valid[a_t] & valid[b_t]
        if not np.any(both):
            continue
        diff = np.linalg.norm(lab[a_t] - lab[b_t], axis=2) > threshold
        edge[a_t][both & diff] = 255
        edge[b_t][both & diff] = 255
    return edge


def local_min_depth(depth: np.ndarray, valid: np.ndarray, radius: int) -> np.ndarray:
    """Return local foreground depth, ignoring invalid pixels.

    A dense global cloud can render far surfaces through foreground sampling
    holes.  The local minimum is a conservative first-touch estimate for the
    current view: if a rendered pixel is much farther than the nearest depth in
    its neighborhood, it is probably evidence behind the visible surface.
    """
    if radius <= 0:
        out = np.full(depth.shape, np.inf, dtype=np.float32)
        out[valid] = depth[valid]
        return out
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    depth_for_erode = np.where(valid, depth, np.inf).astype(np.float32)
    return cv2.erode(depth_for_erode, kernel)


def continuity_support_count(depth: np.ndarray, valid: np.ndarray, radius: int, threshold: float) -> np.ndarray:
    """Count same-depth neighbors for a rendered pixel.

    This is the "strong echo" allowance: a non-nearest layer can survive if it
    forms a locally coherent surface instead of an isolated see-through speckle.
    The implementation is intentionally small-radius and explicit; map sizes are
    modest and this runs after z-buffering.
    """
    count = np.zeros(depth.shape, dtype=np.uint16)
    if radius <= 0 or threshold <= 0 or not np.any(valid):
        return count
    h, w = depth.shape
    for dy in range(-radius, radius + 1):
        y_src0 = max(0, -dy)
        y_src1 = min(h, h - dy)
        y_dst0 = max(0, dy)
        y_dst1 = min(h, h + dy)
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            x_src0 = max(0, -dx)
            x_src1 = min(w, w - dx)
            x_dst0 = max(0, dx)
            x_dst1 = min(w, w + dx)
            center_valid = valid[y_dst0:y_dst1, x_dst0:x_dst1]
            neighbor_valid = valid[y_src0:y_src1, x_src0:x_src1]
            if not np.any(center_valid & neighbor_valid):
                continue
            delta = np.abs(
                depth[y_dst0:y_dst1, x_dst0:x_dst1]
                - depth[y_src0:y_src1, x_src0:x_src1]
            )
            count[y_dst0:y_dst1, x_dst0:x_dst1] += (center_valid & neighbor_valid & (delta <= threshold)).astype(np.uint16)
    return count


def compute_view_surface_gate(
    depth: np.ndarray,
    valid: np.ndarray,
    mode: str,
    radius: int,
    first_touch_threshold: float,
    continuous_threshold: float,
    continuous_min_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter z-buffered pixels to the visible surface for this camera view."""
    if mode == "off" or not np.any(valid):
        support = np.zeros(depth.shape, dtype=np.uint16)
        return valid.copy(), depth.copy(), support
    near = local_min_depth(depth, valid, max(radius, 0))
    first_touch = valid & np.isfinite(near) & ((depth - near) <= first_touch_threshold)
    if mode == "first":
        support = np.zeros(depth.shape, dtype=np.uint16)
        return first_touch, near, support
    support = continuity_support_count(
        depth,
        valid,
        max(radius, 0),
        max(continuous_threshold, 0.0),
    )
    continuous = valid & (support >= int(continuous_min_neighbors))
    return first_touch | continuous, near, support


def fill_first_touch_holes(
    depth: np.ndarray,
    valid: np.ndarray,
    radius: int,
    depth_range_threshold: float,
    min_neighbors: int,
    min_mean_depth: float = 0.0,
    candidate_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill small holes inside the accepted first-touch surface layer.

    This is deliberately not a second-layer recovery.  A hole is filled only
    when nearby accepted depths are numerous and have a small depth range.  That
    recovers sparse LiDAR sampling holes on a visible surface while rejecting
    holes whose neighborhood mixes foreground and background depths.
    """
    if radius <= 0 or min_neighbors <= 0 or depth_range_threshold <= 0 or not np.any(valid):
        return depth, valid, np.zeros(depth.shape, dtype=bool)
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.float32)
    valid_float = valid.astype(np.float32)
    count = cv2.filter2D(valid_float, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    depth_sum = cv2.filter2D(np.where(valid, depth, 0.0).astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    mean_depth = depth_sum / np.maximum(count, 1.0)
    local_min = cv2.erode(np.where(valid, depth, np.inf).astype(np.float32), kernel.astype(np.uint8))
    local_max = cv2.dilate(np.where(valid, depth, -np.inf).astype(np.float32), kernel.astype(np.uint8))
    fill = (
        (~valid)
        & (count >= float(min_neighbors))
        & np.isfinite(local_min)
        & np.isfinite(local_max)
        & ((local_max - local_min) <= float(depth_range_threshold))
    )
    if min_mean_depth > 0:
        fill &= mean_depth >= float(min_mean_depth)
    if candidate_mask is not None:
        fill &= candidate_mask
    if not np.any(fill):
        return depth, valid, fill
    filled_depth = depth.copy()
    filled_valid = valid.copy()
    filled_depth[fill] = mean_depth[fill]
    filled_valid[fill] = True
    return filled_depth, filled_valid, fill


def splat_visible_surface(
    depth: np.ndarray,
    point_index: np.ndarray,
    semantic: np.ndarray,
    rendered_rgb: np.ndarray,
    valid: np.ndarray,
    image_bgr: np.ndarray,
    radius: int,
    color_lab_threshold: float,
    far_depth_start: float,
    far_radius: int,
    far_color_lab_threshold: float,
    candidate_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expand accepted first-touch samples into adjacent image pixels.

    Projected voxel centers are sparse even when the underlying cloud is dense.
    This splat is intentionally conservative: it only fills currently empty
    pixels from an accepted surface sample, and only when the source/target image
    colors are close.  It increases usable depth coverage without accepting a
    separate background layer.
    """
    filled = np.zeros(valid.shape, dtype=bool)
    max_radius = max(int(radius), int(far_radius) if far_depth_start > 0 else 0)
    if max_radius <= 0 or color_lab_threshold <= 0 or not np.any(valid):
        return depth, point_index, semantic, rendered_rgb, valid, filled
    h, w = valid.shape
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    out_depth = depth.copy()
    out_point_index = point_index.copy()
    out_semantic = semantic.copy()
    out_rgb = rendered_rgb.copy()
    out_valid = valid.copy()
    base_valid = valid.copy()
    offsets = [
        (dy, dx)
        for dy in range(-max_radius, max_radius + 1)
        for dx in range(-max_radius, max_radius + 1)
        if not (dy == 0 and dx == 0) and (dy * dy + dx * dx) <= max_radius * max_radius
    ]
    offsets.sort(key=lambda item: item[0] * item[0] + item[1] * item[1])
    for dy, dx in offsets:
        y_src0 = max(0, -dy)
        y_src1 = min(h, h - dy)
        y_dst0 = max(0, dy)
        y_dst1 = min(h, h + dy)
        x_src0 = max(0, -dx)
        x_src1 = min(w, w - dx)
        x_dst0 = max(0, dx)
        x_dst1 = min(w, w + dx)
        src_yx = (slice(y_src0, y_src1), slice(x_src0, x_src1))
        dst_yx = (slice(y_dst0, y_dst1), slice(x_dst0, x_dst1))
        src_valid = base_valid[src_yx]
        dst_empty = ~out_valid[dst_yx]
        if candidate_mask is not None:
            dst_empty = dst_empty & candidate_mask[dst_yx]
        if not np.any(src_valid & dst_empty):
            continue
        src_depth = depth[src_yx]
        offset_dist2 = float(dy * dy + dx * dx)
        if far_depth_start > 0 and far_radius > radius:
            far_src = src_depth >= float(far_depth_start)
            radius_ok = np.where(far_src, offset_dist2 <= float(far_radius * far_radius), offset_dist2 <= float(radius * radius))
            color_limit = np.where(far_src, float(far_color_lab_threshold), float(color_lab_threshold))
        else:
            radius_ok = offset_dist2 <= float(radius * radius)
            color_limit = float(color_lab_threshold)
        color_delta = np.linalg.norm(
            lab[src_yx] - lab[dst_yx],
            axis=2,
        )
        take = src_valid & dst_empty & radius_ok & (color_delta <= color_limit)
        if not np.any(take):
            continue
        dst_depth = out_depth[dst_yx]
        dst_point_index = out_point_index[dst_yx]
        dst_semantic = out_semantic[dst_yx]
        dst_rgb = out_rgb[dst_yx]
        dst_valid = out_valid[dst_yx]
        dst_filled = filled[dst_yx]
        dst_depth[take] = depth[src_yx][take]
        dst_point_index[take] = point_index[src_yx][take]
        dst_semantic[take] = semantic[src_yx][take]
        dst_rgb[take] = rendered_rgb[src_yx][take]
        dst_valid[take] = True
        dst_filled[take] = True
    return out_depth, out_point_index, out_semantic, out_rgb, out_valid, filled


def build_expansion_candidate_mask(image_bgr: np.ndarray, max_upper_ratio: float, sky_blue_guard: bool) -> np.ndarray:
    """Pixels eligible for synthetic surface expansion.

    Real first-touch samples are never removed here.  This only constrains
    splat/fill expansion so sparse depth cannot grow into sky-heavy regions.
    """
    h, w = image_bgr.shape[:2]
    mask = np.ones((h, w), dtype=bool)
    if max_upper_ratio > 0:
        top = int(round(h * max_upper_ratio))
        if top > 0:
            mask[:top, :] = False
    if sky_blue_guard:
        b = image_bgr[:, :, 0].astype(np.int16)
        g = image_bgr[:, :, 1].astype(np.int16)
        r = image_bgr[:, :, 2].astype(np.int16)
        blue_sky = (b > 90) & (b > r + 18) & (g > r + 8)
        mask &= ~blue_sky
    return mask


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
    colors_for_point: np.ndarray | None,
    cam_id: int,
    frame_id: int,
    image: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    h, w = image.shape[:2]
    depth = np.zeros((h, w), dtype=np.float32)
    local_point_index = np.full((h, w), -1, dtype=np.int32)
    semantic = np.zeros((h, w), dtype=np.uint8)
    rendered_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    valid_map = np.zeros((h, w), dtype=np.uint8)
    empty_support = np.zeros((h, w), dtype=np.uint16)

    def empty_result() -> dict[str, Any]:
        return {
            "depth": depth,
            "point_index": local_point_index,
            "semantic": semantic,
            "rendered_rgb": rendered_rgb,
            "edge": np.zeros((h, w), dtype=np.uint8),
            "color_edge": np.zeros((h, w), dtype=np.uint8),
            "valid": valid_map,
            "surface_near_depth": depth.copy(),
            "surface_support": empty_support,
            "surface_rejected": 0,
            "surface_splatted": 0,
            "surface_filled": 0,
            "surface_far_filled": 0,
            "visible": 0,
            "surface_visible": 0,
        }

    t_cl = config.Tcl[cam_id]
    p_cam = (t_cl[:3, :3] @ p_lidar.T + t_cl[:3, 3:]).T
    z = p_cam[:, 2]
    valid = z > args.min_depth
    if not np.any(valid):
        return empty_result()

    valid_idx = np.where(valid)[0]
    uv_h = (config.CAMERA_PARAMS[cam_id]["K"] @ p_cam[valid].T).T
    u = uv_h[:, 0] / uv_h[:, 2]
    v = uv_h[:, 1] / uv_h[:, 2]
    in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not np.any(in_img):
        return empty_result()

    idx = valid_idx[in_img]
    uu = np.clip(np.rint(u[in_img]).astype(np.int32), 0, w - 1)
    vv = np.clip(np.rint(v[in_img]).astype(np.int32), 0, h - 1)
    depths = z[valid][in_img].astype(np.float32)
    keep = zbuffer_visible(idx, uu, vv, depths, w)
    idx, uu, vv, depths = idx[keep], uu[keep], vv[keep], depths[keep]
    depth[vv, uu] = depths
    local_point_index[vv, uu] = idx.astype(np.int32)
    semantic[vv, uu] = semantic_for_point[idx]
    if colors_for_point is not None and len(colors_for_point) == len(points_world):
        rendered_rgb[vv, uu] = colors_for_point[idx]
    valid_map[vv, uu] = 255
    surface_valid, surface_near_depth, surface_support = compute_view_surface_gate(
        depth,
        valid_map > 0,
        args.view_surface_gate,
        args.view_surface_radius,
        args.view_surface_first_threshold,
        args.view_surface_continuous_threshold,
        args.view_surface_min_neighbors,
    )
    rejected = (valid_map > 0) & ~surface_valid
    if np.any(rejected):
        depth[rejected] = 0.0
        local_point_index[rejected] = -1
        semantic[rejected] = 0
        rendered_rgb[rejected] = 0
        valid_map[rejected] = 0
    expansion_candidate = build_expansion_candidate_mask(
        image,
        args.view_surface_expand_block_upper_ratio,
        args.view_surface_expand_sky_blue_guard,
    )
    valid_bool = valid_map > 0
    depth, local_point_index, semantic, rendered_rgb, splat_valid, splatted_surface = splat_visible_surface(
        depth,
        local_point_index,
        semantic,
        rendered_rgb,
        valid_bool,
        image,
        args.view_surface_splat_radius,
        args.view_surface_splat_color_lab_threshold,
        args.view_surface_far_depth_start,
        args.view_surface_far_splat_radius,
        args.view_surface_far_splat_color_lab_threshold,
        expansion_candidate,
    )
    if np.any(splatted_surface):
        valid_map[splatted_surface] = 255
    valid_bool = splat_valid
    depth, _filled_valid, filled_surface = fill_first_touch_holes(
        depth,
        valid_bool,
        args.view_surface_fill_radius,
        args.view_surface_fill_depth_range,
        args.view_surface_fill_min_neighbors,
        0.0,
        expansion_candidate,
    )
    if np.any(filled_surface):
        valid_map[filled_surface] = 255
    depth, _far_filled_valid, far_filled_surface = fill_first_touch_holes(
        depth,
        valid_map > 0,
        args.view_surface_far_fill_radius,
        args.view_surface_far_fill_depth_range,
        args.view_surface_far_fill_min_neighbors,
        args.view_surface_far_depth_start,
        expansion_candidate,
    )
    if np.any(far_filled_surface):
        valid_map[far_filled_surface] = 255
    edge = compute_depth_edges(depth, valid_map > 0, args.edge_depth_threshold, args.mark_invalid_boundary)
    color_edge = compute_color_edges(rendered_rgb, valid_map > 0, args.color_edge_lab_threshold)
    return {
        "depth": depth,
        "point_index": local_point_index,
        "semantic": semantic,
        "rendered_rgb": rendered_rgb,
        "edge": edge,
        "color_edge": color_edge,
        "valid": valid_map,
        "surface_near_depth": surface_near_depth,
        "surface_support": surface_support,
        "surface_rejected": int(np.count_nonzero(rejected)),
        "surface_splatted": int(np.count_nonzero(splatted_surface)),
        "surface_filled": int(np.count_nonzero(filled_surface)),
        "surface_far_filled": int(np.count_nonzero(far_filled_surface)),
        "visible": int(len(idx)),
        "surface_visible": int(np.count_nonzero(valid_map)),
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lx", type=Path)
    parser.add_argument("--global-colored-ply", type=Path, help="Fused or raw XYZ/RGB PLY to reverse-render dense depth/color guidance")
    parser.add_argument("--global-point-stride", type=int, default=1)
    parser.add_argument("--max-global-points", type=int, default=0)
    parser.add_argument(
        "--global-source-frame-window",
        type=int,
        default=20,
        help="When global PLY has frame metadata, keep only points observed within +/- this many frames of the image frame.",
    )
    parser.add_argument(
        "--global-source-filter-mode",
        choices=["none", "mean", "span"],
        default="mean",
        help="Source-frame filter for global PLY metadata. mean uses frame_mean; span uses frame_min/frame_max overlap.",
    )
    parser.add_argument(
        "--allow-unguarded-global",
        action="store_true",
        help="Allow full-global reverse projection without source-frame metadata/filtering. This is for diagnostics only.",
    )
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--semantic-prior-ply", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--frame-ids", type=int, nargs="*", default=None,
                        help="Optional explicit frame ids. When set, start/end/stride only bound pose loading.")
    parser.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument(
        "--view-surface-gate",
        choices=["off", "first", "first_or_continuous"],
        default="first",
        help="Reject see-through pixels after z-buffering. first is the production default; first_or_continuous is a diagnostic relaxed mode for coherent deeper layers.",
    )
    parser.add_argument("--view-surface-radius", type=int, default=6)
    parser.add_argument("--view-surface-first-threshold", type=float, default=0.12)
    parser.add_argument("--view-surface-continuous-threshold", type=float, default=0.18)
    parser.add_argument("--view-surface-min-neighbors", type=int, default=8)
    parser.add_argument("--view-surface-splat-radius", type=int, default=1)
    parser.add_argument("--view-surface-splat-color-lab-threshold", type=float, default=18.0)
    parser.add_argument("--view-surface-far-depth-start", type=float, default=0.0,
                        help="Enable diagnostic far-distance relaxation from this depth. Production default 0 disables it.")
    parser.add_argument("--view-surface-far-splat-radius", type=int, default=1)
    parser.add_argument("--view-surface-far-splat-color-lab-threshold", type=float, default=18.0)
    parser.add_argument("--view-surface-fill-radius", type=int, default=3)
    parser.add_argument("--view-surface-fill-depth-range", type=float, default=0.10)
    parser.add_argument("--view-surface-fill-min-neighbors", type=int, default=6)
    parser.add_argument("--view-surface-far-fill-radius", type=int, default=0)
    parser.add_argument("--view-surface-far-fill-depth-range", type=float, default=0.10)
    parser.add_argument("--view-surface-far-fill-min-neighbors", type=int, default=8)
    parser.add_argument("--view-surface-expand-block-upper-ratio", type=float, default=0.18)
    parser.add_argument("--view-surface-expand-sky-blue-guard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--edge-depth-threshold", type=float, default=0.35)
    parser.add_argument("--color-edge-lab-threshold", type=float, default=16.0)
    parser.add_argument("--mark-invalid-boundary", action="store_true")
    parser.add_argument("--prior-voxel-size", type=float, default=0.20)
    parser.add_argument("--prior-neighbor-radius", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--save-npz", action="store_true", default=True)
    return parser


def validate_global_source_guard(args: argparse.Namespace, global_metadata: dict[str, np.ndarray]) -> None:
    if not args.global_colored_ply:
        return
    if args.global_source_filter_mode == "none" and not args.allow_unguarded_global:
        raise SystemExit(
            "Refusing unguarded full-global reverse projection. "
            "Use --global-source-filter-mode mean/span with frame metadata, "
            "or pass --allow-unguarded-global for a diagnostic-only run."
        )
    if args.global_source_filter_mode != "none" and not global_metadata and not args.allow_unguarded_global:
        raise SystemExit(
            "Global PLY has no frame metadata, so source-frame filtering cannot be applied. "
            "Use a metadata PLY from build_raw_lx_voxel_cloud.py or pass "
            "--allow-unguarded-global for a diagnostic-only run."
        )
    if args.global_source_filter_mode != "none" and not global_metadata:
        print("warning: source filtering disabled because global PLY has no frame metadata", file=sys.stderr, flush=True)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    t0 = time.time()
    if not args.lx and not args.global_colored_ply:
        raise SystemExit("Provide either --lx for per-frame guidance or --global-colored-ply for dense reverse-render guidance.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("maps", "depth_viz", "depth_edge", "rendered_rgb", "color_edge", "semantic_prior"):
        (args.output_dir / name).mkdir(exist_ok=True)

    prior = build_semantic_prior(args.semantic_prior_ply, args.prior_voxel_size)
    pose_start = min(args.frame_ids) if args.frame_ids else args.start
    pose_end = max(args.frame_ids) if args.frame_ids else args.end
    poses = {row["frame_id"]: row for row in config.load_img_pos(pose_start, pose_end)}
    if args.global_colored_ply:
        global_points, global_colors, global_metadata = read_xyzrgb_ply_with_metadata(
            args.global_colored_ply,
            args.max_global_points,
            args.global_point_stride,
        )
        if len(global_points) == 0:
            raise SystemExit(f"No points loaded from {args.global_colored_ply}")
        validate_global_source_guard(args, global_metadata)
        sections = []
        if args.frame_ids:
            frame_ids = [int(i) for i in args.frame_ids if int(i) in poses]
        else:
            frame_ids = [i for i in range(args.start, args.end + 1, max(args.stride, 1)) if i in poses]
    else:
        global_points = np.empty((0, 3), dtype=np.float32)
        global_colors = np.empty((0, 3), dtype=np.uint8)
        global_metadata: dict[str, np.ndarray] = {}
        sections = read_lx_sections(args.lx)
        if args.frame_ids:
            frame_ids = [int(i) for i in args.frame_ids if int(i) < len(sections) and int(i) in poses]
        else:
            frame_ids = [i for i in range(args.start, args.end + 1, max(args.stride, 1)) if i < len(sections) and i in poses]
    if args.max_frames:
        frame_ids = frame_ids[: args.max_frames]
    if not frame_ids:
        raise SystemExit("No overlapping .lx sections, img_pos rows, and frame range.")

    rows: list[dict[str, Any]] = []
    contact_paths: list[Path] = []
    with (args.lx.open("rb") if args.lx and not args.global_colored_ply else open(os.devnull, "rb")) as lx_f:
        for frame_id in frame_ids:
            if args.global_colored_ply:
                mask = source_frame_mask(
                    global_metadata,
                    frame_id,
                    args.global_source_frame_window,
                    args.global_source_filter_mode,
                )
                if mask is not None:
                    points = global_points[mask]
                    colors_for_point: np.ndarray | None = global_colors[mask]
                    source_kept = int(np.count_nonzero(mask))
                else:
                    points = global_points
                    colors_for_point = global_colors
                    source_kept = int(len(points))
            else:
                points = read_lx_points(lx_f, sections[frame_id])
                colors_for_point = None
                source_kept = int(len(points))
            semantic_for_point = query_semantic_prior(points, prior, args.prior_voxel_size, args.prior_neighbor_radius)
            pose = poses[frame_id]
            p_lidar = transform_world_to_lidar(points, pose)
            for cam_id in args.cams:
                img_path = frame_path(args.frame_root, cam_id, frame_id)
                image = cv2.imread(str(img_path))
                if image is None:
                    rows.append({"frame_id": frame_id, "cam_id": cam_id, "status": "missing_image", "image_path": str(img_path)})
                    continue
                out = project_one_camera(points, p_lidar, semantic_for_point, colors_for_point, cam_id, frame_id, image, args)
                image_id = f"cam{cam_id}_{frame_id:06d}"
                npz_path = args.output_dir / "maps" / f"{image_id}_geometry.npz"
                depth_viz_path = args.output_dir / "depth_viz" / f"{image_id}_depth.jpg"
                edge_path = args.output_dir / "depth_edge" / f"{image_id}_edge.png"
                rendered_rgb_path = args.output_dir / "rendered_rgb" / f"{image_id}_rendered_rgb.jpg"
                color_edge_path = args.output_dir / "color_edge" / f"{image_id}_color_edge.png"
                semantic_path = args.output_dir / "semantic_prior" / f"{image_id}_semantic_prior.png"
                if args.save_npz:
                    np.savez_compressed(
                        npz_path,
                        depth=out["depth"],
                        point_index=out["point_index"],
                        semantic=out["semantic"],
                        rendered_rgb=out["rendered_rgb"],
                        edge=out["edge"],
                        color_edge=out["color_edge"],
                        valid=out["valid"],
                        surface_near_depth=out["surface_near_depth"],
                        surface_support=out["surface_support"],
                    )
                cv2.imwrite(str(depth_viz_path), depth_to_viz(out["depth"], out["valid"] > 0))
                cv2.imwrite(str(edge_path), out["edge"])
                cv2.imwrite(str(rendered_rgb_path), out["rendered_rgb"][:, :, ::-1])
                cv2.imwrite(str(color_edge_path), out["color_edge"])
                cv2.imwrite(str(semantic_path), semantic_to_rgb(out["semantic"])[:, :, ::-1])
                if len(contact_paths) < 48:
                    contact_paths.extend([img_path, depth_viz_path, edge_path, rendered_rgb_path, color_edge_path, semantic_path])
                hist = Counter(int(x) for x in out["semantic"][out["valid"] > 0].tolist())
                rows.append({
                    "frame_id": frame_id,
                    "cam_id": cam_id,
                    "status": "ok",
                    "image_path": str(img_path),
                    "npz_path": str(npz_path),
                    "depth_viz_path": str(depth_viz_path),
                    "edge_path": str(edge_path),
                    "rendered_rgb_path": str(rendered_rgb_path),
                    "color_edge_path": str(color_edge_path),
                    "semantic_prior_path": str(semantic_path),
                    "raw_points": int(len(points)),
                    "source_points_kept": source_kept,
                    "visible_pixels": int(out["visible"]),
                    "surface_visible_pixels": int(out["surface_visible"]),
                    "surface_rejected_pixels": int(out["surface_rejected"]),
                    "surface_splatted_pixels": int(out["surface_splatted"]),
                    "surface_filled_pixels": int(out["surface_filled"]),
                    "surface_far_filled_pixels": int(out["surface_far_filled"]),
                    "semantic_prior_counts": {str(k): int(v) for k, v in sorted(hist.items())},
                })

    status_counts = Counter(row["status"] for row in rows)
    report = {
        "lx": str(args.lx) if args.lx else "",
        "global_colored_ply": str(args.global_colored_ply) if args.global_colored_ply else "",
        "global_point_stride": args.global_point_stride,
        "max_global_points": args.max_global_points,
        "global_source_frame_window": args.global_source_frame_window,
        "global_source_filter_mode": args.global_source_filter_mode,
        "global_has_frame_metadata": bool(global_metadata) if args.global_colored_ply else False,
        "frame_root": str(args.frame_root),
        "semantic_prior_ply": str(args.semantic_prior_ply) if args.semantic_prior_ply else "",
        "output_dir": str(args.output_dir),
        "start": args.start,
        "end": args.end,
        "stride": args.stride,
        "frame_ids": frame_ids,
        "cams": args.cams,
        "prior_voxel_size": args.prior_voxel_size,
        "prior_neighbor_radius": args.prior_neighbor_radius,
        "prior_voxel_count": len(prior),
        "view_surface_gate": args.view_surface_gate,
        "view_surface_radius": args.view_surface_radius,
        "view_surface_first_threshold": args.view_surface_first_threshold,
        "view_surface_continuous_threshold": args.view_surface_continuous_threshold,
        "view_surface_min_neighbors": args.view_surface_min_neighbors,
        "view_surface_splat_radius": args.view_surface_splat_radius,
        "view_surface_splat_color_lab_threshold": args.view_surface_splat_color_lab_threshold,
        "view_surface_far_depth_start": args.view_surface_far_depth_start,
        "view_surface_far_splat_radius": args.view_surface_far_splat_radius,
        "view_surface_far_splat_color_lab_threshold": args.view_surface_far_splat_color_lab_threshold,
        "view_surface_fill_radius": args.view_surface_fill_radius,
        "view_surface_fill_depth_range": args.view_surface_fill_depth_range,
        "view_surface_fill_min_neighbors": args.view_surface_fill_min_neighbors,
        "view_surface_far_fill_radius": args.view_surface_far_fill_radius,
        "view_surface_far_fill_depth_range": args.view_surface_far_fill_depth_range,
        "view_surface_far_fill_min_neighbors": args.view_surface_far_fill_min_neighbors,
        "view_surface_expand_block_upper_ratio": args.view_surface_expand_block_upper_ratio,
        "view_surface_expand_sky_blue_guard": args.view_surface_expand_sky_blue_guard,
        "image_count": len(rows),
        "status_counts": dict(status_counts),
        "elapsed_sec": time.time() - t0,
        "items": rows,
    }
    (args.output_dir / "geometry_guidance_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_contact_sheet(contact_paths[:48], args.output_dir / "geometry_guidance_contact.jpg", cols=6)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "image_count": len(rows),
        "status_counts": dict(status_counts),
        "prior_voxel_count": len(prior),
        "elapsed_sec": report["elapsed_sec"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
