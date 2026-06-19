#!/usr/bin/env python3
"""Stream-colorize MANIFOLD .lx sections into merged colored point clouds.

This avoids materializing every `section_XXXX.ply` and every undistorted camera
frame on disk. It reads each .lx section, reads the synchronized video frame,
undistorts in memory, projects points through the validated Tcl/Til pose chain,
and writes:

- a binary PLY containing all successfully colored points
- an optional voxel-downsampled binary PLY for quick review
- a JSON report with coverage and bounds
"""

import argparse
import json
import os
import struct
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from sync_frame_map import load_frame_map, resolve_video_idx


LX_HEADER_SIZE = 48
LX_COUNT_SIZE = 4
LX_POINT_SIZE = 16


def read_lx_sections(lx_path):
    sections = []
    file_size = os.path.getsize(lx_path)
    offset = 0
    section_idx = 0
    with open(lx_path, "rb") as f:
        while offset + LX_HEADER_SIZE + LX_COUNT_SIZE <= file_size:
            f.seek(offset + LX_HEADER_SIZE)
            count_raw = f.read(LX_COUNT_SIZE)
            if len(count_raw) < LX_COUNT_SIZE:
                break
            count = struct.unpack("<I", count_raw)[0]
            if count == 0 or count > 50_000_000:
                break
            data_offset = offset + LX_HEADER_SIZE + LX_COUNT_SIZE
            next_offset = data_offset + count * LX_POINT_SIZE
            if next_offset > file_size + 16:
                break
            sections.append({
                "index": section_idx,
                "offset": offset,
                "data_offset": data_offset,
                "count": count,
            })
            offset = next_offset
            section_idx += 1
    return sections


def read_lx_points(f, section):
    f.seek(section["data_offset"])
    raw = f.read(section["count"] * LX_POINT_SIZE)
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("marker", "<u4"),
    ])
    data = np.frombuffer(raw, dtype=dtype)
    points = np.empty((len(data), 3), dtype=np.float32)
    points[:, 0] = data["x"]
    points[:, 1] = data["y"]
    points[:, 2] = data["z"]
    return points


def make_undistort_maps():
    maps = {}
    for cam_id, params in config.CAMERA_PARAMS.items():
        K = params["K"]
        D = params["D"]
        size = (config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
        maps[cam_id] = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), K, size, cv2.CV_16SC2
        )
    return maps


def open_video_caps():
    caps = {}
    for cam_id, video_path in config.VIDEO_FILES.items():
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video cam{cam_id}: {video_path}")
        caps[cam_id] = cap
    return caps


def read_video_frame(cap, frame_id):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def undistort(frame, maps, cam_id):
    map1, map2 = maps[cam_id]
    return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)


def heuristic_sky_mask(image_bgr, upper_ratio=0.72):
    """Conservative sky mask for undistorted outdoor frames.

    This is not a semantic sky model. It only removes common blue-sky and bright
    low-saturation sky pixels in the upper image region, so it is suitable as a
    color-contamination guard before dense semantic processing.
    """
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    y = np.arange(h, dtype=np.float32)[:, None]
    upper = y < (h * float(upper_ratio))
    blue_sky = (hue >= 85) & (hue <= 130) & (sat >= 25) & (val >= 95)
    bright_haze = (sat <= 65) & (val >= 185)
    mask = upper & (blue_sky | bright_haze)
    if np.any(mask):
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    return mask


def project_color_points(points_world, pose, images, cams, sky_filter="none", sky_upper_ratio=0.72):
    n = len(points_world)
    colors = np.zeros((n, 3), dtype=np.uint8)
    depths = np.full(n, np.inf, dtype=np.float32)

    T = pose["T_world_robot"]
    R_rw = T[:3, :3]
    t_rw = T[:3, 3]
    R_wr = R_rw.T
    t_wr = (-R_wr @ t_rw).reshape(3)

    R_li = config.Til[:3, :3].T
    t_li = (-R_li @ config.Til[:3, 3]).reshape(3)

    points64 = points_world.astype(np.float64, copy=False)
    P_robot = (R_wr @ points64.T + t_wr.reshape(3, 1)).T
    P_lidar = (R_li @ P_robot.T + t_li.reshape(3, 1)).T

    sky_masks = {}
    if sky_filter == "heuristic":
        for cam_id, img in images.items():
            sky_masks[cam_id] = None if img is None else heuristic_sky_mask(img, sky_upper_ratio)

    sky_rejected = 0
    for cam_id in cams:
        img = images.get(cam_id)
        if img is None:
            continue

        K = config.CAMERA_PARAMS[cam_id]["K"]
        T_cl = config.Tcl[cam_id]
        P_cam = (T_cl[:3, :3] @ P_lidar.T + T_cl[:3, 3:]).T

        z = P_cam[:, 2]
        valid = z > 0.1
        if not np.any(valid):
            continue

        valid_idx = np.where(valid)[0]
        uv_h = (K @ P_cam[valid].T).T
        u = uv_h[:, 0] / uv_h[:, 2]
        v = uv_h[:, 1] / uv_h[:, 2]
        h, w = img.shape[:2]
        in_img = (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        if not np.any(in_img):
            continue

        idx = valid_idx[in_img]
        u_v = u[in_img]
        v_v = v[in_img]
        z_v = z[valid][in_img]

        sky_mask = sky_masks.get(cam_id)
        if sky_mask is not None:
            sx = np.clip(u_v.astype(np.int32), 0, w - 1)
            sy = np.clip(v_v.astype(np.int32), 0, h - 1)
            non_sky = ~sky_mask[sy, sx]
            sky_rejected += int((~non_sky).sum())
            if not np.any(non_sky):
                continue
            idx = idx[non_sky]
            u_v = u_v[non_sky]
            v_v = v_v[non_sky]
            z_v = z_v[non_sky]

        u0 = u_v.astype(np.int32)
        v0 = v_v.astype(np.int32)
        su = u_v - u0
        sv = v_v - v0
        c00 = img[v0, u0]
        c10 = img[v0, u0 + 1]
        c01 = img[v0 + 1, u0]
        c11 = img[v0 + 1, u0 + 1]
        sampled = (
            c00 * (1 - su[:, None]) * (1 - sv[:, None]) +
            c10 * su[:, None] * (1 - sv[:, None]) +
            c01 * (1 - su[:, None]) * sv[:, None] +
            c11 * su[:, None] * sv[:, None]
        )
        sampled = np.clip(sampled, 0, 255).astype(np.uint8)[:, ::-1]

        better = z_v < depths[idx]
        if np.any(better):
            out_idx = idx[better]
            colors[out_idx] = sampled[better]
            depths[out_idx] = z_v[better].astype(np.float32)

    keep = np.isfinite(depths)
    return points_world[keep], colors[keep], int(keep.sum()), sky_rejected


def append_binary_xyzrgb(body_f, points, colors):
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    data = np.empty(len(points), dtype=dtype)
    data["x"] = points[:, 0]
    data["y"] = points[:, 1]
    data["z"] = points[:, 2]
    data["red"] = colors[:, 0]
    data["green"] = colors[:, 1]
    data["blue"] = colors[:, 2]
    body_f.write(data.tobytes())


def write_binary_ply_from_body(body_path, output_path, vertex_count):
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {vertex_count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with open(output_path, "wb") as out, open(body_path, "rb") as body:
        out.write(header)
        while True:
            chunk = body.read(16 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def update_bounds(bounds, pts):
    if len(pts) == 0:
        return bounds
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    if bounds is None:
        return {"min": mins.astype(float).tolist(), "max": maxs.astype(float).tolist()}
    bounds["min"] = np.minimum(np.asarray(bounds["min"]), mins).astype(float).tolist()
    bounds["max"] = np.maximum(np.asarray(bounds["max"]), maxs).astype(float).tolist()
    return bounds


def update_voxels(voxels, points, colors, voxel_size):
    if voxel_size <= 0 or len(points) == 0:
        return
    ijk = np.floor(points / voxel_size).astype(np.int64)
    dtype = np.dtype([("x", "<i8"), ("y", "<i8"), ("z", "<i8")])
    structured = np.empty(len(ijk), dtype=dtype)
    structured["x"] = ijk[:, 0]
    structured["y"] = ijk[:, 1]
    structured["z"] = ijk[:, 2]
    unique, inverse = np.unique(structured, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    sums = []
    for col in range(3):
        sums.append(np.bincount(inverse, weights=points[:, col]))
    for col in range(3):
        sums.append(np.bincount(inverse, weights=colors[:, col]))

    for i, key in enumerate(unique):
        k = (int(key["x"]), int(key["y"]), int(key["z"]))
        acc = voxels.get(k)
        vals = [arr[i] for arr in sums]
        if acc is None:
            voxels[k] = vals + [counts[i]]
        else:
            for j in range(6):
                acc[j] += vals[j]
            acc[6] += counts[i]


def write_voxel_ply(voxels, output_path):
    points = np.empty((len(voxels), 3), dtype=np.float32)
    colors = np.empty((len(voxels), 3), dtype=np.uint8)
    for i, acc in enumerate(voxels.values()):
        count = max(acc[6], 1.0)
        points[i] = [acc[0] / count, acc[1] / count, acc[2] / count]
        colors[i] = np.clip([acc[3] / count, acc[4] / count, acc[5] / count], 0, 255)
    with tempfile.NamedTemporaryFile(delete=False) as body:
        append_binary_xyzrgb(body, points, colors)
        body_path = body.name
    try:
        write_binary_ply_from_body(body_path, output_path, len(points))
    finally:
        os.remove(body_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lx-file", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--voxel-output", type=Path, default=None)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--skip-full-output", action="store_true")
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--frame-map-jsonl", type=Path, default=None,
                        help="Optional JSONL mapping with frame_id, cam_id, and video_idx/selected_video_idx.")
    parser.add_argument("--require-frame-map", action="store_true",
                        help="Fail image reads when --frame-map-jsonl has no row for a frame/cam pair.")
    parser.add_argument("--allow-rejected-frame-map", action="store_true",
                        help="Diagnostic only: allow rejected/unstable sync rows instead of failing fast.")
    parser.add_argument("--sky-filter", choices=["none", "heuristic"], default="none")
    parser.add_argument("--sky-upper-ratio", type=float, default=0.72)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()
    if args.require_frame_map and args.frame_map_jsonl is None:
        raise SystemExit("--frame-map-jsonl is required when --require-frame-map is used")

    t0 = time.time()
    sections = read_lx_sections(args.lx_file)
    poses = {p["frame_id"]: p for p in config.load_img_pos(args.start, args.end)}
    if args.end is None:
        args.end = min(len(sections), max(poses) + 1 if poses else 0) - 1

    target_ids = [i for i in range(args.start, args.end + 1, max(args.frame_step, 1))
                  if i < len(sections) and i in poses]
    if not target_ids:
        raise SystemExit("No overlapping .lx sections and img_pos poses.")

    print(f"lx_file={args.lx_file}")
    print(f"sections={len(sections)} target_frames={len(target_ids)} range={target_ids[0]}..{target_ids[-1]}")
    print(f"calib={config.CALIB_FILE}")
    print(f"image_dir={config.IMAGE_DIR}")
    print(f"video_dir={config.VIDEO_DIR}")
    frame_map = load_frame_map(args.frame_map_jsonl, allow_rejected=args.allow_rejected_frame_map)
    if args.frame_map_jsonl:
        print(
            f"frame_map={args.frame_map_jsonl} rows={len(frame_map)} "
            f"require={args.require_frame_map} allow_rejected={args.allow_rejected_frame_map}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.voxel_output:
        args.voxel_output.parent.mkdir(parents=True, exist_ok=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)

    maps = make_undistort_maps()
    caps = open_video_caps()
    voxels = {} if args.voxel_output else None
    bounds = None
    raw_points = 0
    colored_points = 0
    sky_rejected_points = 0
    failed_images = 0
    mapped_image_reads = 0
    direct_image_reads = 0
    missing_frame_map_reads = 0

    body_path = None
    body_f = None
    if not args.skip_full_output:
        body_f = tempfile.NamedTemporaryFile(delete=False)
        body_path = body_f.name

    try:
        with open(args.lx_file, "rb") as lx_f:
            for n, frame_id in enumerate(target_ids, 1):
                points = read_lx_points(lx_f, sections[frame_id])
                raw_points += len(points)

                images = {}
                for cam_id in args.cams:
                    video_idx = resolve_video_idx(
                        frame_map,
                        frame_id,
                        cam_id,
                        fallback_to_direct=not args.require_frame_map,
                    )
                    if video_idx is None:
                        failed_images += 1
                        missing_frame_map_reads += 1
                        images[cam_id] = None
                        continue
                    if video_idx == frame_id:
                        direct_image_reads += 1
                    else:
                        mapped_image_reads += 1
                    frame = read_video_frame(caps[cam_id], video_idx)
                    if frame is None:
                        failed_images += 1
                        images[cam_id] = None
                    else:
                        images[cam_id] = undistort(frame, maps, cam_id)

                pts_col, cols, keep, sky_rejected = project_color_points(
                    points,
                    poses[frame_id],
                    images,
                    args.cams,
                    sky_filter=args.sky_filter,
                    sky_upper_ratio=args.sky_upper_ratio,
                )
                colored_points += keep
                sky_rejected_points += sky_rejected
                bounds = update_bounds(bounds, pts_col)
                if body_f is not None and keep:
                    append_binary_xyzrgb(body_f, pts_col, cols)
                if voxels is not None and keep:
                    update_voxels(voxels, pts_col, cols, args.voxel_size)

                if n == 1 or n % args.progress_every == 0:
                    ratio = colored_points / max(raw_points, 1)
                    elapsed = time.time() - t0
                    print(f"progress={n}/{len(target_ids)} frame={frame_id} raw={raw_points} colored={colored_points} ratio={ratio:.3f} elapsed={elapsed:.1f}s")

        if body_f is not None:
            body_f.close()
            write_binary_ply_from_body(body_path, args.output, colored_points)
        if voxels is not None:
            write_voxel_ply(voxels, args.voxel_output)
    finally:
        for cap in caps.values():
            cap.release()
        if body_f is not None and not body_f.closed:
            body_f.close()
        if body_path and os.path.exists(body_path):
            os.remove(body_path)

    report = {
        "lx_file": str(args.lx_file),
        "start": args.start,
        "end": args.end,
        "frames": len(target_ids),
        "raw_points": raw_points,
        "colored_points": colored_points,
        "colored_ratio": colored_points / max(raw_points, 1),
        "sky_filter": args.sky_filter,
        "sky_upper_ratio": args.sky_upper_ratio,
        "sky_rejected_projected_samples": sky_rejected_points,
        "failed_image_reads": failed_images,
        "frame_map_jsonl": str(args.frame_map_jsonl) if args.frame_map_jsonl else None,
        "require_frame_map": bool(args.require_frame_map),
        "allow_rejected_frame_map": bool(args.allow_rejected_frame_map),
        "frame_map_rows": len(frame_map),
        "mapped_image_reads": mapped_image_reads,
        "direct_image_reads": direct_image_reads,
        "missing_frame_map_reads": missing_frame_map_reads,
        "bounds": bounds,
        "output": None if args.skip_full_output else str(args.output),
        "output_exists": False if args.skip_full_output else args.output.exists(),
        "voxel_output": str(args.voxel_output) if args.voxel_output else None,
        "voxel_count": len(voxels) if voxels is not None else None,
        "voxel_size": args.voxel_size if args.voxel_output else None,
        "calib_file": config.CALIB_FILE,
        "image_dir": config.IMAGE_DIR,
        "video_dir": config.VIDEO_DIR,
        "elapsed_sec": time.time() - t0,
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
