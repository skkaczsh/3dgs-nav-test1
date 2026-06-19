#!/usr/bin/env python3
"""Build a raw XYZ voxel cloud from MANIFOLD .lx sections.

This preserves geometry density for reverse depth rendering.  Unlike
colorize_lx_stream.py, it never drops points because an image color is missing.
The output is a binary little-endian XYZ PLY suitable as the geometry input for
build_geometry_guidance_maps.py --global-colored-ply.  It can also record the
source frame span for each voxel, which lets reverse depth rendering reject
points that were observed far away in scan time and would otherwise project
through walls from the current camera pose.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np


LX_HEADER_SIZE = 48
LX_COUNT_SIZE = 4
LX_POINT_SIZE = 16


def read_lx_sections(lx_path: Path) -> list[dict]:
    sections = []
    file_size = os.path.getsize(lx_path)
    offset = 0
    section_idx = 0
    with lx_path.open("rb") as f:
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
            sections.append({"index": section_idx, "data_offset": data_offset, "count": count})
            offset = next_offset
            section_idx += 1
    return sections


def read_lx_points(handle, section: dict) -> np.ndarray:
    handle.seek(section["data_offset"])
    raw = handle.read(section["count"] * LX_POINT_SIZE)
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("marker", "<u4")])
    data = np.frombuffer(raw, dtype=dtype)
    points = np.empty((len(data), 3), dtype=np.float32)
    points[:, 0] = data["x"]
    points[:, 1] = data["y"]
    points[:, 2] = data["z"]
    return points


def iter_frame_batches(
    lx_path: Path,
    sections: list[dict],
    start: int,
    end: int,
    frame_step: int,
    batch_points: int,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    batch = []
    frame_batch = []
    count = 0
    with lx_path.open("rb") as f:
        for frame_id in range(start, end + 1, max(frame_step, 1)):
            if frame_id >= len(sections):
                break
            pts = read_lx_points(f, sections[frame_id])
            if len(pts) == 0:
                continue
            batch.append(pts)
            frame_batch.append(np.full(len(pts), int(frame_id), dtype=np.int32))
            count += len(pts)
            if count >= batch_points:
                yield np.vstack(batch), np.concatenate(frame_batch)
                batch = []
                frame_batch = []
                count = 0
        if batch:
            yield np.vstack(batch), np.concatenate(frame_batch)


def voxel_reduce(points: np.ndarray, voxel_size: float, frames: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    keys = np.floor(points / float(voxel_size)).astype(np.int32)
    unique, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((len(unique), 7), dtype=np.float64)
    for axis in range(3):
        sums[:, axis] = np.bincount(inverse, weights=points[:, axis], minlength=len(unique))
    sums[:, 3] = counts.astype(np.float64)
    if frames is None:
        sums[:, 4] = -1.0
        sums[:, 5] = -1.0
    else:
        frames = np.asarray(frames, dtype=np.int32)
        sums[:, 6] = np.bincount(inverse, weights=frames.astype(np.float64), minlength=len(unique))
        order = np.argsort(inverse, kind="mergesort")
        sorted_inverse = inverse[order]
        sorted_frames = frames[order]
        starts = np.r_[0, np.flatnonzero(sorted_inverse[1:] != sorted_inverse[:-1]) + 1]
        ends = np.r_[starts[1:], len(sorted_inverse)]
        frame_min = np.full(len(unique), -1, dtype=np.int32)
        frame_max = np.full(len(unique), -1, dtype=np.int32)
        group_ids = sorted_inverse[starts]
        frame_min[group_ids] = np.minimum.reduceat(sorted_frames, starts)
        frame_max[group_ids] = np.maximum.reduceat(sorted_frames, starts)
        sums[:, 4] = frame_min.astype(np.float64)
        sums[:, 5] = frame_max.astype(np.float64)
    return unique, sums


def write_chunk(path: Path, keys: np.ndarray, accum: np.ndarray) -> None:
    np.savez_compressed(path, keys=keys.astype(np.int32), accum=accum.astype(np.float64))


def load_chunk(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["keys"].astype(np.int32), data["accum"].astype(np.float64)


def merge_chunks(chunk_paths: list[Path], merge_batch: int) -> tuple[np.ndarray, np.ndarray]:
    keys_parts = []
    accum_parts = []
    for path in chunk_paths:
        keys, accum = load_chunk(path)
        keys_parts.append(keys)
        accum_parts.append(accum)
        if sum(len(k) for k in keys_parts) >= merge_batch:
            merged_keys, merged_accum = reduce_key_accum(keys_parts, accum_parts)
            keys_parts, accum_parts = [merged_keys], [merged_accum]
    if not keys_parts:
        return np.empty((0, 3), dtype=np.int32), np.empty((0, 7), dtype=np.float64)
    keys, accum = reduce_key_accum(keys_parts, accum_parts)
    return keys, accum


def reduce_key_accum(keys_parts: list[np.ndarray], accum_parts: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    keys = np.vstack(keys_parts)
    accum = np.vstack(accum_parts)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    out = np.zeros((len(unique), 7), dtype=np.float64)
    for axis in range(4):
        out[:, axis] = np.bincount(inverse, weights=accum[:, axis], minlength=len(unique))
    if accum.shape[1] > 6:
        out[:, 6] = np.bincount(inverse, weights=accum[:, 6], minlength=len(unique))
    order = np.argsort(inverse, kind="mergesort")
    sorted_inverse = inverse[order]
    sorted_min = accum[order, 4]
    sorted_max = accum[order, 5]
    starts = np.r_[0, np.flatnonzero(sorted_inverse[1:] != sorted_inverse[:-1]) + 1]
    group_ids = sorted_inverse[starts]
    valid_min = sorted_min.copy()
    valid_min[valid_min < 0] = np.inf
    frame_min = np.minimum.reduceat(valid_min, starts)
    frame_min[~np.isfinite(frame_min)] = -1
    frame_max = np.maximum.reduceat(sorted_max, starts)
    out[group_ids, 4] = frame_min
    out[group_ids, 5] = frame_max
    return unique.astype(np.int32), out


def write_binary_xyz_ply(
    path: Path,
    points: np.ndarray,
    frame_min: np.ndarray | None = None,
    frame_max: np.ndarray | None = None,
    frame_mean: np.ndarray | None = None,
    frame_count: np.ndarray | None = None,
) -> None:
    include_frame = frame_min is not None and frame_max is not None and frame_mean is not None and frame_count is not None
    dtype_fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if include_frame:
        dtype_fields.extend([("frame_min", "<i4"), ("frame_max", "<i4"), ("frame_mean", "<f4"), ("frame_count", "<u4")])
    dtype = np.dtype(dtype_fields)
    data = np.empty(len(points), dtype=dtype)
    data["x"] = points[:, 0]
    data["y"] = points[:, 1]
    data["z"] = points[:, 2]
    if include_frame:
        data["frame_min"] = np.asarray(frame_min, dtype=np.int32)
        data["frame_max"] = np.asarray(frame_max, dtype=np.int32)
        data["frame_mean"] = np.asarray(frame_mean, dtype=np.float32)
        data["frame_count"] = np.asarray(frame_count, dtype=np.uint32)
    props = "property float x\nproperty float y\nproperty float z\n"
    if include_frame:
        props += "property int frame_min\nproperty int frame_max\nproperty float frame_mean\nproperty uint frame_count\n"
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        f"{props}"
        "end_header\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header)
        f.write(data.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lx-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--batch-points", type=int, default=4_000_000)
    parser.add_argument("--merge-batch-voxels", type=int, default=8_000_000)
    parser.add_argument("--no-frame-metadata", action="store_true")
    parser.add_argument("--keep-chunks", action="store_true")
    parser.add_argument("--progress-every", type=int, default=5)
    args = parser.parse_args()

    t0 = time.time()
    sections = read_lx_sections(args.lx_file)
    if args.end is None:
        args.end = len(sections) - 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)

    raw_points = 0
    chunk_paths: list[Path] = []
    tmpdir = Path(tempfile.mkdtemp(prefix="raw_lx_voxel_", dir=str(args.output.parent)))
    try:
        for i, (points, frames) in enumerate(
            iter_frame_batches(args.lx_file, sections, args.start, args.end, args.frame_step, args.batch_points),
            1,
        ):
            raw_points += len(points)
            keys, accum = voxel_reduce(points, args.voxel_size, None if args.no_frame_metadata else frames)
            chunk_path = tmpdir / f"chunk_{i:05d}.npz"
            write_chunk(chunk_path, keys, accum)
            chunk_paths.append(chunk_path)
            if i == 1 or i % args.progress_every == 0:
                print(
                    f"chunk={i} raw_points={raw_points} chunk_voxels={len(keys)} "
                    f"elapsed={time.time() - t0:.1f}s",
                    flush=True,
                )
        keys, accum = merge_chunks(chunk_paths, args.merge_batch_voxels)
        counts = np.maximum(accum[:, 3:4], 1.0)
        points = (accum[:, :3] / counts).astype(np.float32)
        frame_min = accum[:, 4].astype(np.int32)
        frame_max = accum[:, 5].astype(np.int32)
        frame_mean = (accum[:, 6] / np.maximum(accum[:, 3], 1.0)).astype(np.float32)
        frame_count = accum[:, 3].astype(np.uint32)
        write_binary_xyz_ply(
            args.output,
            points,
            None if args.no_frame_metadata else frame_min,
            None if args.no_frame_metadata else frame_max,
            None if args.no_frame_metadata else frame_mean,
            None if args.no_frame_metadata else frame_count,
        )
        report = {
            "lx_file": str(args.lx_file),
            "output": str(args.output),
            "voxel_size": args.voxel_size,
            "start": args.start,
            "end": args.end,
            "frame_step": args.frame_step,
            "sections": len(sections),
            "raw_points": int(raw_points),
            "voxel_points": int(len(points)),
            "frame_metadata": not args.no_frame_metadata,
            "frame_min": int(frame_min[frame_min >= 0].min()) if (not args.no_frame_metadata and np.any(frame_min >= 0)) else None,
            "frame_max": int(frame_max.max()) if not args.no_frame_metadata and len(frame_max) else None,
            "frame_mean_min": float(frame_mean.min()) if not args.no_frame_metadata and len(frame_mean) else None,
            "frame_mean_max": float(frame_mean.max()) if not args.no_frame_metadata and len(frame_mean) else None,
            "output_mb": os.path.getsize(args.output) / 1024 / 1024,
            "elapsed_sec": time.time() - t0,
        }
        if args.report:
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    finally:
        if args.keep_chunks:
            print(f"kept_chunks={tmpdir}", flush=True)
        else:
            for path in chunk_paths:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            try:
                tmpdir.rmdir()
            except OSError:
                pass


if __name__ == "__main__":
    main()
