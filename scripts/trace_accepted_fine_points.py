#!/usr/bin/env python3
"""Attach frame/camera/mask metadata to accepted fine-object points."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def read_ascii_ply(path: Path) -> tuple[list[str], int, np.ndarray]:
    props: list[str] = []
    vertex_count = 0
    header_lines = 0
    in_vertex = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            header_lines += 1
            s = line.strip()
            if s.startswith("element vertex"):
                vertex_count = int(s.split()[-1])
                in_vertex = True
            elif s.startswith("element "):
                in_vertex = False
            elif in_vertex and s.startswith("property"):
                props.append(s.split()[-1])
            elif s == "end_header":
                break
    if vertex_count == 0:
        return props, header_lines, np.empty((0, len(props)), dtype=np.float64)
    data = np.loadtxt(path, skiprows=header_lines, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return props, header_lines, data


def point_key(row: np.ndarray, idx: dict[str, int], scale: int) -> tuple[int, int, int, int, int, int, int]:
    return (
        int(round(float(row[idx["x"]]) * scale)),
        int(round(float(row[idx["y"]]) * scale)),
        int(round(float(row[idx["z"]]) * scale)),
        int(round(float(row[idx["semantic"]]))),
        int(round(float(row[idx["visual_red"]]))),
        int(round(float(row[idx["visual_green"]]))),
        int(round(float(row[idx["visual_blue"]]))),
    )


def load_object_by_candidate(objects_jsonl: Path) -> dict[int, int]:
    mapping = {}
    if not objects_jsonl.exists():
        return mapping
    with objects_jsonl.open("r", encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            for candidate_id in obj.get("candidate_ids", []):
                mapping[int(candidate_id)] = number
    return mapping


def write_enriched_ply(path: Path, props: list[str], data: np.ndarray, idx: dict[str, int], meta_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(data)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property uchar semantic\n")
        f.write("property int accepted_candidate\n")
        f.write("property int fine_object\n")
        f.write("property uchar source_type\n")
        f.write("property int source_cluster\n")
        f.write("property int subcluster\n")
        f.write("property uchar visual_red\nproperty uchar visual_green\nproperty uchar visual_blue\n")
        f.write("property int frame\n")
        f.write("property int camera\n")
        f.write("property int mask\n")
        f.write("property int point_index\n")
        f.write("property uchar trace_status\n")
        f.write("end_header\n")
        for row, meta in zip(data, meta_rows):
            f.write(
                f"{row[idx['x']]:.6f} {row[idx['y']]:.6f} {row[idx['z']]:.6f} "
                f"{int(row[idx['red']])} {int(row[idx['green']])} {int(row[idx['blue']])} "
                f"{int(row[idx['semantic']])} {int(row[idx['accepted_candidate']])} {int(meta['fine_object'])} "
                f"{int(row[idx['source_type']])} {int(row[idx['source_cluster']])} {int(row[idx['subcluster']])} "
                f"{int(row[idx['visual_red']])} {int(row[idx['visual_green']])} {int(row[idx['visual_blue']])} "
                f"{int(meta['frame'])} {int(meta['camera'])} {int(meta['mask'])} {int(meta['point_index'])} "
                f"{int(meta['trace_status'])}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-ply", type=Path, required=True)
    parser.add_argument("--residual-dir", type=Path, required=True)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--coord-scale", type=int, default=1000000)
    args = parser.parse_args()

    props, _, data = read_ascii_ply(args.accepted_ply)
    idx = {name: i for i, name in enumerate(props)}
    required = {
        "x",
        "y",
        "z",
        "red",
        "green",
        "blue",
        "semantic",
        "accepted_candidate",
        "source_type",
        "source_cluster",
        "subcluster",
        "visual_red",
        "visual_green",
        "visual_blue",
    }
    if not required.issubset(idx):
        raise ValueError(f"missing required accepted PLY fields. required={required} available={props}")

    key_to_indices: dict[tuple[int, int, int, int, int, int, int], list[int]] = defaultdict(list)
    for i, row in enumerate(data):
        key_to_indices[point_key(row, idx, args.coord_scale)].append(i)

    object_by_candidate = load_object_by_candidate(args.objects_jsonl)
    meta_rows = [
        {
            "frame": -1,
            "camera": -1,
            "mask": -1,
            "point_index": -1,
            "fine_object": object_by_candidate.get(int(row[idx["accepted_candidate"]]), -1),
            "trace_status": 0,
        }
        for row in data
    ]
    matched = np.zeros(len(data), dtype=bool)
    duplicate_matches = 0
    residual_files = sorted(args.residual_dir.glob("residuals_frame_*.ply"))
    for residual_path in residual_files:
        rprops, _, rdata = read_ascii_ply(residual_path)
        ridx = {name: i for i, name in enumerate(rprops)}
        rrequired = {
            "x",
            "y",
            "z",
            "semantic",
            "visual_red",
            "visual_green",
            "visual_blue",
            "frame",
            "camera",
            "mask",
            "point_index",
        }
        if not rrequired.issubset(ridx):
            continue
        for rrow in rdata:
            hits = key_to_indices.get(point_key(rrow, ridx, args.coord_scale))
            if not hits:
                continue
            for hit in hits:
                if matched[hit]:
                    duplicate_matches += 1
                    continue
                meta_rows[hit].update(
                    {
                        "frame": int(rrow[ridx["frame"]]),
                        "camera": int(rrow[ridx["camera"]]),
                        "mask": int(rrow[ridx["mask"]]),
                        "point_index": int(rrow[ridx["point_index"]]),
                        "trace_status": 1,
                    }
                )
                matched[hit] = True

    write_enriched_ply(args.output_ply, props, data, idx, meta_rows)
    frame_counts = Counter(int(row["frame"]) for row in meta_rows if int(row["frame"]) >= 0)
    camera_counts = Counter(int(row["camera"]) for row in meta_rows if int(row["camera"]) >= 0)
    candidate_counts = Counter(int(data[i, idx["accepted_candidate"]]) for i, ok in enumerate(matched) if ok)
    report = {
        "accepted_ply": str(args.accepted_ply),
        "residual_dir": str(args.residual_dir),
        "objects_jsonl": str(args.objects_jsonl),
        "output_ply": str(args.output_ply),
        "points": int(len(data)),
        "matched_points": int(matched.sum()),
        "unmatched_points": int((~matched).sum()),
        "matched_ratio": float(matched.sum() / max(len(data), 1)),
        "duplicate_matches": int(duplicate_matches),
        "frame_count": int(len(frame_counts)),
        "camera_counts": dict(sorted(camera_counts.items())),
        "top_frames": frame_counts.most_common(30),
        "candidate_count": int(len(set(int(x) for x in data[:, idx["accepted_candidate"]].tolist()))),
        "matched_candidate_count": int(len(candidate_counts)),
        "unmatched_candidate_ids": sorted(
            set(int(x) for x in data[:, idx["accepted_candidate"]].tolist()) - set(candidate_counts)
        ),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "points": report["points"],
                "matched_points": report["matched_points"],
                "unmatched_points": report["unmatched_points"],
                "matched_ratio": report["matched_ratio"],
                "duplicate_matches": report["duplicate_matches"],
                "frame_count": report["frame_count"],
                "matched_candidate_count": report["matched_candidate_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
