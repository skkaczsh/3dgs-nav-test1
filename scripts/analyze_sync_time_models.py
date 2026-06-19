#!/usr/bin/env python3
"""Analyze simple section->video time models against scored sync candidates.

This script is a diagnostic gate.  It does not declare a synchronization truth.
It asks whether any cheap time model is consistent with the visual candidate
scores produced by `calibrate_lx_video_frame_mapping.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_candidates(rows: list[dict[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["frame_id"]), int(row["cam_id"]))].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
    return grouped


def nearest_candidate_metrics(candidates: list[dict[str, Any]], predicted_video_idx: int) -> dict[str, Any]:
    if not candidates:
        return {"present": False}
    best_score = float(candidates[0].get("score", 0.0))
    by_video = {int(row["video_idx"]): row for row in candidates}
    exact = by_video.get(int(predicted_video_idx))
    nearest = min(candidates, key=lambda row: abs(int(row["video_idx"]) - int(predicted_video_idx)))
    exact_rank = None
    exact_score = None
    if exact is not None:
        for idx, row in enumerate(candidates, start=1):
            if int(row["video_idx"]) == int(predicted_video_idx):
                exact_rank = idx
                exact_score = float(row.get("score", 0.0))
                break
    nearest_rank = next(
        idx for idx, row in enumerate(candidates, start=1)
        if int(row["video_idx"]) == int(nearest["video_idx"])
    )
    nearest_score = float(nearest.get("score", 0.0))
    return {
        "present": exact is not None,
        "predicted_video_idx": int(predicted_video_idx),
        "best_video_idx": int(candidates[0]["video_idx"]),
        "best_score": best_score,
        "exact_rank": exact_rank,
        "exact_score": exact_score,
        "exact_score_loss": best_score - exact_score if exact_score is not None else None,
        "nearest_video_idx": int(nearest["video_idx"]),
        "nearest_distance": abs(int(nearest["video_idx"]) - int(predicted_video_idx)),
        "nearest_rank": nearest_rank,
        "nearest_score": nearest_score,
        "nearest_score_loss": best_score - nearest_score,
    }


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(x) for x in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def summarize_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact = [row for row in rows if row["metrics"]["present"]]
    exact_ranks = [float(row["metrics"]["exact_rank"]) for row in exact if row["metrics"]["exact_rank"] is not None]
    exact_losses = [float(row["metrics"]["exact_score_loss"]) for row in exact if row["metrics"]["exact_score_loss"] is not None]
    nearest_distances = [float(row["metrics"]["nearest_distance"]) for row in rows]
    nearest_ranks = [float(row["metrics"]["nearest_rank"]) for row in rows]
    nearest_losses = [float(row["metrics"]["nearest_score_loss"]) for row in rows]
    return {
        "probe_count": len(rows),
        "exact_candidate_count": len(exact),
        "exact_candidate_ratio": len(exact) / max(len(rows), 1),
        "exact_rank": {
            "p50": percentile(exact_ranks, 50),
            "mean": float(statistics.fmean(exact_ranks)) if exact_ranks else None,
            "max": int(max(exact_ranks)) if exact_ranks else None,
        },
        "exact_score_loss": {
            "p50": percentile(exact_losses, 50),
            "mean": float(statistics.fmean(exact_losses)) if exact_losses else None,
            "max": float(max(exact_losses)) if exact_losses else None,
        },
        "nearest_distance": {
            "p50": percentile(nearest_distances, 50),
            "mean": float(statistics.fmean(nearest_distances)) if nearest_distances else None,
            "max": int(max(nearest_distances)) if nearest_distances else None,
        },
        "nearest_rank": {
            "p50": percentile(nearest_ranks, 50),
            "mean": float(statistics.fmean(nearest_ranks)) if nearest_ranks else None,
            "max": int(max(nearest_ranks)) if nearest_ranks else None,
        },
        "nearest_score_loss": {
            "p50": percentile(nearest_losses, 50),
            "mean": float(statistics.fmean(nearest_losses)) if nearest_losses else None,
            "max": float(max(nearest_losses)) if nearest_losses else None,
        },
    }


def fit_line(xs: list[float], ys: list[float]) -> tuple[float, float]:
    if len(xs) < 2:
        return 1.0, 0.0
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0:
        return 1.0, y_mean - x_mean
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom
    intercept = y_mean - slope * x_mean
    return float(slope), float(intercept)


def build_models(
    poses: dict[int, dict[str, Any]],
    video_frame_count: int,
    candidate_groups: dict[tuple[int, int], list[dict[str, Any]]],
) -> dict[str, Callable[[int, int], int]]:
    frame_ids = sorted(poses)
    first_frame = frame_ids[0]
    last_frame = frame_ids[-1]
    first_ts = float(poses[first_frame]["timestamp"])
    last_ts = float(poses[last_frame]["timestamp"])
    span = max(last_ts - first_ts, 1e-9)

    def direct(frame_id: int, cam_id: int) -> int:
        return frame_id

    def timestamp_10fps(frame_id: int, cam_id: int) -> int:
        return round((float(poses[frame_id]["timestamp"]) - first_ts) * 10.0)

    def timestamp_compressed(frame_id: int, cam_id: int) -> int:
        ratio = (float(poses[frame_id]["timestamp"]) - first_ts) / span
        return round(ratio * max(video_frame_count - 1, 0))

    def cam_info_first(frame_id: int, cam_id: int) -> int:
        return int(poses[frame_id]["cam_info"].get(cam_id, (frame_id, 0))[0])

    def cam_info_second(frame_id: int, cam_id: int) -> int:
        return int(poses[frame_id]["cam_info"].get(cam_id, (0, frame_id))[1])

    best_x = []
    best_y = []
    for (frame_id, _cam_id), rows in candidate_groups.items():
        best_x.append(float(frame_id))
        best_y.append(float(rows[0]["video_idx"]))
    slope, intercept = fit_line(best_x, best_y)

    def best_affine(frame_id: int, cam_id: int) -> int:
        return round(slope * frame_id + intercept)

    return {
        "direct_frame_id": direct,
        "timestamp_at_10fps": timestamp_10fps,
        "timestamp_compressed_to_video_span": timestamp_compressed,
        "cam_info_first_value": cam_info_first,
        "cam_info_second_value": cam_info_second,
        f"affine_fit_to_independent_best_slope_{slope:.6f}_intercept_{intercept:.2f}": best_affine,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-frame-count", type=int, default=6181)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_jsonl(args.candidates_jsonl)
    groups = group_candidates(candidates)
    frame_ids = sorted({frame_id for frame_id, _cam_id in groups})
    poses = {row["frame_id"]: row for row in config.load_img_pos(min(frame_ids), max(frame_ids))}
    missing = sorted(set(frame_ids) - set(poses))
    if missing:
        raise SystemExit(f"Missing img_pos rows for frames: {missing[:8]}")

    models = build_models(poses, args.video_frame_count, groups)
    report: dict[str, Any] = {
        "candidates_jsonl": str(args.candidates_jsonl),
        "probe_count": len(groups),
        "frame_range": [min(frame_ids), max(frame_ids)],
        "video_frame_count": args.video_frame_count,
        "model_summaries": {},
    }
    detail_rows = []
    for model_name, model in models.items():
        model_rows = []
        for (frame_id, cam_id), rows in sorted(groups.items()):
            predicted = int(model(frame_id, cam_id))
            predicted = max(0, min(args.video_frame_count - 1, predicted))
            metrics = nearest_candidate_metrics(rows, predicted)
            item = {
                "model": model_name,
                "frame_id": frame_id,
                "cam_id": cam_id,
                "metrics": metrics,
            }
            model_rows.append(item)
            detail_rows.append(item)
        report["model_summaries"][model_name] = summarize_model(model_rows)

    (args.output_dir / "sync_time_model_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (args.output_dir / "sync_time_model_details.jsonl").open("w", encoding="utf-8") as f:
        for row in detail_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
