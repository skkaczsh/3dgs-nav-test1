#!/usr/bin/env python3
"""Solve a temporally smooth section->video sync path from candidate scores.

`calibrate_lx_video_frame_mapping.py` scores each section/camera independently.
This script adds the missing temporal prior: a usable mapping should be
monotonic and have reasonably stable frame-rate ratio.  It runs a Viterbi-style
dynamic program per camera and reports whether the solved path is stable enough
to become a production frame map.

This is still a gate, not an automatic truth source.  A path that passes numeric
checks must still be reviewed with visual QA sheets before semantic production.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_candidates(rows: list[dict[str, Any]]) -> dict[int, dict[int, list[dict[str, Any]]]]:
    grouped: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[int(row["cam_id"])][int(row["frame_id"])].append(row)
    for by_frame in grouped.values():
        for frame_rows in by_frame.values():
            frame_rows.sort(key=lambda item: (int(item["video_idx"]), -float(item["score"])))
    return grouped


def transition_penalty(
    prev: dict[str, Any],
    cur: dict[str, Any],
    target_ratio: float,
    velocity_weight: float,
    nonmonotonic_penalty: float,
) -> float:
    df = max(int(cur["frame_id"]) - int(prev["frame_id"]), 1)
    dv = int(cur["video_idx"]) - int(prev["video_idx"])
    if dv < 0:
        return nonmonotonic_penalty + abs(dv) * velocity_weight
    ratio = dv / float(df)
    return velocity_weight * abs(ratio - target_ratio)


def solve_cam_path(
    frame_candidates: dict[int, list[dict[str, Any]]],
    target_ratio: float,
    velocity_weight: float,
    nonmonotonic_penalty: float,
    score_weight: float,
) -> list[dict[str, Any]]:
    frames = sorted(frame_candidates)
    if not frames:
        return []
    first = frame_candidates[frames[0]]
    costs: list[list[float]] = [[-score_weight * float(row["score"]) for row in first]]
    backptrs: list[list[int]] = [[-1 for _ in first]]
    for i in range(1, len(frames)):
        prev_rows = frame_candidates[frames[i - 1]]
        cur_rows = frame_candidates[frames[i]]
        prev_cost = costs[-1]
        cur_cost = [math.inf for _ in cur_rows]
        cur_back = [-1 for _ in cur_rows]
        for j, cur in enumerate(cur_rows):
            score_cost = -score_weight * float(cur["score"])
            best_cost = math.inf
            best_k = -1
            for k, prev in enumerate(prev_rows):
                cost = prev_cost[k] + transition_penalty(
                    prev,
                    cur,
                    target_ratio,
                    velocity_weight,
                    nonmonotonic_penalty,
                ) + score_cost
                if cost < best_cost:
                    best_cost = cost
                    best_k = k
            cur_cost[j] = best_cost
            cur_back[j] = best_k
        costs.append(cur_cost)
        backptrs.append(cur_back)
    idx = min(range(len(costs[-1])), key=lambda i: costs[-1][i])
    path_indices = [idx]
    for i in range(len(frames) - 1, 0, -1):
        idx = int(backptrs[i][idx])
        path_indices.append(idx)
    path_indices.reverse()
    path = []
    for frame_id, idx in zip(frames, path_indices):
        row = dict(frame_candidates[frame_id][idx])
        best_score = max(float(item["score"]) for item in frame_candidates[frame_id])
        row["best_score_for_probe"] = best_score
        row["score_loss_from_best"] = best_score - float(row["score"])
        path.append(row)
    return path


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
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


def summarize_path(
    path: list[dict[str, Any]],
    target_ratio: float,
    max_ratio_deviation: float,
    max_score_loss_mean: float,
    max_score_loss_max: float,
) -> dict[str, Any]:
    if len(path) < 2:
        return {"status": "insufficient_path", "accepted": False, "path_count": len(path)}
    ratios = []
    negative_steps = 0
    for prev, cur in zip(path, path[1:]):
        df = max(int(cur["frame_id"]) - int(prev["frame_id"]), 1)
        dv = int(cur["video_idx"]) - int(prev["video_idx"])
        if dv < 0:
            negative_steps += 1
        ratios.append(dv / float(df))
    losses = [float(row.get("score_loss_from_best", 0.0)) for row in path]
    direct_ranks = [int(row["direct_rank"]) for row in path if row.get("direct_rank") is not None]
    max_dev = max(abs(x - target_ratio) for x in ratios) if ratios else float("inf")
    loss_mean = float(statistics.fmean(losses))
    loss_max = float(max(losses))
    accepted = (
        negative_steps == 0
        and max_dev <= max_ratio_deviation
        and loss_mean <= max_score_loss_mean
        and loss_max <= max_score_loss_max
    )
    return {
        "status": "accepted" if accepted else "rejected_unstable_temporal_path",
        "accepted": accepted,
        "path_count": len(path),
        "negative_steps": int(negative_steps),
        "step_ratio": {
            "min": float(min(ratios)),
            "p50": float(percentile(ratios, 50)),
            "mean": float(statistics.fmean(ratios)),
            "max": float(max(ratios)),
            "target": float(target_ratio),
            "max_abs_deviation": max_dev,
        },
        "score_loss_from_independent_best": {
            "p50": float(percentile(losses, 50)),
            "mean": loss_mean,
            "max": loss_max,
            "max_allowed_mean": float(max_score_loss_mean),
            "max_allowed_max": float(max_score_loss_max),
        },
        "direct_rank_on_path": {
            "count": len(direct_ranks),
            "p50": float(percentile([float(x) for x in direct_ranks], 50)) if direct_ranks else None,
            "mean": float(statistics.fmean(direct_ranks)) if direct_ranks else None,
            "max": int(max(direct_ranks)) if direct_ranks else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-ratio", type=float, default=1.0)
    parser.add_argument("--max-ratio-deviation", type=float, default=0.6)
    parser.add_argument("--velocity-weight", type=float, default=1.0)
    parser.add_argument("--nonmonotonic-penalty", type=float, default=1000.0)
    parser.add_argument("--score-weight", type=float, default=1.0)
    parser.add_argument("--max-score-loss-mean", type=float, default=0.10)
    parser.add_argument("--max-score-loss-max", type=float, default=0.25)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.candidates_jsonl)
    grouped = group_candidates(rows)
    report = {
        "candidates_jsonl": str(args.candidates_jsonl),
        "target_ratio": args.target_ratio,
        "max_ratio_deviation": args.max_ratio_deviation,
        "velocity_weight": args.velocity_weight,
        "nonmonotonic_penalty": args.nonmonotonic_penalty,
        "score_weight": args.score_weight,
        "cam_reports": {},
    }
    all_accepted = True
    with (args.output_dir / "sync_smooth_paths.jsonl").open("w", encoding="utf-8") as f:
        for cam_id in sorted(grouped):
            path = solve_cam_path(
                grouped[cam_id],
                args.target_ratio,
                args.velocity_weight,
                args.nonmonotonic_penalty,
                args.score_weight,
            )
            summary = summarize_path(
                path,
                args.target_ratio,
                args.max_ratio_deviation,
                args.max_score_loss_mean,
                args.max_score_loss_max,
            )
            report["cam_reports"][str(cam_id)] = summary
            all_accepted = all_accepted and bool(summary.get("accepted"))
            for row in path:
                row["cam_path_status"] = summary["status"]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report["status"] = "accepted" if all_accepted else "rejected"
    (args.output_dir / "sync_smooth_path_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "cam_reports": report["cam_reports"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
