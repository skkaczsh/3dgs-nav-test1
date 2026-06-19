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


def load_frame_timestamps(path: Path | None) -> dict[int, float]:
    if path is None:
        return {}
    timestamps: dict[int, float] = {}
    with path.open("rb") as f:
        for line in f:
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            parts = text.split()
            if len(parts) < 2:
                continue
            try:
                timestamps[int(parts[0])] = float(parts[1])
            except ValueError:
                continue
    return timestamps


def apply_timestamp_phase(
    timestamps: dict[int, float],
    phase_fraction: float = 0.0,
) -> dict[int, float]:
    """Shift timestamps inside each local section interval.

    A constant timestamp offset is absorbed by the absolute intercept.  The
    useful uncertainty is whether img_pos time refers to the start/middle/end
    of a nonuniform section interval:

      adjusted_t[i] = raw_t[i] + phase_fraction * (raw_t[i + 1] - raw_t[i])

    The last frame reuses the previous interval.  phase 0.0 preserves existing
    behavior.
    """
    phase = float(phase_fraction)
    if not timestamps or abs(phase) < 1e-12:
        return dict(timestamps)
    frames = sorted(timestamps)
    out: dict[int, float] = {}
    last_dt = 0.0
    for idx, frame_id in enumerate(frames):
        if idx + 1 < len(frames):
            dt = float(timestamps[frames[idx + 1]]) - float(timestamps[frame_id])
            last_dt = dt
        else:
            dt = last_dt
        out[frame_id] = float(timestamps[frame_id]) + phase * dt
    return out


def attach_frame_times(
    grouped: dict[int, dict[int, list[dict[str, Any]]]],
    timestamps: dict[int, float],
    raw_timestamps: dict[int, float] | None = None,
) -> dict[int, dict[int, list[dict[str, Any]]]]:
    if not timestamps:
        return grouped
    out: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(dict)
    missing: list[int] = []
    for cam_id, by_frame in grouped.items():
        for frame_id, rows in by_frame.items():
            if frame_id not in timestamps:
                missing.append(frame_id)
                continue
            raw_timestamp = None if raw_timestamps is None else raw_timestamps.get(frame_id)
            out[cam_id][frame_id] = [
                dict(
                    row,
                    sync_timestamp=float(timestamps[frame_id]),
                    raw_sync_timestamp=float(raw_timestamp) if raw_timestamp is not None else float(timestamps[frame_id]),
                )
                for row in rows
            ]
    if missing:
        unique = sorted(set(missing))
        raise ValueError(f"timestamps missing for frame ids: {unique[:10]}")
    return out


def load_accepted_anchors(path: Path | None) -> dict[tuple[int, int], int]:
    if path is None:
        return {}
    anchors = {}
    for row in load_jsonl(path):
        if str(row.get("anchor_status", "")).lower() != "accepted":
            continue
        selected = row.get("selected_video_idx")
        if selected is None and row.get("selected_option_idx") is not None:
            option_idx = int(row["selected_option_idx"])
            for option in row.get("options", []):
                if int(option.get("option_idx", -1)) == option_idx:
                    selected = option.get("video_idx")
                    break
        if selected is None:
            raise ValueError(f"accepted anchor missing selected video: frame={row.get('frame_id')} cam={row.get('cam_id')}")
        key = (int(row["frame_id"]), int(row["cam_id"]))
        value = int(selected)
        if key in anchors and anchors[key] != value:
            raise ValueError(f"conflicting anchors for frame={key[0]} cam={key[1]}: {anchors[key]} vs {value}")
        anchors[key] = value
    return anchors


def apply_anchors(
    grouped: dict[int, dict[int, list[dict[str, Any]]]],
    anchors: dict[tuple[int, int], int],
) -> dict[int, dict[int, list[dict[str, Any]]]]:
    if not anchors:
        return grouped
    out: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(dict)
    for cam_id, by_frame in grouped.items():
        for frame_id, rows in by_frame.items():
            selected = anchors.get((frame_id, cam_id))
            if selected is None:
                out[cam_id][frame_id] = rows
                continue
            anchored_rows = [dict(row, anchor_status="accepted") for row in rows if int(row["video_idx"]) == int(selected)]
            if not anchored_rows:
                raise ValueError(f"accepted anchor video_idx={selected} not found in candidates for frame={frame_id} cam={cam_id}")
            out[cam_id][frame_id] = anchored_rows
    unused = sorted(set(anchors) - {(frame_id, cam_id) for cam_id, by_frame in grouped.items() for frame_id in by_frame})
    if unused:
        raise ValueError(f"accepted anchors not present in candidates: {unused[:5]}")
    return out


def transition_penalty(
    prev: dict[str, Any],
    cur: dict[str, Any],
    target_ratio: float,
    velocity_weight: float,
    nonmonotonic_penalty: float,
    time_mode: str,
    video_fps: float,
) -> float:
    if time_mode == "timestamp":
        df = max((float(cur["sync_timestamp"]) - float(prev["sync_timestamp"])) * float(video_fps), 1e-6)
    else:
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
    time_mode: str = "frame-id",
    video_fps: float = 10.0,
    absolute_prior_weight: float = 0.0,
    absolute_prior_tolerance: float = 100.0,
    absolute_intercept: float = 0.0,
) -> list[dict[str, Any]]:
    frames = sorted(frame_candidates)
    if not frames:
        return []
    time_origin = None
    if time_mode == "timestamp" and absolute_prior_weight > 0:
        time_origin = min(float(row["sync_timestamp"]) for rows in frame_candidates.values() for row in rows)
    first = frame_candidates[frames[0]]
    costs: list[list[float]] = [[
        -score_weight * float(row["score"]) + absolute_prior_cost(
            row,
            time_origin,
            video_fps,
            absolute_intercept,
            absolute_prior_weight,
            absolute_prior_tolerance,
        )
        for row in first
    ]]
    backptrs: list[list[int]] = [[-1 for _ in first]]
    for i in range(1, len(frames)):
        prev_rows = frame_candidates[frames[i - 1]]
        cur_rows = frame_candidates[frames[i]]
        prev_cost = costs[-1]
        cur_cost = [math.inf for _ in cur_rows]
        cur_back = [-1 for _ in cur_rows]
        for j, cur in enumerate(cur_rows):
            score_cost = -score_weight * float(cur["score"]) + absolute_prior_cost(
                cur,
                time_origin,
                video_fps,
                absolute_intercept,
                absolute_prior_weight,
                absolute_prior_tolerance,
            )
            best_cost = math.inf
            best_k = -1
            for k, prev in enumerate(prev_rows):
                cost = prev_cost[k] + transition_penalty(
                    prev,
                    cur,
                    target_ratio,
                    velocity_weight,
                    nonmonotonic_penalty,
                    time_mode,
                    video_fps,
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
        annotate_absolute_prior(row, time_origin, video_fps, absolute_intercept)
        best_score = max(float(item["score"]) for item in frame_candidates[frame_id])
        row["best_score_for_probe"] = best_score
        row["score_loss_from_best"] = best_score - float(row["score"])
        path.append(row)
    return path


def expected_video_idx(
    row: dict[str, Any],
    time_origin: float | None,
    video_fps: float,
    absolute_intercept: float,
) -> float | None:
    if time_origin is None or "sync_timestamp" not in row:
        return None
    return (float(row["sync_timestamp"]) - float(time_origin)) * float(video_fps) + float(absolute_intercept)


def absolute_prior_cost(
    row: dict[str, Any],
    time_origin: float | None,
    video_fps: float,
    absolute_intercept: float,
    absolute_prior_weight: float,
    absolute_prior_tolerance: float,
) -> float:
    if absolute_prior_weight <= 0:
        return 0.0
    expected = expected_video_idx(row, time_origin, video_fps, absolute_intercept)
    if expected is None:
        return 0.0
    tolerance = max(float(absolute_prior_tolerance), 1e-6)
    return float(absolute_prior_weight) * abs(float(row["video_idx"]) - expected) / tolerance


def annotate_absolute_prior(
    row: dict[str, Any],
    time_origin: float | None,
    video_fps: float,
    absolute_intercept: float,
) -> None:
    expected = expected_video_idx(row, time_origin, video_fps, absolute_intercept)
    if expected is None:
        return
    row["absolute_expected_video_idx"] = float(expected)
    row["absolute_prior_error"] = abs(float(row["video_idx"]) - expected)


def estimate_absolute_intercept_from_anchors(
    frame_candidates: dict[int, list[dict[str, Any]]],
    video_fps: float,
    fallback: float,
) -> float:
    accepted = [
        row
        for rows in frame_candidates.values()
        for row in rows
        if str(row.get("anchor_status", "")).lower() == "accepted" and row.get("sync_timestamp") is not None
    ]
    if not accepted:
        return float(fallback)
    time_origin = min(float(row["sync_timestamp"]) for rows in frame_candidates.values() for row in rows)
    values = [
        float(row["video_idx"]) - (float(row["sync_timestamp"]) - time_origin) * float(video_fps)
        for row in accepted
    ]
    return float(statistics.median(values))


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
    time_mode: str = "frame-id",
    video_fps: float = 10.0,
) -> dict[str, Any]:
    if len(path) < 2:
        return {"status": "insufficient_path", "accepted": False, "path_count": len(path)}
    ratios = []
    negative_steps = 0
    for prev, cur in zip(path, path[1:]):
        if time_mode == "timestamp":
            df = max((float(cur["sync_timestamp"]) - float(prev["sync_timestamp"])) * float(video_fps), 1e-6)
        else:
            df = max(int(cur["frame_id"]) - int(prev["frame_id"]), 1)
        dv = int(cur["video_idx"]) - int(prev["video_idx"])
        if dv < 0:
            negative_steps += 1
        ratios.append(dv / float(df))
    losses = [float(row.get("score_loss_from_best", 0.0)) for row in path]
    prior_errors = [float(row["absolute_prior_error"]) for row in path if row.get("absolute_prior_error") is not None]
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
            "mode": time_mode,
            "video_fps": float(video_fps),
        },
        "score_loss_from_independent_best": {
            "p50": float(percentile(losses, 50)),
            "mean": loss_mean,
            "max": loss_max,
            "max_allowed_mean": float(max_score_loss_mean),
            "max_allowed_max": float(max_score_loss_max),
        },
        "absolute_prior_error": {
            "count": len(prior_errors),
            "p50": float(percentile(prior_errors, 50)) if prior_errors else None,
            "mean": float(statistics.fmean(prior_errors)) if prior_errors else None,
            "max": float(max(prior_errors)) if prior_errors else None,
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
    parser.add_argument("--anchors-jsonl", type=Path, help="Manual anchor manifest; accepted rows are hard constraints.")
    parser.add_argument("--time-mode", choices=["frame-id", "timestamp"], default="frame-id",
                        help="Use frame-id deltas or img_pos timestamp deltas for transition smoothness.")
    parser.add_argument("--img-pos-file", type=Path,
                        help="img_pos.txt used when --time-mode timestamp.")
    parser.add_argument("--video-fps", type=float, default=10.0,
                        help="Video frames per second for timestamp-mode expected video_idx deltas.")
    parser.add_argument("--timestamp-phase-fraction", type=float, default=0.0,
                        help="Local section phase: 0=start, 0.5=middle, 1=end of the next img_pos interval.")
    parser.add_argument("--absolute-prior-weight", type=float, default=0.0,
                        help="Penalty weight for absolute timestamp->video index prior. Default off.")
    parser.add_argument("--absolute-prior-tolerance", type=float, default=100.0,
                        help="Video-frame error corresponding to one unit of absolute prior penalty.")
    parser.add_argument("--absolute-intercept", type=float, default=0.0,
                        help="Expected video_idx at the first timestamp when absolute prior is enabled.")
    parser.add_argument("--absolute-intercept-source", choices=["fixed", "anchors"], default="fixed",
                        help="Use fixed --absolute-intercept or estimate per-camera intercept from accepted anchors.")
    args = parser.parse_args()
    if args.time_mode == "timestamp" and args.img_pos_file is None:
        raise SystemExit("--img-pos-file is required when --time-mode timestamp")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.candidates_jsonl)
    anchors = load_accepted_anchors(args.anchors_jsonl)
    raw_timestamps = load_frame_timestamps(args.img_pos_file)
    timestamps = apply_timestamp_phase(raw_timestamps, args.timestamp_phase_fraction)
    grouped = attach_frame_times(apply_anchors(group_candidates(rows), anchors), timestamps, raw_timestamps)
    report = {
        "candidates_jsonl": str(args.candidates_jsonl),
        "anchors_jsonl": str(args.anchors_jsonl) if args.anchors_jsonl else None,
        "accepted_anchor_count": len(anchors),
        "target_ratio": args.target_ratio,
        "max_ratio_deviation": args.max_ratio_deviation,
        "velocity_weight": args.velocity_weight,
        "nonmonotonic_penalty": args.nonmonotonic_penalty,
        "score_weight": args.score_weight,
        "time_mode": args.time_mode,
        "img_pos_file": str(args.img_pos_file) if args.img_pos_file else None,
        "video_fps": args.video_fps,
        "timestamp_phase_fraction": args.timestamp_phase_fraction,
        "absolute_prior_weight": args.absolute_prior_weight,
        "absolute_prior_tolerance": args.absolute_prior_tolerance,
        "absolute_intercept": args.absolute_intercept,
        "absolute_intercept_source": args.absolute_intercept_source,
        "cam_reports": {},
    }
    all_accepted = True
    with (args.output_dir / "sync_smooth_paths.jsonl").open("w", encoding="utf-8") as f:
        for cam_id in sorted(grouped):
            absolute_intercept = args.absolute_intercept
            if args.absolute_intercept_source == "anchors":
                absolute_intercept = estimate_absolute_intercept_from_anchors(
                    grouped[cam_id],
                    args.video_fps,
                    args.absolute_intercept,
                )
            path = solve_cam_path(
                grouped[cam_id],
                args.target_ratio,
                args.velocity_weight,
                args.nonmonotonic_penalty,
                args.score_weight,
                args.time_mode,
                args.video_fps,
                args.absolute_prior_weight,
                args.absolute_prior_tolerance,
                absolute_intercept,
            )
            summary = summarize_path(
                path,
                args.target_ratio,
                args.max_ratio_deviation,
                args.max_score_loss_mean,
                args.max_score_loss_max,
                args.time_mode,
                args.video_fps,
            )
            summary["absolute_intercept"] = float(absolute_intercept)
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
