#!/usr/bin/env python3
"""Sweep timestamp-mode effective FPS for sync path solving."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import solve_sync_path_from_candidates as solver


def parse_float_range(text: str) -> list[float]:
    values: list[float] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            start_s, end_s, step_s = chunk.split(":")
            start, end, step = float(start_s), float(end_s), float(step_s)
            if step <= 0:
                raise ValueError(f"step must be positive: {chunk}")
            cur = start
            while cur <= end + step * 0.5:
                values.append(round(cur, 6))
                cur += step
        else:
            values.append(float(chunk))
    return sorted(set(values))


def solve_for_fps(
    grouped: dict[int, dict[int, list[dict[str, Any]]]],
    args: argparse.Namespace,
    fps: float,
) -> tuple[dict[str, Any], dict[int, list[dict[str, Any]]]]:
    cam_reports: dict[str, Any] = {}
    paths: dict[int, list[dict[str, Any]]] = {}
    intercepts: dict[str, float] = {}
    all_accepted = True
    for cam_id in sorted(grouped):
        summary, path, intercept = solve_cam_for_fps_and_intercepts(grouped[cam_id], args, fps)
        cam_reports[str(cam_id)] = summary
        paths[cam_id] = path
        intercepts[str(cam_id)] = intercept
        all_accepted = all_accepted and bool(summary.get("accepted"))
    mean_loss = sum(
        float(report["score_loss_from_independent_best"]["mean"])
        for report in cam_reports.values()
        if report.get("score_loss_from_independent_best", {}).get("mean") is not None
    ) / max(len(cam_reports), 1)
    max_dev = max(
        float(report["step_ratio"]["max_abs_deviation"])
        for report in cam_reports.values()
        if report.get("step_ratio", {}).get("max_abs_deviation") is not None
    )
    report = {
        "fps": float(fps),
        "timestamp_phase_fraction": float(getattr(args, "timestamp_phase_fraction", 0.0)),
        "status": "accepted" if all_accepted else "rejected",
        "accepted": bool(all_accepted),
        "mean_score_loss": mean_loss,
        "max_step_deviation": max_dev,
        "cam_intercepts": intercepts,
        "cam_reports": cam_reports,
    }
    return report, paths


def solve_cam_for_fps_and_intercepts(
    frame_candidates: dict[int, list[dict[str, Any]]],
    args: argparse.Namespace,
    fps: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    best_summary = None
    best_path = None
    best_key = None
    best_intercept = 0.0
    absolute_prior_weight = float(getattr(args, "absolute_prior_weight", 0.0))
    absolute_prior_tolerance = float(getattr(args, "absolute_prior_tolerance", 100.0))
    intercept_values = parse_float_range(getattr(args, "intercept_values", "0")) if absolute_prior_weight > 0 else [0.0]
    for intercept in intercept_values:
        path = solver.solve_cam_path(
            frame_candidates,
            args.target_ratio,
            args.velocity_weight,
            args.nonmonotonic_penalty,
            args.score_weight,
            time_mode="timestamp",
            video_fps=fps,
            absolute_prior_weight=absolute_prior_weight,
            absolute_prior_tolerance=absolute_prior_tolerance,
            absolute_intercept=intercept,
        )
        summary = solver.summarize_path(
            path,
            args.target_ratio,
            args.max_ratio_deviation,
            args.max_score_loss_mean,
            args.max_score_loss_max,
            time_mode="timestamp",
            video_fps=fps,
        )
        prior_mean = summary.get("absolute_prior_error", {}).get("mean")
        key = (
            not bool(summary.get("accepted")),
            float(summary["score_loss_from_independent_best"]["mean"]),
            float(summary["step_ratio"]["max_abs_deviation"]),
            float(prior_mean) if prior_mean is not None else 0.0,
        )
        if best_key is None or key < best_key:
            best_summary = summary
            best_path = path
            best_intercept = float(intercept)
            best_key = key
    assert best_summary is not None and best_path is not None
    best_summary = dict(best_summary)
    best_summary["absolute_intercept"] = best_intercept
    return best_summary, best_path, best_intercept


def write_paths(path: Path, paths: dict[int, list[dict[str, Any]]], status: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        for cam_id in sorted(paths):
            summary_status = status
            for row in paths[cam_id]:
                out = dict(row)
                out["cam_path_status"] = summary_status
                f.write(json.dumps(out, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-jsonl", type=Path, required=True)
    parser.add_argument("--img-pos-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps-values", default="6.0:10.5:0.25")
    parser.add_argument("--phase-values", default="0",
                        help="Local timestamp phase fractions to sweep, e.g. 0,0.5,1 for start/middle/end.")
    parser.add_argument("--intercept-values", default="0:1800:100",
                        help="Expected video_idx at first probe timestamp; swept independently per camera.")
    parser.add_argument("--target-ratio", type=float, default=1.0)
    parser.add_argument("--max-ratio-deviation", type=float, default=0.45)
    parser.add_argument("--velocity-weight", type=float, default=2.0)
    parser.add_argument("--nonmonotonic-penalty", type=float, default=1000.0)
    parser.add_argument("--score-weight", type=float, default=1.0)
    parser.add_argument("--absolute-prior-weight", type=float, default=0.0)
    parser.add_argument("--absolute-prior-tolerance", type=float, default=100.0)
    parser.add_argument("--max-score-loss-mean", type=float, default=0.12)
    parser.add_argument("--max-score-loss-max", type=float, default=0.30)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = solver.load_jsonl(args.candidates_jsonl)
    raw_timestamps = solver.load_frame_timestamps(args.img_pos_file)
    base_grouped = solver.group_candidates(rows)
    sweep = []
    best_report = None
    best_paths = None
    for phase in parse_float_range(args.phase_values):
        args.timestamp_phase_fraction = phase
        timestamps = solver.apply_timestamp_phase(raw_timestamps, phase)
        grouped = solver.attach_frame_times(base_grouped, timestamps, raw_timestamps)
        for fps in parse_float_range(args.fps_values):
            report, paths = solve_for_fps(grouped, args, fps)
            sweep.append(report)
            key = (not report["accepted"], report["mean_score_loss"], report["max_step_deviation"])
            if best_report is None or key < (
                not best_report["accepted"],
                best_report["mean_score_loss"],
                best_report["max_step_deviation"],
            ):
                best_report = report
                best_paths = paths
    assert best_report is not None and best_paths is not None
    result = {
        "candidates_jsonl": str(args.candidates_jsonl),
        "img_pos_file": str(args.img_pos_file),
        "fps_values": parse_float_range(args.fps_values),
        "phase_values": parse_float_range(args.phase_values),
        "intercept_values": parse_float_range(args.intercept_values),
        "absolute_prior_weight": args.absolute_prior_weight,
        "absolute_prior_tolerance": args.absolute_prior_tolerance,
        "best_fps": best_report["fps"],
        "best_status": best_report["status"],
        "best_report": best_report,
        "sweep": sweep,
    }
    (args.output_dir / "timestamp_fps_sweep_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_paths(args.output_dir / "sync_smooth_paths.jsonl", best_paths, best_report["status"])
    (args.output_dir / "sync_smooth_path_report.json").write_text(
        json.dumps({
            "status": best_report["status"],
            "time_mode": "timestamp",
            "video_fps": best_report["fps"],
            "timestamp_phase_fraction": best_report["timestamp_phase_fraction"],
            "cam_intercepts": best_report["cam_intercepts"],
            "absolute_prior_weight": args.absolute_prior_weight,
            "absolute_prior_tolerance": args.absolute_prior_tolerance,
            "cam_reports": best_report["cam_reports"],
            "sweep_report": str(args.output_dir / "timestamp_fps_sweep_report.json"),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "best_fps": best_report["fps"],
        "best_timestamp_phase_fraction": best_report["timestamp_phase_fraction"],
        "best_status": best_report["status"],
        "mean_score_loss": best_report["mean_score_loss"],
        "max_step_deviation": best_report["max_step_deviation"],
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
