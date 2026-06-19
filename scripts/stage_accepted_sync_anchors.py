#!/usr/bin/env python3
"""Validate and stage exported accepted sync anchors for the 5070Ti runner."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_sync_anchors


DEFAULT_REVIEW_NAME = "sync_anchor_review_priority_sky_penalty_timestamp_absprior_dot3_20260619"


def default_source() -> Path:
    return Path.home() / "Downloads" / "accepted_sync_anchors.jsonl"


def default_downloads_dir() -> Path:
    return Path.home() / "Downloads"


def default_target(repo_root: Path, review_name: str) -> Path:
    return repo_root / "server_parking_priority_s10" / review_name / "accepted_sync_anchors.jsonl"


def discover_source(downloads_dir: Path) -> Path:
    candidates = sorted(downloads_dir.glob("accepted_sync_anchors*.jsonl"))
    if not candidates:
        return downloads_dir / "accepted_sync_anchors.jsonl"
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def resolve_source(source: Path | None, downloads_dir: Path) -> Path:
    return source if source is not None else discover_source(downloads_dir)


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    rows = validate_sync_anchors.read_jsonl(path)
    if not rows:
        raise ValueError(f"accepted anchors file is empty: {path}")
    diagnostic_rows = [i for i, row in enumerate(rows, start=1) if row.get("diagnostic_only") is True]
    if diagnostic_rows:
        sample = ",".join(str(i) for i in diagnostic_rows[:10])
        raise ValueError(
            f"refusing diagnostic_only anchors in {path}; "
            f"rows={sample}; inspect and export accepted anchors from the review page instead"
        )
    return rows


def validate_source(
    source: Path,
    cams: list[int],
    min_accepted_per_cam: int,
    img_pos_file: Path | None,
    timestamp_phase_fraction: float,
    expected_fps: float,
    max_fps_error: float,
) -> dict[str, Any]:
    args = argparse.Namespace(
        anchors_jsonl=source,
        img_pos_file=img_pos_file,
        timestamp_phase_fraction=timestamp_phase_fraction,
        expected_fps=expected_fps,
        max_fps_error=max_fps_error,
        cams=cams,
        min_accepted_per_cam=min_accepted_per_cam,
        output=None,
    )
    return validate_sync_anchors.build_report(args)


def run_solver(repo_root: Path, env: dict[str, str]) -> int:
    completed = subprocess.run(
        ["bash", str(repo_root / "scripts" / "run_rtx5070_sync_anchor_solver.sh")],
        cwd=repo_root,
        env={**os.environ, **env},
        check=False,
    )
    return int(completed.returncode)


def stage(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    source = resolve_source(args.source, args.downloads_dir).resolve()
    target = args.target.resolve() if args.target else default_target(repo_root, args.review_name)
    rows = load_rows(source)
    report = validate_source(
        source,
        args.cams,
        args.min_accepted_per_cam,
        args.img_pos_file,
        args.timestamp_phase_fraction,
        args.expected_fps,
        args.max_fps_error,
    )
    if not report["passed"]:
        return {
            "passed": False,
            "staged": False,
            "source": str(source),
            "target": str(target),
            "row_count": len(rows),
            "errors": report["errors"],
            "readiness": report,
        }
    if target.exists() and not args.force:
        return {
            "passed": False,
            "staged": False,
            "source": str(source),
            "target": str(target),
            "row_count": len(rows),
            "errors": [f"target_exists={target}; pass --force to overwrite"],
            "readiness": report,
        }
    if not args.dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    solver_exit_code = None
    if args.run_solver and not args.dry_run:
        solver_exit_code = run_solver(repo_root, {
            "LOCAL_ANCHORS": str(target),
            "LOCAL_IMG_POS": str(args.img_pos_file) if args.img_pos_file else "",
            "REVIEW_NAME": args.review_name,
        })
    return {
        "passed": solver_exit_code in (None, 0),
        "staged": not args.dry_run,
        "dry_run": bool(args.dry_run),
        "run_solver": bool(args.run_solver),
        "solver_exit_code": solver_exit_code,
        "source": str(source),
        "target": str(target),
        "row_count": len(rows),
        "errors": [],
        "readiness": report,
        "next_command": f"cd {repo_root} && LOCAL_ANCHORS={target} scripts/run_rtx5070_sync_anchor_solver.sh",
    }


def error_message(exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        return f"source_missing={exc}"
    return str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=None,
                        help="Explicit accepted_sync_anchors JSONL. Defaults to latest accepted_sync_anchors*.jsonl in --downloads-dir.")
    parser.add_argument("--downloads-dir", type=Path, default=default_downloads_dir())
    parser.add_argument("--target", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--review-name", default=DEFAULT_REVIEW_NAME)
    parser.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--min-accepted-per-cam", type=int, default=2)
    parser.add_argument("--img-pos-file", type=Path, default=Path(__file__).resolve().parents[1] / "../MT20260616-175807/image/img_pos.txt")
    parser.add_argument("--timestamp-phase-fraction", type=float, default=1.0)
    parser.add_argument("--expected-fps", type=float, default=6.0)
    parser.add_argument("--max-fps-error", type=float, default=2.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-solver", action="store_true",
                        help="After staging, run scripts/run_rtx5070_sync_anchor_solver.sh.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        result = stage(args)
    except (FileNotFoundError, ValueError) as exc:
        source = resolve_source(args.source, args.downloads_dir)
        result = {
            "passed": False,
            "staged": False,
            "source": str(source),
            "target": str(args.target) if args.target else str(default_target(args.repo_root.resolve(), args.review_name)),
            "errors": [error_message(exc)],
        }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
