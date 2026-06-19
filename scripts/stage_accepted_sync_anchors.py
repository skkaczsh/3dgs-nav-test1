#!/usr/bin/env python3
"""Validate and stage exported accepted sync anchors for the 5070Ti runner."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_sync_frame_map_readiness as readiness


DEFAULT_REVIEW_NAME = "sync_anchor_review_small_20260619_v2"
DEFAULT_READINESS_FRAMES = [1000, 1600, 2200, 2800, 3400, 4000, 4600, 5200, 5800]


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
    rows = readiness.read_jsonl(path)
    if not rows:
        raise ValueError(f"accepted anchors file is empty: {path}")
    return rows


def validate_source(
    source: Path,
    cams: list[int],
    min_accepted_per_cam: int,
    frames: list[int],
) -> dict[str, Any]:
    # Build the same report shape the production readiness gate uses, scoped to
    # anchors only.  This keeps staging and production policy aligned.
    args = argparse.Namespace(
        anchors_jsonl=source,
        frame_map_jsonl=None,
        solver_report=None,
        frames=frames,
        start=None,
        end=None,
        stride=1,
        cams=cams,
        min_accepted_per_cam=min_accepted_per_cam,
        allow_rejected=False,
        output=None,
    )
    return readiness.build_report(args)


def stage(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    source = resolve_source(args.source, args.downloads_dir).resolve()
    target = args.target.resolve() if args.target else default_target(repo_root, args.review_name)
    rows = load_rows(source)
    report = validate_source(source, args.cams, args.min_accepted_per_cam, args.frames)
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
    return {
        "passed": True,
        "staged": not args.dry_run,
        "dry_run": bool(args.dry_run),
        "source": str(source),
        "target": str(target),
        "row_count": len(rows),
        "errors": [],
        "readiness": report,
        "next_command": f"cd {repo_root} && scripts/run_rtx5070_sync_anchor_solver.sh",
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
    parser.add_argument("--frames", type=int, nargs="*", default=DEFAULT_READINESS_FRAMES)
    parser.add_argument("--min-accepted-per-cam", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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
