#!/usr/bin/env python3
"""Build a minimal manual checklist from sync anchor review rows.

This script does not auto-accept anchors. It ranks review rows so a human can
quickly inspect a small, temporally spread set before exporting accepted anchors
from the interactive review page.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def option_by_idx(row: dict[str, Any], option_idx: int | None) -> dict[str, Any] | None:
    if option_idx is None:
        return None
    for option in row.get("options", []):
        if int(option.get("option_idx", -1)) == int(option_idx):
            return option
    return None


def recommended_option(row: dict[str, Any]) -> dict[str, Any] | None:
    selected = option_by_idx(row, row.get("selected_option_idx"))
    if selected is not None:
        return selected
    options = sorted(row.get("options", []), key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return options[0] if options else None


def risk_penalty(row: dict[str, Any]) -> float:
    penalties = {
        "low_score_margin": 0.10,
        "direct_far_below_best": 0.08,
        "large_best_offset": 0.12,
        "wide_offset_span": 0.08,
        "unknown_best_source": 0.20,
    }
    return sum(penalties.get(str(reason), 0.05) for reason in row.get("risk_reasons", []))


def checklist_score(row: dict[str, Any]) -> float:
    option = recommended_option(row) or {}
    margin = float(row.get("score_margin") or 0.0)
    option_score = float(option.get("score") or row.get("best_score") or 0.0)
    source = str(option.get("review_source") or row.get("best_source") or "")
    source_bonus = {
        "smooth_path": 0.10,
        "independent_best": 0.06,
        "top_candidate": 0.03,
        "direct": 0.0,
    }.get(source, -0.05)
    return option_score + 1.5 * margin + source_bonus - risk_penalty(row)


def frame_bin(frame_id: int, min_frame: int, max_frame: int, bins: int) -> int:
    if bins <= 1 or max_frame <= min_frame:
        return 0
    ratio = (int(frame_id) - min_frame) / max(max_frame - min_frame, 1)
    return max(0, min(bins - 1, int(ratio * bins)))


def select_rows(rows: list[dict[str, Any]], per_cam: int, bins: int) -> list[dict[str, Any]]:
    by_cam: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if recommended_option(row) is None:
            continue
        item = dict(row)
        item["_checklist_score"] = checklist_score(row)
        by_cam[int(row["cam_id"])].append(item)

    selected: list[dict[str, Any]] = []
    for cam_id in sorted(by_cam):
        cam_rows = by_cam[cam_id]
        min_frame = min(int(row["frame_id"]) for row in cam_rows)
        max_frame = max(int(row["frame_id"]) for row in cam_rows)
        by_bin: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in cam_rows:
            by_bin[frame_bin(int(row["frame_id"]), min_frame, max_frame, bins)].append(row)
        # First pass: spread across time bins.
        cam_selected: list[dict[str, Any]] = []
        for bin_id in range(bins):
            candidates = sorted(by_bin.get(bin_id, []), key=lambda row: float(row["_checklist_score"]), reverse=True)
            if candidates:
                cam_selected.append(candidates[0])
            if len(cam_selected) >= per_cam:
                break
        # Second pass: fill gaps with highest remaining rows.
        seen = {(int(row["frame_id"]), int(row["cam_id"])) for row in cam_selected}
        remaining = [
            row for row in sorted(cam_rows, key=lambda item: float(item["_checklist_score"]), reverse=True)
            if (int(row["frame_id"]), int(row["cam_id"])) not in seen
        ]
        cam_selected.extend(remaining[: max(0, per_cam - len(cam_selected))])
        selected.extend(cam_selected[:per_cam])
    return sorted(selected, key=lambda row: (int(row["frame_id"]), int(row["cam_id"])))


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    option = recommended_option(row) or {}
    return {
        "frame_id": int(row["frame_id"]),
        "cam_id": int(row["cam_id"]),
        "recommended_option_idx": option.get("option_idx"),
        "recommended_video_idx": option.get("video_idx"),
        "recommended_source": option.get("review_source"),
        "score": option.get("score"),
        "score_margin": row.get("score_margin"),
        "priority_score": row.get("priority_score"),
        "checklist_score": row.get("_checklist_score"),
        "risk_reasons": row.get("risk_reasons", []),
        "panel_path": option.get("panel_path"),
        "anchor_status": row.get("anchor_status", "unreviewed"),
        "selected_video_idx": row.get("selected_video_idx"),
        "selected_option_idx": row.get("selected_option_idx"),
    }


def diagnostic_anchor_row(row: dict[str, Any]) -> dict[str, Any]:
    option = recommended_option(row) or {}
    out = dict(row)
    out["anchor_status"] = "accepted"
    out["selected_option_idx"] = option.get("option_idx")
    out["selected_video_idx"] = option.get("video_idx")
    out["diagnostic_only"] = True
    out["notes"] = "diagnostic suggestion from suggest_sync_anchor_checklist.py; inspect manually before staging"
    return out


def render_markdown(rows: list[dict[str, Any]], review_url: str | None) -> str:
    lines = [
        "# Sync Anchor Checklist",
        "",
        "This is a manual review aid. Do not use these rows as production anchors until they are inspected and exported from the review page.",
        "",
    ]
    if review_url:
        lines.extend([f"- review page: `{review_url}`", ""])
    lines.extend([
        "| frame | cam | suggested video | source | score | margin | risk | panel |",
        "|---:|---:|---:|---|---:|---:|---|---|",
    ])
    for row in rows:
        risk = ", ".join(row.get("risk_reasons") or []) or "none"
        panel = row.get("panel_path") or ""
        lines.append(
            f"| {row['frame_id']} | {row['cam_id']} | {row.get('recommended_video_idx')} | "
            f"{row.get('recommended_source')} | {float(row.get('score') or 0.0):.3f} | "
            f"{float(row.get('score_margin') or 0.0):.3f} | {risk} | `{panel}` |"
        )
    return "\n".join(lines) + "\n"


def build(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.review_jsonl)
    selected_full = select_rows(rows, args.per_cam, args.bins)
    selected = [public_row(row) for row in selected_full]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.output_md.write_text(render_markdown(selected, args.review_url), encoding="utf-8")
    if args.diagnostic_accepted_jsonl:
        args.diagnostic_accepted_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.diagnostic_accepted_jsonl.open("w", encoding="utf-8") as f:
            for row in selected_full:
                f.write(json.dumps(diagnostic_anchor_row(row), ensure_ascii=False) + "\n")
    by_cam: dict[str, int] = {}
    for row in selected:
        cam = str(row["cam_id"])
        by_cam[cam] = by_cam.get(cam, 0) + 1
    return {
        "review_jsonl": str(args.review_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "output_md": str(args.output_md),
        "selected_count": len(selected),
        "selected_by_cam": dict(sorted(by_cam.items())),
        "per_cam": int(args.per_cam),
        "bins": int(args.bins),
        "review_url": args.review_url,
        "diagnostic_accepted_jsonl": str(args.diagnostic_accepted_jsonl) if args.diagnostic_accepted_jsonl else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--per-cam", type=int, default=3)
    parser.add_argument("--bins", type=int, default=3)
    parser.add_argument("--review-url", default="")
    parser.add_argument("--diagnostic-accepted-jsonl", type=Path,
                        help="Optional diagnostic-only accepted-anchor JSONL for validator prechecks. Do not stage directly.")
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
