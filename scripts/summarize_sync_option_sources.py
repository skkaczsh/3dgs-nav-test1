#!/usr/bin/env python3
"""Summarize direct, independent-best, and smooth sync option quality."""

from __future__ import annotations

import argparse
import json
import math
import statistics
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


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def summarize_numeric(values: list[float]) -> dict[str, float | int | None]:
    clean = [finite_float(value) for value in values if math.isfinite(finite_float(value))]
    if not clean:
        return {"count": 0, "min": None, "p50": None, "mean": None, "p90": None, "max": None}
    return {
        "count": len(clean),
        "min": float(min(clean)),
        "p50": percentile(clean, 50),
        "mean": float(statistics.fmean(clean)),
        "p90": percentile(clean, 90),
        "max": float(max(clean)),
    }


def group_candidates(rows: list[dict[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["frame_id"]), int(row["cam_id"]))].append(row)
    for items in grouped.values():
        items.sort(key=lambda row: finite_float(row.get("score")), reverse=True)
    return grouped


def direct_for_probe(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((row for row in rows if int(row.get("offset", 10**9)) == 0), None)


def source_records(candidates: list[dict[str, Any]], smooth_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = group_candidates(candidates)
    smooth_by_probe = {(int(row["frame_id"]), int(row["cam_id"])): row for row in smooth_rows}
    records: list[dict[str, Any]] = []
    for key in sorted(grouped):
        rows = grouped[key]
        direct = direct_for_probe(rows)
        independent = rows[0] if rows else None
        smooth = smooth_by_probe.get(key)
        for source, row in (
            ("direct", direct),
            ("independent_best", independent),
            ("smooth_path", smooth),
        ):
            if row is None:
                continue
            best_score = finite_float(independent.get("score")) if independent else 0.0
            item = dict(row)
            item["source"] = source
            item["score_loss_from_independent_best"] = best_score - finite_float(row.get("score"))
            records.append(item)
    return records


def summarize_by_source(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_source[str(row["source"])].append(row)
    for source, rows in sorted(by_source.items()):
        out[source] = {
            "count": len(rows),
            "score": summarize_numeric([finite_float(row.get("score")) for row in rows]),
            "raw_score": summarize_numeric([finite_float(row.get("raw_score")) for row in rows]),
            "sky_hit": summarize_numeric([finite_float(row.get("sky_hit")) for row in rows]),
            "offset": summarize_numeric([float(int(row.get("offset", 0))) for row in rows]),
            "abs_offset": summarize_numeric([abs(float(int(row.get("offset", 0)))) for row in rows]),
            "score_loss_from_independent_best": summarize_numeric([
                finite_float(row.get("score_loss_from_independent_best")) for row in rows
            ]),
        }
    return out


def smooth_temporal_summary(smooth_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cam: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in smooth_rows:
        by_cam[int(row["cam_id"])].append(row)
    out: dict[str, Any] = {}
    for cam_id, rows in sorted(by_cam.items()):
        rows.sort(key=lambda row: int(row["frame_id"]))
        ratios: list[float] = []
        for prev, cur in zip(rows, rows[1:]):
            df = max(int(cur["frame_id"]) - int(prev["frame_id"]), 1)
            dv = int(cur["video_idx"]) - int(prev["video_idx"])
            ratios.append(dv / float(df))
        out[str(cam_id)] = {
            "count": len(rows),
            "frame_to_video": [
                {"frame_id": int(row["frame_id"]), "video_idx": int(row["video_idx"]), "offset": int(row.get("offset", 0))}
                for row in rows
            ],
            "step_ratio": summarize_numeric(ratios),
        }
    return out


def risk_rows(records: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    risks = []
    for row in records:
        sky_hit = finite_float(row.get("sky_hit"))
        loss = finite_float(row.get("score_loss_from_independent_best"))
        score = finite_float(row.get("score"))
        risk = sky_hit * 2.0 + max(loss, 0.0) + max(0.0, 0.15 - score)
        if sky_hit >= 0.20 or loss >= 0.20 or score <= 0.05:
            risks.append({
                "source": row["source"],
                "frame_id": int(row["frame_id"]),
                "cam_id": int(row["cam_id"]),
                "video_idx": int(row["video_idx"]),
                "offset": int(row.get("offset", 0)),
                "score": score,
                "raw_score": finite_float(row.get("raw_score")),
                "sky_hit": sky_hit,
                "score_loss_from_independent_best": loss,
                "risk_score": risk,
            })
    risks.sort(key=lambda row: float(row["risk_score"]), reverse=True)
    return risks[:max_rows]


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Sync Option Source Summary",
        "",
        f"- candidates: `{report['candidate_count']}`",
        f"- smooth rows: `{report['smooth_count']}`",
        "",
        "## Source Metrics",
        "",
        "| source | count | score mean | sky_hit mean | sky_hit p90 | abs offset p50 | score loss mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for source, metrics in report["by_source"].items():
        lines.append(
            "| {source} | {count} | {score:.3f} | {sky:.3f} | {sky_p90:.3f} | {offset:.1f} | {loss:.3f} |".format(
                source=source,
                count=metrics["count"],
                score=metrics["score"]["mean"] or 0.0,
                sky=metrics["sky_hit"]["mean"] or 0.0,
                sky_p90=metrics["sky_hit"]["p90"] or 0.0,
                offset=metrics["abs_offset"]["p50"] or 0.0,
                loss=metrics["score_loss_from_independent_best"]["mean"] or 0.0,
            )
        )
    lines.extend(["", "## Top Risks", ""])
    for row in report["top_risks"][:20]:
        lines.append(
            "- {source} f={frame_id} cam={cam_id} v={video_idx} off={offset} "
            "score={score:.3f} sky={sky_hit:.3f} loss={score_loss_from_independent_best:.3f}".format(**row)
        )
    lines.append("")
    return "\n".join(lines)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    candidates = read_jsonl(args.candidates_jsonl)
    smooth_rows = read_jsonl(args.smooth_jsonl)
    records = source_records(candidates, smooth_rows)
    return {
        "candidates_jsonl": str(args.candidates_jsonl),
        "smooth_jsonl": str(args.smooth_jsonl),
        "candidate_count": len(candidates),
        "smooth_count": len(smooth_rows),
        "record_count": len(records),
        "by_source": summarize_by_source(records),
        "smooth_temporal": smooth_temporal_summary(smooth_rows),
        "top_risks": risk_rows(records, args.max_risks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-jsonl", type=Path, required=True)
    parser.add_argument("--smooth-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--max-risks", type=int, default=40)
    args = parser.parse_args()

    report = build_report(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({
        "output_json": str(args.output_json),
        "output_md": str(args.output_md) if args.output_md else None,
        "candidate_count": report["candidate_count"],
        "smooth_count": report["smooth_count"],
        "record_count": report["record_count"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
