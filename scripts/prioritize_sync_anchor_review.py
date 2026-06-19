#!/usr/bin/env python3
"""Prioritize manual sync-anchor review rows without auto-accepting anchors."""

from __future__ import annotations

import argparse
import html
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_option(option: dict[str, Any]) -> float:
    return float(option.get("score", 0.0))


def enrich_row(row: dict[str, Any]) -> dict[str, Any]:
    options = sorted(row.get("options", []), key=score_option, reverse=True)
    best = options[0] if options else {}
    second = options[1] if len(options) > 1 else {}
    direct = next((item for item in options if str(item.get("review_source")) == "direct"), None)
    smooth = next((item for item in options if str(item.get("review_source")) == "smooth_path"), None)
    score_margin = score_option(best) - score_option(second) if second else score_option(best)
    direct_score_gap = score_option(best) - score_option(direct) if direct else None
    smooth_score_gap = score_option(best) - score_option(smooth) if smooth else None
    offset_values = [int(item.get("offset", 0)) for item in options]
    offset_span = max(offset_values) - min(offset_values) if offset_values else 0
    best_source = str(best.get("review_source", "missing"))
    # Manual review is easiest when one option is visually/score-wise dominant.
    # Penalize huge offset ambiguity so the first review batch is not dominated
    # by degenerate edge-score matches.
    priority_score = (
        score_margin * 3.0
        + score_option(best) * 0.5
        - min(abs(int(best.get("offset", 0))) / 4000.0, 0.35)
        - min(offset_span / 8000.0, 0.25)
    )
    risk_reasons = []
    if score_margin < 0.05:
        risk_reasons.append("low_score_margin")
    if direct_score_gap is not None and direct_score_gap > 0.15:
        risk_reasons.append("direct_far_below_best")
    if abs(int(best.get("offset", 0))) >= 700:
        risk_reasons.append("large_best_offset")
    if offset_span >= 1000:
        risk_reasons.append("wide_offset_span")
    if best_source not in {"direct", "smooth_path", "independent_best", "top_candidate"}:
        risk_reasons.append("unknown_best_source")
    out = dict(row)
    out.update(
        {
            "options": options,
            "best_option_idx": best.get("option_idx"),
            "best_video_idx": best.get("video_idx"),
            "best_source": best_source,
            "best_score": score_option(best),
            "score_margin": score_margin,
            "direct_video_idx": direct.get("video_idx") if direct else None,
            "direct_score_gap": direct_score_gap,
            "smooth_video_idx": smooth.get("video_idx") if smooth else None,
            "smooth_score_gap": smooth_score_gap,
            "offset_span": offset_span,
            "priority_score": priority_score,
            "risk_reasons": risk_reasons,
        }
    )
    return out


def select_review_batch(rows: list[dict[str, Any]], per_cam: int) -> list[dict[str, Any]]:
    selected = []
    by_cam: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cam[int(row["cam_id"])].append(row)
    for cam_id in sorted(by_cam):
        cam_rows = sorted(by_cam[cam_id], key=lambda item: float(item["priority_score"]), reverse=True)
        selected.extend(cam_rows[:per_cam])
    return sorted(selected, key=lambda item: (int(item["frame_id"]), int(item["cam_id"])))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_html(rows: list[dict[str, Any]], source_dir: Path, title: str) -> str:
    cards = []
    for row in rows:
        options_html = []
        for option in row.get("options", []):
            panel = option.get("panel_path")
            src = panel
            if panel and source_dir:
                src = str((source_dir / panel).resolve())
            src_attr = html.escape(src or "")
            options_html.append(
                f"""
                <div class="option">
                  <img src="{src_attr}" loading="lazy">
                  <div class="meta">
                    opt {option.get('option_idx')} / {html.escape(str(option.get('review_source')))} /
                    video {option.get('video_idx')} / offset {option.get('offset')}<br>
                    score {float(option.get('score', 0.0)):.3f} /
                    edge {float(option.get('edge_hit', 0.0)):.3f} /
                    dist {float(option.get('edge_distance_mean', 0.0)):.2f}
                  </div>
                </div>
                """
            )
        risk = ", ".join(row.get("risk_reasons", [])) or "none"
        cards.append(
            f"""
            <section class="card">
              <h2>frame {row['frame_id']} / cam {row['cam_id']}</h2>
              <p>
                priority {float(row['priority_score']):.3f};
                best opt {row.get('best_option_idx')} ({html.escape(str(row.get('best_source')))});
                video {row.get('best_video_idx')};
                margin {float(row.get('score_margin', 0.0)):.3f};
                risk: {html.escape(risk)}
              </p>
              <div class="options">{''.join(options_html)}</div>
            </section>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ margin: 0; background: #0d1117; color: #d8dee9; font: 14px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
header {{ position: sticky; top: 0; padding: 14px 18px; background: #111827; border-bottom: 1px solid #30363d; }}
h1 {{ margin: 0; font-size: 18px; }}
main {{ padding: 16px; }}
.card {{ border: 1px solid #30363d; border-radius: 8px; margin-bottom: 18px; background: #151b23; overflow: hidden; }}
h2 {{ margin: 0; padding: 10px 12px; font-size: 15px; background: #1b2430; border-bottom: 1px solid #30363d; }}
p {{ margin: 10px 12px; color: #aab6c5; }}
.options {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 10px; padding: 12px; }}
.option {{ border: 1px solid #30363d; border-radius: 6px; overflow: hidden; background: #0d1117; }}
img {{ width: 100%; display: block; }}
.meta {{ padding: 8px; color: #b8c0cc; font-size: 12px; line-height: 1.45; }}
</style>
</head>
<body>
<header><h1>{html.escape(title)}</h1></header>
<main>{''.join(cards)}</main>
</body>
</html>"""


def build(args: argparse.Namespace) -> dict[str, Any]:
    rows = [enrich_row(row) for row in read_jsonl(args.manifest)]
    rows_sorted = sorted(rows, key=lambda item: float(item["priority_score"]), reverse=True)
    selected = select_review_batch(rows, args.per_cam)
    write_jsonl(args.output_dir / "anchor_review_priority_all.jsonl", rows_sorted)
    write_jsonl(args.output_dir / "anchor_review_priority_batch.jsonl", selected)
    html_text = render_html(selected, args.source_dir or args.manifest.parent, "Prioritized Sync Anchor Review")
    (args.output_dir / "anchor_review_priority.html").write_text(html_text, encoding="utf-8")
    margins = [float(row["score_margin"]) for row in rows]
    by_cam = Counter(int(row["cam_id"]) for row in selected)
    report = {
        "manifest": str(args.manifest),
        "output_dir": str(args.output_dir),
        "row_count": len(rows),
        "selected_count": len(selected),
        "selected_by_cam": {str(k): int(v) for k, v in sorted(by_cam.items())},
        "per_cam": int(args.per_cam),
        "score_margin": {
            "min": min(margins) if margins else None,
            "p50": statistics.median(margins) if margins else None,
            "max": max(margins) if margins else None,
        },
        "top_rows": [
            {
                "frame_id": int(row["frame_id"]),
                "cam_id": int(row["cam_id"]),
                "best_video_idx": row.get("best_video_idx"),
                "best_source": row.get("best_source"),
                "score_margin": row.get("score_margin"),
                "priority_score": row.get("priority_score"),
                "risk_reasons": row.get("risk_reasons", []),
            }
            for row in rows_sorted[:10]
        ],
        "html": str(args.output_dir / "anchor_review_priority.html"),
    }
    (args.output_dir / "anchor_review_priority_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, help="Directory containing panel paths from the manifest.")
    parser.add_argument("--per-cam", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
