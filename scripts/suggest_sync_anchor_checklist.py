#!/usr/bin/env python3
"""Build a minimal manual checklist from sync anchor review rows.

This script does not auto-accept anchors. It ranks review rows so a human can
quickly inspect a small, temporally spread set before exporting accepted anchors
from the interactive review page.
"""

from __future__ import annotations

import argparse
import html
import json
import os
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
    frame_id = int(row["frame_id"])
    cam_id = int(row["cam_id"])
    return {
        "frame_id": frame_id,
        "cam_id": cam_id,
        "review_anchor": f"frame-{frame_id}-cam-{cam_id}",
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
        review_ref = f"{review_url}#{row['review_anchor']}" if review_url else ""
        frame_text = f"[{row['frame_id']}]({review_ref})" if review_ref else str(row["frame_id"])
        lines.append(
            f"| {frame_text} | {row['cam_id']} | {row.get('recommended_video_idx')} | "
            f"{row.get('recommended_source')} | {float(row.get('score') or 0.0):.3f} | "
            f"{float(row.get('score_margin') or 0.0):.3f} | {risk} | `{panel}` |"
        )
    return "\n".join(lines) + "\n"


def rows_for_html(rows: list[dict[str, Any]], source_dir: Path, output_html: Path, review_url: str | None) -> list[dict[str, Any]]:
    out = []
    output_dir = output_html.parent
    for row in rows:
        item = public_row(row)
        item["review_href"] = f"{review_url}#{item['review_anchor']}" if review_url else ""
        item["options"] = []
        for option in sorted(row.get("options", []), key=lambda opt: float(opt.get("score", 0.0)), reverse=True):
            opt = dict(option)
            panel = opt.get("panel_path")
            opt["panel_src"] = os.path.relpath(source_dir / str(panel), output_dir) if panel else ""
            opt["is_recommended"] = opt.get("option_idx") == item.get("recommended_option_idx")
            item["options"].append(opt)
        out.append(item)
    return out


def render_html(rows: list[dict[str, Any]], source_dir: Path, output_html: Path, review_url: str | None) -> str:
    payload = html.escape(json.dumps(rows_for_html(rows, source_dir, output_html, review_url), ensure_ascii=False), quote=False)
    review_link = f'<a href="{html.escape(review_url)}">open main review page</a>' if review_url else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sync Anchor Checklist</title>
<style>
body {{ margin: 0; background: #0d1117; color: #d8dee9; font: 14px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
header {{ position: sticky; top: 0; z-index: 2; padding: 14px 18px; background: #111827; border-bottom: 1px solid #30363d; }}
h1 {{ margin: 0 0 8px; font-size: 18px; }}
a {{ color: #8ab4ff; }}
main {{ padding: 16px; }}
.notice {{ color: #ffcc66; }}
.card {{ border: 1px solid #30363d; border-radius: 8px; margin-bottom: 18px; background: #151b23; overflow: hidden; }}
h2 {{ margin: 0; padding: 10px 12px; font-size: 15px; background: #1b2430; border-bottom: 1px solid #30363d; }}
.summary {{ padding: 10px 12px; color: #aab6c5; border-bottom: 1px solid #30363d; }}
.risk {{ color: #d29922; }}
.options {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 10px; padding: 12px; }}
.option {{ border: 1px solid #30363d; border-radius: 6px; overflow: hidden; background: #0d1117; }}
.option.recommended {{ border-color: #2f8f5b; box-shadow: 0 0 0 1px #2f8f5b; }}
img {{ width: 100%; display: block; }}
.meta {{ padding: 8px; color: #b8c0cc; font-size: 12px; line-height: 1.45; }}
.tag {{ display: inline-block; border: 1px solid #3b4a60; border-radius: 999px; padding: 2px 7px; margin-right: 5px; }}
.tag.rec {{ color: #8bd49c; border-color: #2f8f5b; }}
</style>
</head>
<body>
<header>
  <h1>Sync Anchor Checklist</h1>
  <div class="notice">Manual review aid only. Do not stage these suggestions directly. {review_link}</div>
</header>
<main id="app"></main>
<script type="application/json" id="rows-json">{payload}</script>
<script>
const rows = JSON.parse(document.getElementById('rows-json').textContent);
const app = document.getElementById('app');
function fmt(value, digits=3) {{
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'n/a';
  return Number(value).toFixed(digits);
}}
rows.forEach(row => {{
  const risk = row.risk_reasons && row.risk_reasons.length ? row.risk_reasons.join(', ') : 'none';
  const card = document.createElement('section');
  card.className = 'card';
  card.innerHTML = `
    <h2>frame ${{row.frame_id}} / cam ${{row.cam_id}}</h2>
    <div class="summary">
      suggested video ${{row.recommended_video_idx}} /
      source ${{row.recommended_source}} /
      score ${{fmt(row.score)}} /
      margin ${{fmt(row.score_margin)}} /
      risk <span class="risk">${{risk}}</span><br>
      ${{row.review_href ? `<a href="${{row.review_href}}">jump to main review row</a>` : ''}}
    </div>
    <div class="options"></div>
  `;
  const options = card.querySelector('.options');
  row.options.forEach(option => {{
    const div = document.createElement('div');
    div.className = `option ${{option.is_recommended ? 'recommended' : ''}}`;
    const image = option.panel_src ? `<img src="${{option.panel_src}}" loading="lazy">` : '';
    div.innerHTML = `
      ${{image}}
      <div class="meta">
        ${{option.is_recommended ? '<span class="tag rec">recommended</span>' : ''}}
        <span class="tag">opt ${{option.option_idx}}</span>
        <span class="tag">${{option.review_source}}</span><br>
        video ${{option.video_idx}} / offset ${{option.offset}}<br>
        score ${{fmt(option.score)}} /
        edge ${{fmt(option.edge_hit)}} /
        dist ${{fmt(option.edge_distance_mean, 2)}}<br>
        prior exp ${{fmt(option.absolute_expected_video_idx, 1)}} /
        err ${{fmt(option.absolute_prior_error, 1)}}
      </div>
    `;
    options.appendChild(div);
  }});
  app.appendChild(card);
}});
</script>
</body>
</html>"""


def build(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.review_jsonl)
    selected_full = select_rows(rows, args.per_cam, args.bins)
    selected = [public_row(row) for row in selected_full]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.output_md.write_text(render_markdown(selected, args.review_url), encoding="utf-8")
    if args.output_html:
        args.output_html.parent.mkdir(parents=True, exist_ok=True)
        source_dir = args.source_dir or args.review_jsonl.parent
        args.output_html.write_text(render_html(selected_full, source_dir, args.output_html, args.review_url), encoding="utf-8")
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
        "output_html": str(args.output_html) if args.output_html else None,
        "diagnostic_accepted_jsonl": str(args.diagnostic_accepted_jsonl) if args.diagnostic_accepted_jsonl else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-html", type=Path)
    parser.add_argument("--source-dir", type=Path, help="Directory containing panel paths from the review JSONL.")
    parser.add_argument("--per-cam", type=int, default=3)
    parser.add_argument("--bins", type=int, default=3)
    parser.add_argument("--review-url", default="")
    parser.add_argument("--diagnostic-accepted-jsonl", type=Path,
                        help="Optional diagnostic-only accepted-anchor JSONL for validator prechecks. Do not stage directly.")
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
