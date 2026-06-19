#!/usr/bin/env python3
"""Prioritize manual sync-anchor review rows without auto-accepting anchors."""

from __future__ import annotations

import argparse
import html
import json
import os
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


def preselect_option(row: dict[str, Any], source: str | None) -> dict[str, Any]:
    if not source:
        return row
    out = dict(row)
    options = list(out.get("options", []))
    for option in options:
        if str(option.get("review_source")) == source:
            out["selected_option_idx"] = option.get("option_idx")
            out["selected_video_idx"] = option.get("video_idx")
            out["preselected_source"] = source
            return out
    out["preselected_source"] = None
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def rows_for_html(rows: list[dict[str, Any]], source_dir: Path, output_dir: Path) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        item["options"] = []
        for option in row.get("options", []):
            opt = dict(option)
            panel = opt.get("panel_path")
            if panel:
                opt["panel_src"] = os.path.relpath(source_dir / str(panel), output_dir)
            else:
                opt["panel_src"] = ""
            item["options"].append(opt)
        out.append(item)
    return out


def render_html(rows: list[dict[str, Any]], source_dir: Path, output_dir: Path, title: str) -> str:
    payload_rows = rows_for_html(rows, source_dir, output_dir)
    payload = html.escape(json.dumps(payload_rows, ensure_ascii=False), quote=False)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ margin: 0; background: #0d1117; color: #d8dee9; font: 14px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
header {{ position: sticky; top: 0; z-index: 2; display: flex; justify-content: space-between; align-items: center; gap: 16px; padding: 14px 18px; background: #111827; border-bottom: 1px solid #30363d; }}
h1 {{ margin: 0; font-size: 18px; }}
button {{ border: 1px solid #42526a; border-radius: 6px; background: #1d2736; color: #f2f4f8; padding: 8px 12px; cursor: pointer; }}
main {{ padding: 16px; }}
#coverage {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; padding: 10px 18px; background: #0f1722; border-bottom: 1px solid #30363d; color: #b8c0cc; }}
.pill {{ border: 1px solid #39485c; border-radius: 999px; padding: 4px 9px; background: #151f2c; }}
.pill.ok {{ border-color: #2f8f5b; color: #8bd49c; }}
.pill.bad {{ border-color: #9f4a4a; color: #ff9b9b; }}
.card {{ border: 1px solid #30363d; border-radius: 8px; margin-bottom: 18px; background: #151b23; overflow: hidden; }}
h2 {{ margin: 0; padding: 10px 12px; font-size: 15px; background: #1b2430; border-bottom: 1px solid #30363d; }}
p {{ margin: 10px 12px; color: #aab6c5; }}
.options {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 10px; padding: 12px; }}
.option {{ border: 1px solid #30363d; border-radius: 6px; overflow: hidden; background: #0d1117; }}
.option:has(input:checked) {{ border-color: #6aa5ff; box-shadow: 0 0 0 1px #6aa5ff; }}
img {{ width: 100%; display: block; }}
.meta {{ padding: 8px; color: #b8c0cc; font-size: 12px; line-height: 1.45; }}
.status {{ display: flex; gap: 10px; padding: 8px 12px; color: #aab6c5; border-bottom: 1px solid #30363d; }}
.risk {{ color: #d29922; }}
</style>
</head>
<body>
<header>
  <h1>{html.escape(title)}</h1>
  <div>
    <button id="accept-selected">Mark selected accepted</button>
    <button id="download">Export accepted JSONL</button>
  </div>
</header>
<div id="coverage"></div>
<main id="app"></main>
<script type="application/json" id="rows-json">{payload}</script>
<script>
const rows = JSON.parse(document.getElementById('rows-json').textContent);
const app = document.getElementById('app');
const coverage = document.getElementById('coverage');
const minAcceptedPerCam = 2;

function acceptedRows() {{
  return rows.filter(row => row.anchor_status === 'accepted' && row.selected_video_idx !== null && row.selected_video_idx !== undefined);
}}

function coverageState() {{
  const cams = [...new Set(rows.map(row => Number(row.cam_id)))].sort((a, b) => a - b);
  const accepted = acceptedRows();
  const byCam = new Map(cams.map(cam => [cam, 0]));
  accepted.forEach(row => byCam.set(Number(row.cam_id), (byCam.get(Number(row.cam_id)) || 0) + 1));
  const missingSelection = rows.filter(row => row.anchor_status === 'accepted' && (row.selected_video_idx === null || row.selected_video_idx === undefined));
  const sequenceIssues = [];
  cams.forEach(cam => {{
    const camRows = accepted
      .filter(row => Number(row.cam_id) === cam)
      .sort((a, b) => Number(a.frame_id) - Number(b.frame_id));
    const seenFrames = new Set();
    for (let i = 0; i < camRows.length; i += 1) {{
      const row = camRows[i];
      const frame = Number(row.frame_id);
      if (seenFrames.has(frame)) {{
        sequenceIssues.push(`cam${{cam}} duplicate frame ${{frame}}`);
      }}
      seenFrames.add(frame);
      if (i > 0) {{
        const prev = camRows[i - 1];
        if (Number(row.selected_video_idx) < Number(prev.selected_video_idx)) {{
          sequenceIssues.push(`cam${{cam}} video decreases: frame ${{prev.frame_id}}->${{row.frame_id}} video ${{prev.selected_video_idx}}->${{row.selected_video_idx}}`);
        }}
      }}
    }}
  }});
  const enough = cams.every(cam => (byCam.get(cam) || 0) >= minAcceptedPerCam);
  return {{cams, accepted, byCam, missingSelection, sequenceIssues, enough}};
}}

function renderCoverage() {{
  const state = coverageState();
  const camPills = state.cams.map(cam => {{
    const count = state.byCam.get(cam) || 0;
    const cls = count >= minAcceptedPerCam ? 'ok' : 'bad';
    return `<span class="pill ${{cls}}">cam${{cam}} accepted ${{count}}/${{minAcceptedPerCam}}</span>`;
  }}).join('');
  const missing = state.missingSelection.length;
  const sequenceIssueCount = state.sequenceIssues.length;
  coverage.innerHTML = `
    <span class="pill ${{state.enough && missing === 0 && sequenceIssueCount === 0 ? 'ok' : 'bad'}}">export readiness: ${{state.enough && missing === 0 && sequenceIssueCount === 0 ? 'ok' : 'not ready'}}</span>
    <span class="pill">accepted total ${{state.accepted.length}}</span>
    ${{camPills}}
    <span class="pill ${{missing ? 'bad' : 'ok'}}">accepted rows missing option ${{missing}}</span>
    <span class="pill ${{sequenceIssueCount ? 'bad' : 'ok'}}" title="${{state.sequenceIssues.join('\\n')}}">sequence issues ${{sequenceIssueCount}}</span>
  `;
}}

function optionLabel(option) {{
  return `opt ${{option.option_idx}} / ${{option.review_source}} / video ${{option.video_idx}} / offset ${{option.offset}}`;
}}

function render() {{
  app.innerHTML = '';
  renderCoverage();
  rows.forEach((row, rowIdx) => {{
    const card = document.createElement('section');
    card.className = 'card';
    const risk = row.risk_reasons && row.risk_reasons.length ? row.risk_reasons.join(', ') : 'none';
    card.innerHTML = `
      <h2>frame ${{row.frame_id}} / cam ${{row.cam_id}}</h2>
      <div class="status">
        <label><input type="radio" name="status-${{rowIdx}}" value="accepted" ${{row.anchor_status === 'accepted' ? 'checked' : ''}}> accepted</label>
        <label><input type="radio" name="status-${{rowIdx}}" value="rejected" ${{row.anchor_status === 'rejected' ? 'checked' : ''}}> rejected</label>
        <label><input type="radio" name="status-${{rowIdx}}" value="unreviewed" ${{row.anchor_status === 'unreviewed' ? 'checked' : ''}}> unreviewed</label>
      </div>
      <p>
        priority ${{Number(row.priority_score).toFixed(3)}};
        best opt ${{row.best_option_idx}} (${{row.best_source}});
        video ${{row.best_video_idx}};
        margin ${{Number(row.score_margin).toFixed(3)}};
        risk: <span class="risk">${{risk}}</span>
      </p>
      <div class="options"></div>
    `;
    card.querySelectorAll('input[type="radio"][name^="status-"]').forEach(input => {{
      input.addEventListener('change', event => {{
        row.anchor_status = event.target.value;
        renderCoverage();
      }});
    }});
    const options = card.querySelector('.options');
    row.options.forEach(option => {{
      const label = document.createElement('label');
      label.className = 'option';
      const checked = row.selected_option_idx === option.option_idx ? 'checked' : '';
      const image = option.panel_src ? `<img src="${{option.panel_src}}" loading="lazy">` : '';
      label.innerHTML = `
        ${{image}}
        <div class="meta">
          <input type="radio" name="option-${{rowIdx}}" value="${{option.option_idx}}" ${{checked}}>
          ${{optionLabel(option)}}<br>
          score ${{Number(option.score).toFixed(3)}} /
          edge ${{Number(option.edge_hit || 0).toFixed(3)}} /
          dist ${{Number(option.edge_distance_mean || 0).toFixed(2)}}<br>
          prior exp ${{option.absolute_expected_video_idx === undefined ? 'n/a' : Number(option.absolute_expected_video_idx).toFixed(1)}} /
          err ${{option.absolute_prior_error === undefined ? 'n/a' : Number(option.absolute_prior_error).toFixed(1)}}
        </div>
      `;
      label.querySelector('input').addEventListener('change', () => {{
        row.selected_option_idx = option.option_idx;
        row.selected_video_idx = option.video_idx;
        renderCoverage();
      }});
      options.appendChild(label);
    }});
    app.appendChild(card);
  }});
}}

document.getElementById('accept-selected').addEventListener('click', () => {{
  rows.forEach(row => {{
    if (row.selected_option_idx !== null && row.selected_option_idx !== undefined) {{
      row.anchor_status = 'accepted';
    }}
  }});
  render();
}});

document.getElementById('download').addEventListener('click', () => {{
  const state = coverageState();
  if ((!state.enough || state.missingSelection.length || state.sequenceIssues.length) && !window.confirm('Anchor coverage/sequence is not ready. Export anyway?')) {{
    return;
  }}
  const accepted = acceptedRows();
  const text = accepted.map(row => JSON.stringify(row)).join('\\n') + (accepted.length ? '\\n' : '');
  const blob = new Blob([text], {{type: 'application/x-ndjson'}});
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = 'accepted_sync_anchors.jsonl';
  link.click();
  URL.revokeObjectURL(link.href);
}});

render();
</script>
</body>
</html>"""


def build(args: argparse.Namespace) -> dict[str, Any]:
    rows = [enrich_row(row) for row in read_jsonl(args.manifest)]
    rows_sorted = sorted(rows, key=lambda item: float(item["priority_score"]), reverse=True)
    selected = [preselect_option(row, args.preselect_source) for row in select_review_batch(rows, args.per_cam)]
    write_jsonl(args.output_dir / "anchor_review_priority_all.jsonl", rows_sorted)
    write_jsonl(args.output_dir / "anchor_review_priority_batch.jsonl", selected)
    html_text = render_html(
        selected,
        args.source_dir or args.manifest.parent,
        args.output_dir,
        "Prioritized Sync Anchor Review",
    )
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
        "preselect_source": args.preselect_source,
        "preselected_count": sum(1 for row in selected if row.get("preselected_source")),
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
    parser.add_argument("--preselect-source",
                        help="Preselect this review_source in the HTML/export payload without marking rows accepted.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
