#!/usr/bin/env python3
"""Build a manual anchor review pack for LiDAR/video synchronization.

Inputs are candidate scores from `calibrate_lx_video_frame_mapping.py` and,
optionally, a smooth path from `solve_sync_path_from_candidates.py`.  For each
section/camera probe, this renders a compact set of choices:

- direct index candidate (`offset == 0`)
- independent best score
- smooth-path candidate
- top-N remaining candidates

The output is intentionally human-reviewable.  The generated
`manual_anchor_manifest.jsonl` is a template: fill `selected_video_idx` and
`anchor_status` for reliable anchors, then use it to constrain the next sync
optimizer.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from calibrate_lx_video_frame_mapping import (
    CandidateScore,
    draw_panel,
    project_points,
    read_frame,
    undistort_maps,
    visible_pixels,
)
from project_priority_masks_to_lx import read_lx_points, read_lx_sections


def load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_by_probe(rows: list[dict[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["frame_id"]), int(row["cam_id"])), []).append(row)
    for items in grouped.values():
        items.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
    return grouped


def row_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (int(row["frame_id"]), int(row["cam_id"]), int(row["video_idx"]))


def choose_review_options(
    candidates: list[dict[str, Any]],
    smooth_row: dict[str, Any] | None,
    top_n: int,
) -> list[dict[str, Any]]:
    """Select deduplicated review options in stable priority order."""
    if not candidates:
        return []
    options: list[dict[str, Any]] = []

    def add(row: dict[str, Any], source: str) -> None:
        item = dict(row)
        item["review_source"] = source
        key = row_key(item)
        if key not in {row_key(existing) for existing in options}:
            options.append(item)

    direct = next((row for row in candidates if int(row.get("offset", 10**9)) == 0), None)
    if direct is not None:
        add(direct, "direct")
    add(candidates[0], "independent_best")
    if smooth_row is not None:
        add(smooth_row, "smooth_path")
    for row in candidates[:top_n]:
        add(row, "top_candidate")
    return options


def panel_filename(frame_id: int, cam_id: int, option_idx: int, row: dict[str, Any]) -> str:
    source = str(row.get("review_source", "candidate")).replace("/", "_")
    video_idx = int(row["video_idx"])
    return f"frame_{frame_id:06d}_cam{cam_id}_opt{option_idx}_{source}_v{video_idx:06d}.jpg"


def build_review_html(manifest_rows: list[dict[str, Any]]) -> str:
    """Build a static review UI that exports accepted anchors as JSONL."""
    payload = json.dumps(manifest_rows, ensure_ascii=False)
    escaped_payload = html.escape(payload, quote=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LiDAR/Video Sync Anchor Review</title>
  <style>
    :root {{
      color-scheme: dark;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0e1117;
      color: #d8dee9;
    }}
    body {{ margin: 0; }}
    header {{
      position: sticky;
      top: 0;
      z-index: 2;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 18px;
      background: rgba(14, 17, 23, 0.96);
      border-bottom: 1px solid #2d3440;
    }}
    h1 {{ margin: 0; font-size: 18px; font-weight: 650; }}
    button {{
      border: 1px solid #42526a;
      border-radius: 6px;
      background: #1d2736;
      color: #f2f4f8;
      padding: 8px 12px;
      cursor: pointer;
    }}
    main {{ padding: 16px; }}
    #coverage {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      padding: 10px 18px;
      background: #0f1722;
      border-bottom: 1px solid #2d3440;
      color: #b8c0cc;
    }}
    .pill {{ border: 1px solid #39485c; border-radius: 999px; padding: 4px 9px; background: #151f2c; }}
    .pill.ok {{ border-color: #2f8f5b; color: #8bd49c; }}
    .pill.bad {{ border-color: #9f4a4a; color: #ff9b9b; }}
    .probe {{
      border: 1px solid #2d3440;
      border-radius: 8px;
      margin-bottom: 18px;
      background: #151922;
      overflow: hidden;
    }}
    .probe-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 12px;
      border-bottom: 1px solid #2d3440;
      background: #171d28;
    }}
    .probe-title {{ font-weight: 650; }}
    .status {{ display: flex; align-items: center; gap: 10px; color: #aeb8c6; }}
    .options {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 12px;
      padding: 12px;
    }}
    label.option {{
      display: block;
      border: 1px solid #303948;
      border-radius: 7px;
      background: #10151f;
      overflow: hidden;
    }}
    label.option:has(input:checked) {{ border-color: #6aa5ff; box-shadow: 0 0 0 1px #6aa5ff; }}
    .meta {{ padding: 8px 10px; font-size: 12px; color: #b8c0cc; line-height: 1.5; }}
    img {{ display: block; width: 100%; height: auto; background: #05070a; }}
    input[type="text"] {{
      width: 280px;
      max-width: 38vw;
      border: 1px solid #3b4656;
      border-radius: 6px;
      background: #0d1118;
      color: #d8dee9;
      padding: 6px 8px;
    }}
    .muted {{ color: #8994a5; }}
  </style>
</head>
<body>
  <header>
    <h1>LiDAR/Video Sync Anchor Review</h1>
    <div>
      <button id="accept-selected">Mark selected accepted</button>
      <button id="download">Export accepted JSONL</button>
    </div>
  </header>
  <div id="coverage"></div>
  <main id="app"></main>
  <script type="application/json" id="manifest-json">{escaped_payload}</script>
  <script>
    const rows = JSON.parse(document.getElementById('manifest-json').textContent);
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

    function render() {{
      app.innerHTML = '';
      renderCoverage();
      rows.forEach((row, rowIdx) => {{
        const probe = document.createElement('section');
        probe.className = 'probe';
        probe.id = `frame-${{row.frame_id}}-cam-${{row.cam_id}}`;
        const title = document.createElement('div');
        title.className = 'probe-head';
        title.innerHTML = `
          <div class="probe-title">frame ${{row.frame_id}} / cam ${{row.cam_id}}</div>
          <div class="status">
            <label><input type="radio" name="status-${{rowIdx}}" value="accepted" ${{row.anchor_status === 'accepted' ? 'checked' : ''}}> accepted</label>
            <label><input type="radio" name="status-${{rowIdx}}" value="rejected" ${{row.anchor_status === 'rejected' ? 'checked' : ''}}> rejected</label>
            <label><input type="radio" name="status-${{rowIdx}}" value="unreviewed" ${{row.anchor_status === 'unreviewed' ? 'checked' : ''}}> unreviewed</label>
            <input type="text" placeholder="notes" value="${{row.notes || ''}}">
          </div>`;
        title.querySelectorAll('input[type="radio"]').forEach(input => {{
          input.addEventListener('change', event => {{
            row.anchor_status = event.target.value;
            renderCoverage();
          }});
        }});
        title.querySelector('input[type="text"]').addEventListener('input', event => row.notes = event.target.value);
        probe.appendChild(title);

        const options = document.createElement('div');
        options.className = 'options';
        row.options.forEach(option => {{
          const label = document.createElement('label');
          label.className = 'option';
          const checked = row.selected_option_idx === option.option_idx ? 'checked' : '';
          const imageHtml = option.panel_path ? `<img src="${{option.panel_path}}" loading="lazy" alt="frame ${{row.frame_id}} cam ${{row.cam_id}} option ${{option.option_idx}}">` : '<div class="meta muted">panel missing</div>';
          label.innerHTML = `
            ${{imageHtml}}
            <div class="meta">
              <input type="radio" name="option-${{rowIdx}}" value="${{option.option_idx}}" ${{checked}}>
              option ${{option.option_idx}} / ${{option.review_source}}<br>
              video ${{option.video_idx}} / offset ${{option.offset}} / score ${{Number(option.score).toFixed(3)}}<br>
              edge hit ${{Number(option.edge_hit).toFixed(3)}} / mean dist ${{Number(option.edge_distance_mean).toFixed(2)}}<br>
              prior exp ${{option.absolute_expected_video_idx === undefined ? 'n/a' : Number(option.absolute_expected_video_idx).toFixed(1)}} /
              err ${{option.absolute_prior_error === undefined ? 'n/a' : Number(option.absolute_prior_error).toFixed(1)}}
            </div>`;
          label.querySelector('input').addEventListener('change', () => {{
            row.selected_option_idx = option.option_idx;
            row.selected_video_idx = option.video_idx;
            renderCoverage();
          }});
          options.appendChild(label);
        }});
        probe.appendChild(options);
        app.appendChild(probe);
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
</html>
"""


def make_sheet(panels: list[np.ndarray], output: Path, cols: int) -> None:
    if not panels:
        return
    while len(panels) % cols:
        panels.append(np.zeros_like(panels[0]))
    rows = [np.hstack(panels[i:i + cols]) for i in range(0, len(panels), cols)]
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.vstack(rows))


def render_option(
    lx_handle,
    sections: list[dict[str, Any]],
    poses: dict[int, dict[str, Any]],
    maps: dict[int, tuple[np.ndarray, np.ndarray]],
    caps: dict[int, cv2.VideoCapture],
    row: dict[str, Any],
    dot_px: int,
) -> np.ndarray | None:
    frame_id = int(row["frame_id"])
    cam_id = int(row["cam_id"])
    video_idx = int(row["video_idx"])
    if frame_id >= len(sections) or frame_id not in poses:
        return None
    points = read_lx_points(lx_handle, sections[frame_id])
    u, v, z = project_points(points, poses[frame_id], cam_id, 0.1)
    raw = read_frame(caps[cam_id], video_idx)
    if raw is None:
        return None
    map1, map2 = maps[cam_id]
    image = cv2.remap(raw, map1, map2, cv2.INTER_LINEAR)
    uu, vv, _depth = visible_pixels(u, v, z, image.shape[1], image.shape[0])
    score = CandidateScore(
        frame_id=frame_id,
        cam_id=cam_id,
        video_idx=video_idx,
        offset=int(row.get("offset", video_idx - frame_id)),
        visible=int(row.get("visible", len(uu))),
        edge_hit=float(row.get("edge_hit", 0.0)),
        edge_distance_mean=float(row.get("edge_distance_mean", 0.0)),
        edge_distance_p50=float(row.get("edge_distance_p50", 0.0)),
        score=float(row.get("score", 0.0)),
    )
    panel = draw_panel(image, uu, vv, score, dot_px)
    source = str(row.get("review_source", "candidate"))
    cv2.putText(panel, source, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(panel, source, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lx-file", type=Path, required=True)
    parser.add_argument("--candidates-jsonl", type=Path, required=True)
    parser.add_argument("--smooth-path-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=4)
    parser.add_argument("--sheet-cols", type=int, default=4)
    parser.add_argument("--dot-px", type=int, default=7)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = group_by_probe(load_jsonl(args.candidates_jsonl))
    smooth = {row_key(row): row for row in load_jsonl(args.smooth_path_jsonl)}
    sections = read_lx_sections(args.lx_file)
    frame_ids = sorted({key[0] for key in candidates})
    poses = {row["frame_id"]: row for row in config.load_img_pos(min(frame_ids), max(frame_ids))}
    maps = {cam_id: undistort_maps(cam_id) for cam_id in config.VIDEO_FILES}
    caps = {cam_id: cv2.VideoCapture(path) for cam_id, path in config.VIDEO_FILES.items()}
    for cam_id, cap in caps.items():
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video cam{cam_id}: {config.VIDEO_FILES[cam_id]}")

    manifest_rows = []
    panels = []
    panels_dir = args.output_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    with args.lx_file.open("rb") as lx_handle:
        for key in sorted(candidates):
            frame_id, cam_id = key
            smooth_row = next((row for row in smooth.values() if int(row["frame_id"]) == frame_id and int(row["cam_id"]) == cam_id), None)
            options = choose_review_options(candidates[key], smooth_row, args.top_n)
            option_records = []
            for option_idx, row in enumerate(options):
                record = {
                    "option_idx": option_idx,
                    "review_source": row.get("review_source"),
                    "video_idx": int(row["video_idx"]),
                    "offset": int(row.get("offset", int(row["video_idx"]) - frame_id)),
                    "score": float(row.get("score", 0.0)),
                    "edge_hit": float(row.get("edge_hit", 0.0)),
                    "edge_distance_mean": float(row.get("edge_distance_mean", 0.0)),
                }
                if row.get("absolute_expected_video_idx") is not None:
                    record["absolute_expected_video_idx"] = float(row["absolute_expected_video_idx"])
                if row.get("absolute_prior_error") is not None:
                    record["absolute_prior_error"] = float(row["absolute_prior_error"])
                panel = render_option(lx_handle, sections, poses, maps, caps, row, args.dot_px)
                if panel is not None:
                    rel_panel_path = Path("panels") / panel_filename(frame_id, cam_id, option_idx, row)
                    cv2.imwrite(str(args.output_dir / rel_panel_path), panel)
                    record["panel_path"] = rel_panel_path.as_posix()
                    panels.append(panel)
                option_records.append(record)
            manifest_rows.append({
                "frame_id": frame_id,
                "cam_id": cam_id,
                "anchor_status": "unreviewed",
                "selected_video_idx": None,
                "selected_option_idx": None,
                "notes": "",
                "options": option_records,
            })
    for cap in caps.values():
        cap.release()

    with (args.output_dir / "manual_anchor_manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "manual_anchor_review.html").write_text(
        build_review_html(manifest_rows),
        encoding="utf-8",
    )
    report = {
        "candidates_jsonl": str(args.candidates_jsonl),
        "smooth_path_jsonl": str(args.smooth_path_jsonl) if args.smooth_path_jsonl else None,
        "probe_count": len(manifest_rows),
        "panel_count": len(panels),
        "html": str(args.output_dir / "manual_anchor_review.html"),
        "panels_dir": str(panels_dir),
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "manual_anchor_review_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    make_sheet(panels, args.output_dir / "manual_anchor_review_sheet.jpg", args.sheet_cols)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
