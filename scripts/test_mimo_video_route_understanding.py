#!/usr/bin/env python3
"""Probe whether Mimo can understand a full camera route from video evidence.

The script avoids committing any credentials. It reads MIMO_API_KEY from the
environment, samples a cam video uniformly, sends one global contact sheet, and
then asks frame-local questions with the global summary as context.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

import cv2
from PIL import Image, ImageDraw, ImageFont


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def read_video_meta(video: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {
        "video": str(video),
        "size_bytes": video.stat().st_size,
        "fps": fps,
        "frames": frames,
        "width": width,
        "height": height,
        "duration_sec": frames / fps if fps > 0 else None,
    }


def frame_indices(total_frames: int, count: int) -> list[int]:
    if count <= 1:
        return [0]
    return [round(i * (total_frames - 1) / (count - 1)) for i in range(count)]


def extract_frames(video: Path, out_dir: Path, count: int, long_edge: int) -> list[dict[str, Any]]:
    meta = read_video_meta(video)
    indices = frame_indices(int(meta["frames"]), count)
    cap = cv2.VideoCapture(str(video))
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for rank, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        scale = min(1.0, long_edge / max(h, w))
        if scale < 1.0:
            frame = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        path = out_dir / f"cam1_sample_{rank:02d}_frame_{idx:06d}.jpg"
        Image.fromarray(rgb).save(path, quality=90)
        records.append({
            "rank": rank,
            "frame_index": idx,
            "time_sec": idx / float(meta["fps"]) if meta["fps"] else None,
            "path": str(path),
        })
    cap.release()
    return records


def make_contact_sheet(records: list[dict[str, Any]], output: Path, columns: int, thumb_width: int) -> Path:
    images = [Image.open(row["path"]).convert("RGB") for row in records]
    thumbs: list[Image.Image] = []
    label_h = 42
    for row, img in zip(records, images):
        scale = thumb_width / img.width
        thumb = img.resize((thumb_width, round(img.height * scale)), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb.width, thumb.height + label_h), (18, 18, 18))
        canvas.paste(thumb, (0, label_h))
        draw = ImageDraw.Draw(canvas)
        text = f"#{row['rank']:02d} frame={row['frame_index']} t={row['time_sec']:.1f}s"
        draw.text((8, 10), text, fill=(255, 255, 255), font=ImageFont.load_default())
        thumbs.append(canvas)
    rows = (len(thumbs) + columns - 1) // columns
    cell_w = max(t.width for t in thumbs)
    cell_h = max(t.height for t in thumbs)
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), (8, 8, 8))
    for i, thumb in enumerate(thumbs):
        x = (i % columns) * cell_w
        y = (i // columns) * cell_h
        sheet.paste(thumb, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)
    return output


def chat(
    base_url: str,
    api_key: str,
    model: str,
    content: list[dict[str, Any]],
    timeout: float,
    max_tokens: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    body["_elapsed_sec"] = time.time() - t0
    return body


def assistant_text(response: dict[str, Any]) -> str:
    return str(response.get("choices", [{}])[0].get("message", {}).get("content", ""))


def global_prompt(meta: dict[str, Any], records: list[dict[str, Any]]) -> str:
    return (
        "你是机器人三维扫描数据的场景理解助手。下面是一段 cam1 视频按时间均匀抽取的全程关键帧拼图，"
        "每张图左上角标注了 rank/frame/time。请一次性理解扫描路线经过哪些区域。\n"
        "请不要逐帧复述；需要输出结构化 JSON，包含：\n"
        "{"
        "\"can_understand_route\": boolean, "
        "\"route_summary_zh\": string, "
        "\"segments\": [{\"start_rank\": int, \"end_rank\": int, \"area_name_zh\": string, \"visual_landmarks_zh\": string, \"confidence\": number}], "
        "\"stable_landmarks\": [string], "
        "\"uncertain_parts\": [string], "
        "\"usefulness_for_semantic_pointcloud\": string"
        "}。\n"
        "重点识别：室内/室外、停车场、通道、墙面、地面、草地、汽车、栏杆/玻璃围挡/围墙、楼梯或坡道、建筑立面等。\n"
        f"视频元数据：{json.dumps(meta, ensure_ascii=False)}\n"
        f"抽帧记录：{json.dumps([{k:v for k,v in r.items() if k!='path'} for r in records], ensure_ascii=False)}"
    )


def local_prompt(global_summary: str, row: dict[str, Any]) -> str:
    return (
        "你已经看过整段 cam1 扫描视频的关键帧路线总结。现在给你其中一张单帧图。"
        "请判断这张图大约是在扫描路线的哪个区域/阶段拍摄的，并说明依据。\n"
        "输出严格 JSON：{"
        "\"frame_index\": int, \"time_sec\": number, \"area_name_zh\": string, "
        "\"position_in_route_zh\": string, \"visible_landmarks_zh\": [string], "
        "\"likely_semantic_targets_zh\": [string], \"confidence\": number, \"reason_zh\": string"
        "}。\n"
        f"全局路线总结如下：{global_summary}\n"
        f"当前图像记录：{json.dumps({k:v for k,v in row.items() if k!='path'}, ensure_ascii=False)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--local-ranks", default="0,5,11,17,23")
    parser.add_argument("--long-edge", type=int, default=720)
    parser.add_argument("--contact-columns", type=int, default=4)
    parser.add_argument("--contact-thumb-width", type=int, default=360)
    parser.add_argument("--base-url", default=os.environ.get("MIMO_BASE_URL", "https://token-plan-sgp.xiaomimimo.com/v1"))
    parser.add_argument("--model", default=os.environ.get("MIMO_MODEL", "mimo-v2.5-pro"))
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-tokens", type=int, default=4096)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("MIMO_API_KEY")
    if not api_key:
        raise SystemExit("MIMO_API_KEY is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = read_video_meta(args.video)
    records = extract_frames(args.video, args.output_dir / "frames", args.sample_count, args.long_edge)
    contact = make_contact_sheet(
        records,
        args.output_dir / "cam1_route_contact_sheet.jpg",
        args.contact_columns,
        args.contact_thumb_width,
    )

    global_response = chat(
        args.base_url,
        api_key,
        args.model,
        [
            {"type": "text", "text": global_prompt(meta, records)},
            {"type": "image_url", "image_url": {"url": data_url(contact)}},
        ],
        args.timeout,
        args.max_tokens,
    )
    global_text = assistant_text(global_response)
    (args.output_dir / "mimo_global_route_response.json").write_text(
        json.dumps(global_response, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "mimo_global_route_response.txt").write_text(global_text + "\n", encoding="utf-8")

    local_rows = []
    wanted = [int(x) for x in args.local_ranks.split(",") if x.strip()]
    for rank in wanted:
        if rank < 0 or rank >= len(records):
            continue
        row = records[rank]
        response = chat(
            args.base_url,
            api_key,
            args.model,
            [
                {"type": "text", "text": local_prompt(global_text, row)},
                {"type": "image_url", "image_url": {"url": data_url(Path(row["path"]))}},
            ],
            args.timeout,
            args.max_tokens,
        )
        local_rows.append({"rank": rank, "frame": row, "response": response, "text": assistant_text(response)})
    (args.output_dir / "mimo_local_frame_responses.json").write_text(
        json.dumps(local_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = {
        "video_meta": meta,
        "sample_count": len(records),
        "contact_sheet": str(contact),
        "global_response_text": global_text,
        "local_response_count": len(local_rows),
        "local_responses": [
            {"rank": r["rank"], "frame_index": r["frame"]["frame_index"], "time_sec": r["frame"]["time_sec"], "text": r["text"]}
            for r in local_rows
        ],
    }
    (args.output_dir / "mimo_video_route_test_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "contact_sheet": str(contact), "local_count": len(local_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
