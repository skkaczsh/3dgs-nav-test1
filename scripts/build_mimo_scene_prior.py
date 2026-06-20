#!/usr/bin/env python3
"""Build a route-level scene prior from a camera video with Mimo.

For a long scan video, a single VLM image is too coarse. This script samples the
video at a fixed frame stride, creates multiple contact-sheet chunks, asks a
vision-capable Mimo model for each chunk, and merges the chunk summaries into a
stable JSON prior that downstream point/object stages can query by frame/time.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import mimetypes
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def read_video_meta(video: Path) -> dict[str, Any]:
    import cv2

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


def stride_indices(total_frames: int, stride: int) -> list[int]:
    if total_frames <= 0:
        return []
    indices = list(range(0, total_frames, max(stride, 1)))
    if indices[-1] != total_frames - 1:
        indices.append(total_frames - 1)
    return indices


def extract_stride_frames(video: Path, out_dir: Path, stride: int, long_edge: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import cv2
    from PIL import Image

    meta = read_video_meta(video)
    indices = stride_indices(int(meta["frames"]), stride)
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
        path = out_dir / f"sample_{rank:04d}_frame_{idx:06d}.jpg"
        Image.fromarray(rgb).save(path, quality=88)
        records.append({
            "rank": rank,
            "frame_index": idx,
            "time_sec": idx / float(meta["fps"]) if meta["fps"] else None,
            "path": str(path),
        })
    cap.release()
    return meta, records


def chunks(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def make_contact_sheet(records: list[dict[str, Any]], output: Path, columns: int, thumb_width: int) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    thumbs: list[Image.Image] = []
    label_h = 38
    for row in records:
        img = Image.open(row["path"]).convert("RGB")
        scale = thumb_width / img.width
        thumb = img.resize((thumb_width, round(img.height * scale)), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb.width, thumb.height + label_h), (16, 16, 16))
        canvas.paste(thumb, (0, label_h))
        text = f"r{row['rank']:03d} f{row['frame_index']} t{row['time_sec']:.1f}s"
        ImageDraw.Draw(canvas).text((6, 10), text, fill=(255, 255, 255), font=ImageFont.load_default())
        thumbs.append(canvas)
    rows = math.ceil(len(thumbs) / columns)
    cell_w = max(t.width for t in thumbs)
    cell_h = max(t.height for t in thumbs)
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), (8, 8, 8))
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % columns) * cell_w, (i // columns) * cell_h))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=90)
    return output


def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start:end + 1]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, str(exc)
    if not isinstance(parsed, dict):
        return None, "not_object"
    return parsed, ""


def chat(base_url: str, api_key: str, model: str, content: list[dict[str, Any]], timeout: float, max_tokens: int) -> dict[str, Any]:
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


def chunk_prompt(meta: dict[str, Any], records: list[dict[str, Any]], chunk_id: int, chunk_count: int) -> str:
    return (
        "你是移动扫描视频的场景先验标注器。输入是一组按时间顺序排列的关键帧拼图，"
        "每张图左上角有 rank/frame/time。请只根据这些图判断本 chunk 的场景段落。"
        "输出严格 JSON，不要 markdown，不要额外解释。Schema："
        "{"
        "\"chunk_id\": int, "
        "\"segments\": [{\"start_rank\": int, \"end_rank\": int, \"area_type\": string, "
        "\"area_name_zh\": string, \"visual_landmarks_zh\": [string], "
        "\"expected_labels\": [string], \"unlikely_labels\": [string], "
        "\"ground_subtypes\": [\"ordinary_ground\"|\"grass\"|\"stair\"|\"roof\"|\"indoor_floor\"], "
        "\"confidence\": number}], "
        "\"notes_zh\": string"
        "}。"
        "area_type 可用：indoor_lobby, outdoor_parking, grass_landscape, building_entrance, indoor_corridor, stairwell, rooftop, unknown。"
        "expected_labels/ unlikely_labels 用于后续语义点云先验，可包含 ground, wall, grass, car, railing, glass_fence, stair, roof, equipment, tree。"
        f"chunk={chunk_id}/{chunk_count}; video_meta={json.dumps(meta, ensure_ascii=False)}; "
        f"records={json.dumps([{k:v for k,v in r.items() if k!='path'} for r in records], ensure_ascii=False)}"
    )


def merge_prompt(meta: dict[str, Any], chunk_priors: list[dict[str, Any]]) -> str:
    return (
        "你是语义点云场景先验合并器。下面是同一个扫描视频按时间 chunk 产生的 JSON 摘要。"
        "请合并相邻同类区域，输出一个可按 frame/time 查询的全局 scene prior。"
        "输出严格 JSON，不要 markdown。Schema："
        "{"
        "\"schema\":\"mimo-scene-prior/v1\", "
        "\"route_summary_zh\": string, "
        "\"segments\": [{\"segment_id\": string, \"start_rank\": int, \"end_rank\": int, "
        "\"start_frame\": int, \"end_frame\": int, \"start_time_sec\": number, \"end_time_sec\": number, "
        "\"area_type\": string, \"area_name_zh\": string, \"visual_landmarks_zh\": [string], "
        "\"expected_labels\": [string], \"unlikely_labels\": [string], "
        "\"ground_subtypes\": [string], \"confidence\": number}], "
        "\"global_label_priors\": {\"label\": number}, "
        "\"usage_notes_zh\": string"
        "}。"
        "global_label_priors 是 0 到 1 的粗略权重，不是概率总和。"
        f"video_meta={json.dumps(meta, ensure_ascii=False)}\n"
        f"chunk_priors={json.dumps(chunk_priors, ensure_ascii=False)}"
    )


def add_frame_bounds(prior: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    by_rank = {int(r["rank"]): r for r in records}
    for i, segment in enumerate(prior.get("segments", []) if isinstance(prior.get("segments"), list) else []):
        segment.setdefault("segment_id", f"scene_{i:03d}")
        start = int(segment.get("start_rank", 0))
        end = int(segment.get("end_rank", start))
        start_row = by_rank.get(start) or by_rank[min(by_rank)]
        end_row = by_rank.get(end) or by_rank[max(by_rank)]
        segment["start_frame"] = int(segment.get("start_frame", start_row["frame_index"]))
        segment["end_frame"] = int(segment.get("end_frame", end_row["frame_index"]))
        segment["start_time_sec"] = float(segment.get("start_time_sec", start_row["time_sec"]))
        segment["end_time_sec"] = float(segment.get("end_time_sec", end_row["time_sec"]))
    return prior


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-stride", type=int, default=30)
    parser.add_argument("--long-edge", type=int, default=540)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--contact-columns", type=int, default=5)
    parser.add_argument("--contact-thumb-width", type=int, default=260)
    parser.add_argument("--base-url", default=os.environ.get("MIMO_BASE_URL", "https://token-plan-sgp.xiaomimimo.com/v1"))
    parser.add_argument("--vision-model", default=os.environ.get("MIMO_VISION_MODEL", "mimo-v2.5"))
    parser.add_argument("--merge-model", default=os.environ.get("MIMO_MERGE_MODEL", "mimo-v2.5-pro"))
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--vision-max-tokens", type=int, default=3072)
    parser.add_argument("--merge-max-tokens", type=int, default=4096)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("MIMO_API_KEY")
    if not api_key:
        raise SystemExit("MIMO_API_KEY is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta, records = extract_stride_frames(args.video, args.output_dir / "frames", args.sample_stride, args.long_edge)
    (args.output_dir / "scene_records.json").write_text(
        json.dumps({"video_meta": meta, "sample_stride": args.sample_stride, "records": records}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    chunk_rows = chunks(records, args.chunk_size)
    chunk_priors: list[dict[str, Any]] = []
    for chunk_id, rows in enumerate(chunk_rows):
        sheet = make_contact_sheet(
            rows,
            args.output_dir / "contact_sheets" / f"chunk_{chunk_id:03d}.jpg",
            args.contact_columns,
            args.contact_thumb_width,
        )
        response = chat(
            args.base_url,
            api_key,
            args.vision_model,
            [
                {"type": "text", "text": chunk_prompt(meta, rows, chunk_id, len(chunk_rows))},
                {"type": "image_url", "image_url": {"url": data_url(sheet)}},
            ],
            args.timeout,
            args.vision_max_tokens,
        )
        text = assistant_text(response)
        parsed, parse_error = parse_json_object(text)
        chunk_prior = parsed or {
            "chunk_id": chunk_id,
            "segments": [],
            "notes_zh": "parse_failed",
            "parse_error": parse_error,
            "raw_response": text,
        }
        chunk_prior["_contact_sheet"] = str(sheet)
        chunk_prior["_response_file"] = f"chunk_responses/chunk_{chunk_id:03d}.json"
        chunk_priors.append(chunk_prior)
        response_path = args.output_dir / "chunk_responses" / f"chunk_{chunk_id:03d}.json"
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_text(json.dumps(response, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    (args.output_dir / "chunk_scene_priors.json").write_text(
        json.dumps(chunk_priors, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    merge_response = chat(
        args.base_url,
        api_key,
        args.merge_model,
        [{"type": "text", "text": merge_prompt(meta, chunk_priors)}],
        args.timeout,
        args.merge_max_tokens,
    )
    merge_text = assistant_text(merge_response)
    scene_prior, parse_error = parse_json_object(merge_text)
    if scene_prior is None:
        scene_prior = {
            "schema": "mimo-scene-prior/v1",
            "route_summary_zh": "merge_parse_failed",
            "segments": [],
            "global_label_priors": {},
            "usage_notes_zh": parse_error,
            "raw_response": merge_text,
        }
    scene_prior = add_frame_bounds(scene_prior, records)
    scene_prior["video_meta"] = meta
    scene_prior["sample_stride"] = args.sample_stride
    scene_prior["record_count"] = len(records)
    scene_prior["chunk_count"] = len(chunk_rows)
    scene_prior["vision_model"] = args.vision_model
    scene_prior["merge_model"] = args.merge_model
    (args.output_dir / "mimo_scene_prior.json").write_text(
        json.dumps(scene_prior, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "mimo_scene_prior_merge_response.json").write_text(
        json.dumps(merge_response, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "record_count": len(records),
        "chunk_count": len(chunk_rows),
        "segment_count": len(scene_prior.get("segments", [])),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
