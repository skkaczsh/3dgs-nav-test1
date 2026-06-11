#!/usr/bin/env python3
"""Read-only readiness check for the next dense semantic frame increment."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/Users/skkac/Work/SCAN")


def run(cmd: list[str], timeout: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "returncode": -1,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or f"timeout after {timeout}s",
            "passed": False,
        }
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "passed": proc.returncode == 0,
    }


def remote_python(start: int, end: int, combo: str) -> str:
    return f"""
import json
from pathlib import Path

start={start}
end={end}
combo={combo!r}
frames=list(range(start,end+1))
cams=[0,1,2]

root=Path('/root/epfs')
calib=root/'new_route_data/calib'
ply_dir=root/'new_route_data/ply'
frames_dir=root/'new_route_stage1_skymask/frames'
color_dir=root/'new_route_stage1_skymask/output'
sky_dir=root/'new_route_data/sky_masks_color'
sam_dir=root/f'new_route_stage1_skymask/sam_masks_{{start:04d}}_{{end:04d}}_combined'
semantic_dir=root/f'manifold_3dgs_project/processed/semantic_eval_new_route_{{start:04d}}_{{end:04d}}'

def sky_exists(cam, frame):
    return any((sky_dir / f'cam{{cam}}_{{frame:0{{width}}d}}_sky.png').exists() for width in (7,6,5,4))

def sample_missing(items, limit=20):
    return items[:limit]

missing = {{
    'section_ply': [],
    'camera_frame': [],
    'color_ply': [],
    'sky_mask': [],
    'sam2_mask': [],
    'semantic_completion': [],
}}
counts = {{key: 0 for key in missing}}
for frame in frames:
    if (ply_dir / f'section_{{frame:04d}}.ply').exists():
        counts['section_ply'] += 1
    else:
        missing['section_ply'].append(frame)
    if (color_dir / f'frame_{{frame:04d}}.ply').exists():
        counts['color_ply'] += 1
    else:
        missing['color_ply'].append(frame)
    camera_ok = True
    for cam in cams:
        image_id=f'cam{{cam}}_{{frame:06d}}'
        if not (frames_dir / f'cam{{cam}}' / f'frame_{{frame:04d}}.png').exists():
            camera_ok = False
            missing['camera_frame'].append(image_id)
        if sky_exists(cam, frame):
            counts['sky_mask'] += 1
        else:
            missing['sky_mask'].append(image_id)
        if (sam_dir / f'{{image_id}}_sam_masks.json').exists():
            counts['sam2_mask'] += 1
        else:
            missing['sam2_mask'].append(image_id)
        if (semantic_dir / 'images' / image_id / combo / 'semantic.png').exists():
            counts['semantic_completion'] += 1
        else:
            missing['semantic_completion'].append(image_id)
    if camera_ok:
        counts['camera_frame'] += 1

total_frames=len(frames)
total_images=total_frames*len(cams)
ratios={{
    'section_ply': counts['section_ply']/max(total_frames,1),
    'camera_frame': counts['camera_frame']/max(total_frames,1),
    'color_ply': counts['color_ply']/max(total_frames,1),
    'sky_mask': counts['sky_mask']/max(total_images,1),
    'sam2_mask': counts['sam2_mask']/max(total_images,1),
    'semantic_completion': counts['semantic_completion']/max(total_images,1),
}}
calibration={{
    'img_pos': str(calib/'img_pos.txt'),
    'img_pos_exists': (calib/'img_pos.txt').exists(),
    'cam_in_ex': str(calib/'cam_in_ex.txt'),
    'cam_in_ex_exists': (calib/'cam_in_ex.txt').exists(),
}}
source_ready=calibration['img_pos_exists'] and calibration['cam_in_ex_exists'] and ratios['section_ply'] >= 0.99
camera_ready=ratios['camera_frame'] >= 0.99
sky_ready=ratios['sky_mask'] >= 0.95
generated_ready=ratios['color_ply'] >= 0.95 and ratios['sam2_mask'] >= 0.95 and ratios['semantic_completion'] >= 0.90
status='ready_for_generation' if source_ready else 'missing_sources'
if source_ready and camera_ready and sky_ready and generated_ready:
    status='ready_for_target_object_fusion'
elif source_ready and not camera_ready:
    status='needs_frame_generation'
elif source_ready and not sky_ready:
    status='needs_sky_generation'
elif source_ready:
    status='ready_for_color_sam_semantic_generation'

report={{
    'generated_at': None,
    'range': {{'start': start, 'end': end, 'frames': total_frames, 'images': total_images}},
    'paths': {{
        'calib_dir': str(calib),
        'ply_dir': str(ply_dir),
        'frames_dir': str(frames_dir),
        'color_dir': str(color_dir),
        'sky_dir': str(sky_dir),
        'sam_dir': str(sam_dir),
        'semantic_dir': str(semantic_dir),
    }},
    'calibration': calibration,
    'counts': counts,
    'ratios': ratios,
    'missing_samples': {{k: sample_missing(v) for k,v in missing.items()}},
    'source_ready': source_ready,
    'camera_ready': camera_ready,
    'sky_ready': sky_ready,
    'generated_ready': generated_ready,
    'status': status,
    'next_steps': [],
}}
if not source_ready:
    report['next_steps'].append('restore missing section PLY or calibration sources before compute work')
elif not camera_ready:
    report['next_steps'].append('extract camera frames for the increment')
elif not sky_ready:
    report['next_steps'].append('generate sky masks for the increment')
elif not generated_ready:
    report['next_steps'].append('run color projection, SAM2 masks, and Qwen semantic completion for the increment')
else:
    report['next_steps'].append('run target/object fusion for the increment after visual gate opens')
print(json.dumps(report, ensure_ascii=False, indent=2))
"""


def check_remote(args: argparse.Namespace) -> dict[str, Any]:
    cmd = [
        "ssh",
        "-F",
        "/dev/null",
        "-o",
        f"ConnectTimeout={min(args.timeout, 10)}",
        "-p",
        str(args.port),
        f"{args.user}@{args.host}",
        f"python3 - <<'PY'\n{remote_python(args.start, args.end, args.combo)}\nPY",
    ]
    result = run(cmd, args.timeout)
    if not result["passed"]:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "range": {"start": args.start, "end": args.end},
            "status": "ssh_failed",
            "passed": False,
            "ssh": {"returncode": result["returncode"], "stderr": result["stderr"][-2000:]},
        }
    report = json.loads(result["stdout"])
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["passed"] = report.get("status") in {
        "ready_for_generation",
        "needs_frame_generation",
        "needs_sky_generation",
        "ready_for_color_sam_semantic_generation",
        "ready_for_target_object_fusion",
    }
    report["ssh"] = {"host": args.host, "port": args.port, "user": args.user}
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Next Increment Readiness",
        "",
        f"- generated at: `{report.get('generated_at')}`",
        f"- status: `{report.get('status')}`",
        f"- passed: `{report.get('passed')}`",
        f"- range: `{report.get('range')}`",
        "",
        "## Ratios",
        "",
    ]
    for key, value in report.get("ratios", {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Missing Samples", ""])
    for key, value in report.get("missing_samples", {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Next Steps", ""])
    lines.extend(f"- {step}" for step in report.get("next_steps", []))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="10.0.8.114")
    parser.add_argument("--port", type=int, default=31909)
    parser.add_argument("--user", default="root")
    parser.add_argument("--start", type=int, default=1000)
    parser.add_argument("--end", type=int, default=1999)
    parser.add_argument("--combo", default="sam2_prompt_v3_sky_label_merge_completion")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--output", type=Path, default=ROOT / "route_status_20260610/next_increment_readiness_1000_1999.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "route_status_20260610/next_increment_readiness_1000_1999.md")
    args = parser.parse_args()

    report = check_remote(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(args.output), "markdown": str(args.markdown), "status": report.get("status"), "passed": report.get("passed")}, indent=2))
    if not report.get("passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
