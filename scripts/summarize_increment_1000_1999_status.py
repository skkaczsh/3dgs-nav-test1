#!/usr/bin/env python3
"""Summarize the current 1000-1999 increment state from local evidence.

This is intentionally conservative: remote production state is reported only
from the latest monitor artifact. If the latest monitor says the servers are
unreachable, this script does not reuse older conversational counts as fact.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/Users/skkac/Work/SCAN")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "bytes": 0, "mtime_utc": None}
    return {
        "path": str(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
    }


def summarize_remote(monitor: dict[str, Any]) -> dict[str, Any]:
    servers = monitor.get("servers") or []
    rows = []
    any_reachable = False
    for server in servers:
        reachable = bool(server.get("reachable"))
        any_reachable = any_reachable or reachable
        rows.append(
            {
                "name": server.get("name"),
                "reachable": reachable,
                "date_utc": server.get("date_utc"),
                "counts": server.get("counts") or {},
                "gpu": server.get("gpu") or [],
                "tmux": server.get("tmux") or [],
            }
        )
    return {
        "monitor_generated_at": monitor.get("generated_at"),
        "range": monitor.get("range"),
        "any_reachable": any_reachable,
        "servers": rows,
    }


def build_status(args: argparse.Namespace) -> dict[str, Any]:
    target_qa = read_json(args.target_qa)
    identity = read_json(args.identity_report)
    monitor = read_json(args.remote_monitor)

    local = {
        "target_root": str(args.target_root),
        "ply": file_info(args.ply),
        "objects_jsonl": file_info(args.objects_jsonl),
        "target_qa": file_info(args.target_qa),
        "identity_report": file_info(args.identity_report),
        "frames": target_qa.get("frames", {}),
        "objects": target_qa.get("objects", {}),
        "identity": {
            "objects": identity.get("objects"),
            "changed": identity.get("changed"),
            "changed_ratio": identity.get("changed_ratio"),
            "old_label_counts": identity.get("old_label_counts", {}),
            "new_label_counts": identity.get("new_label_counts", {}),
            "change_counts": identity.get("change_counts", {}),
        },
    }

    remote = summarize_remote(monitor)
    confirmed_state = "remote_unreachable_local_best_available"
    if remote["any_reachable"]:
        confirmed_state = "remote_reachable_check_counts_before_pull"

    return {
        "status_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "increment": {"start": 1000, "end": 1999, "images": 3000},
        "confirmed_state": confirmed_state,
        "local_best_artifact": local,
        "remote_latest_monitor": remote,
        "next_actions": [
            "When scan-train is reachable, check label_records and target refresh state.",
            "If label_records - target_refresh_state >= 60 or a newer target run exists, pull objects JSONL, relabeled PLY, and reports.",
            "Run local object/identity summaries after every pull before replacing the viewer artifact.",
            "Do not treat stale local active_route_progress ratios as live remote production counts while servers are unreachable.",
        ],
    }


def render_markdown(status: dict[str, Any]) -> str:
    local = status["local_best_artifact"]
    frames = local.get("frames", {})
    objects = local.get("objects", {})
    identity = local.get("identity", {})
    remote = status["remote_latest_monitor"]
    lines = [
        "# Increment 1000-1999 Status",
        "",
        f"- generated at: `{status['generated_at']}`",
        f"- confirmed state: `{status['confirmed_state']}`",
        f"- range: `{status['increment']}`",
        "",
        "## Local Best Artifact",
        "",
        f"- PLY: `{local['ply']['path']}`",
        f"- PLY exists/bytes/mtime: `{local['ply']['exists']}` / `{local['ply']['bytes']}` / `{local['ply']['mtime_utc']}`",
        f"- objects JSONL: `{local['objects_jsonl']['path']}`",
        f"- frames ok/missing: `{frames.get('ok_count')}` / `{frames.get('missing_or_failed_count')}`",
        f"- frame count: `{frames.get('count')}`",
        f"- objects: `{objects.get('count')}`",
        f"- merge ratio: `{objects.get('merge_ratio')}`",
        f"- ambiguous ratio: `{objects.get('ambiguous_ratio')}`",
        f"- semantic labels before identity relabel: `{objects.get('semantic_label_counts')}`",
        f"- identity changed: `{identity.get('changed')}` / `{identity.get('objects')}` (`{identity.get('changed_ratio')}`)",
        f"- semantic labels after identity relabel: `{identity.get('new_label_counts')}`",
        "",
        "## Remote Monitor",
        "",
        f"- monitor generated at: `{remote.get('monitor_generated_at')}`",
        f"- any reachable: `{remote.get('any_reachable')}`",
    ]
    for server in remote.get("servers", []):
        lines.append(f"- {server.get('name')}: reachable `{server.get('reachable')}`, counts `{server.get('counts')}`")
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in status["next_actions"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    root = ROOT
    target_root = root / "server_target_object_fusion_1000_1999_surface024_fine012"
    parser.add_argument("--target-root", type=Path, default=target_root)
    parser.add_argument("--ply", type=Path, default=target_root / "objects/object_points_identity_relabel_stride10.ply")
    parser.add_argument("--objects-jsonl", type=Path, default=target_root / "objects/objects_identity_relabel.jsonl")
    parser.add_argument("--target-qa", type=Path, default=target_root / "reports/target_object_qa.json")
    parser.add_argument("--identity-report", type=Path, default=target_root / "reports/identity_relabel_report.json")
    parser.add_argument("--remote-monitor", type=Path, default=root / "route_status_20260610/remote_production_monitor_20260611.json")
    parser.add_argument("--output-json", type=Path, default=root / "route_status_20260610/increment_1000_1999_status_20260612.json")
    parser.add_argument("--output-md", type=Path, default=root / "route_status_20260610/increment_1000_1999_status_20260612.md")
    args = parser.parse_args()

    status = build_status(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.write_text(render_markdown(status), encoding="utf-8")
    print(json.dumps({"json": str(args.output_json), "markdown": str(args.output_md), "state": status["confirmed_state"]}, indent=2))


if __name__ == "__main__":
    main()
