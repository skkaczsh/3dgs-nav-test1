#!/usr/bin/env python3
"""Conservatively relabel fused Objects using identity descriptions.

The VLM taxonomy stays coarse, but descriptions often carry stronger instance
evidence than the selected label. This post-process keeps the original object
records traceable while producing a corrected variant for QA/viewing.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


LABEL_IDS = {
    "unknown": 0,
    "other": 1,
    "wall": 2,
    "floor": 3,
    "ceiling": 4,
    "grass": 5,
    "tree": 6,
    "person": 7,
    "car": 8,
    "railing": 9,
    "building": 10,
    "sky": 11,
    "road": 12,
    "water": 13,
    "furniture": 14,
    "pipe": 15,
    "equipment": 16,
    "ignore": 255,
}

LABEL_COLORS = {
    "unknown": (128, 128, 128),
    "other": (180, 180, 180),
    "wall": (170, 170, 170),
    "floor": (190, 170, 135),
    "building": (150, 150, 150),
    "railing": (255, 210, 40),
    "pipe": (90, 160, 255),
    "equipment": (30, 210, 190),
    "ambiguous": (230, 40, 210),
}

EQUIPMENT_TERMS = {
    "air conditioner",
    "air-conditioning",
    "antenna",
    "cabinet",
    "compressor",
    "control box",
    "electrical",
    "equipment",
    "fixture",
    "hvac",
    "machinery",
    "outdoor unit",
    "sensor",
}
RAILING_TERMS = {
    "fence",
    "guardrail",
    "handrail",
    "mesh fence",
    "railing",
    "wire mesh",
}
PIPE_TERMS = {
    "cable tray",
    "conduit",
    "duct",
    "pipe",
}
FLOOR_TERMS = {
    "concrete roof",
    "concrete rooftop",
    "floor",
    "paving",
    "roof surface",
    "rooftop surface",
    "walkable",
}
WALL_TERMS = {
    "parapet",
    "vertical concrete wall",
    "vertical wall",
    "wall panel",
    "wall section",
    "wall surface",
}
BUILDING_TERMS = {
    "building facade",
    "facade",
    "high-rise",
    "window",
}


def norm(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def object_number(object_id: str) -> int:
    match = re.search(r"(\d+)$", str(object_id or ""))
    return int(match.group(1)) if match else 0


def primary_identity_text(obj: dict) -> str:
    return norm(
        " ".join(
            [
                str(obj.get("object_identity", "")),
                str(obj.get("dominant_freeform_label", "")),
                str(obj.get("description", "")),
                str(obj.get("identity_hint", "")),
            ]
        )
    )


def secondary_identity_text(obj: dict) -> str:
    chunks: list[str] = [
    ]
    for value in (obj.get("attributes") or {}).values():
        chunks.append(str(value))
    for value in (obj.get("dominant_attributes") or {}).values():
        if isinstance(value, dict):
            chunks.append(str(value.get("value", "")))
        else:
            chunks.append(str(value))
    secondary = norm(" ".join(chunks))
    if secondary:
        return secondary
    return norm(
        " ".join(
            [str(desc) for desc in (obj.get("description_votes") or {}).keys()]
            + [str(desc) for desc in (obj.get("freeform_label_votes") or {}).keys()]
        )
    )


def infer_from_text(text: str) -> tuple[str | None, str]:
    has_equipment = contains_any(text, EQUIPMENT_TERMS)
    has_railing = contains_any(text, RAILING_TERMS)
    has_pipe = contains_any(text, PIPE_TERMS)
    has_floor = contains_any(text, FLOOR_TERMS)
    has_wall = contains_any(text, WALL_TERMS)
    has_building = contains_any(text, BUILDING_TERMS)

    if has_equipment:
        return "equipment", "identity_equipment"
    if has_railing:
        return "railing", "identity_railing"
    if has_pipe:
        return "pipe", "identity_pipe"
    if has_floor:
        return "floor", "identity_floor"
    if has_wall:
        return "wall", "identity_wall"
    if has_building:
        return "building", "identity_building"
    return None, "no_matching_identity_rule"


def infer_identity_label(obj: dict) -> tuple[str | None, str]:
    current = str(obj.get("semantic_label") or obj.get("dominant_label") or "unknown")
    primary = primary_identity_text(obj)
    secondary = secondary_identity_text(obj)
    if not primary and not secondary:
        return None, "no_identity_text"

    inferred, reason = infer_from_text(primary)
    if inferred is None:
        inferred, reason = infer_from_text(secondary)
    if inferred in {"floor", "wall", "building"}:
        if inferred == "floor" and current not in {"ambiguous", "wall", "building", "unknown", "other"}:
            return None, "surface_identity_blocked"
        if inferred == "wall" and current not in {"ambiguous", "floor", "building", "unknown", "other"}:
            return None, "surface_identity_blocked"
        if inferred == "building" and current not in {"ambiguous", "wall", "unknown", "other"}:
            return None, "surface_identity_blocked"
    if inferred:
        return inferred, reason
    return None, "no_matching_identity_rule"


def should_apply(current: str, inferred: str, obj: dict) -> bool:
    if inferred == current:
        return False
    if current == "ambiguous":
        return True
    if {current, inferred} <= {"building", "railing", "pipe", "equipment"}:
        return True
    if {current, inferred} <= {"floor", "wall", "building"}:
        return True
    if current in {"unknown", "other"}:
        return True
    votes = obj.get("label_votes") or {}
    total = sum(float(v) for v in votes.values())
    winner = max((float(v) for v in votes.values()), default=0.0)
    return total > 0 and winner / total < 0.8


def relabel_object(obj: dict) -> tuple[dict, dict]:
    out = dict(obj)
    current = str(out.get("semantic_label") or out.get("dominant_label") or "unknown")
    inferred, reason = infer_identity_label(out)
    decision = {
        "object_id": out.get("object_id"),
        "old_label": current,
        "new_label": current,
        "reason": reason,
        "changed": False,
        "description": out.get("description", ""),
        "object_identity": out.get("object_identity", ""),
        "label_votes": out.get("label_votes", {}),
    }
    if inferred and should_apply(current, inferred, out):
        out["original_semantic_label"] = current
        out["semantic_label"] = inferred
        out["identity_relabel_reason"] = reason
        out["identity_relabel_applied"] = True
        decision["new_label"] = inferred
        decision["changed"] = True
    else:
        out["identity_relabel_applied"] = False
    return out, decision


def read_objects(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_objects(path: Path, objects: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in objects:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_ply_header(path: Path) -> tuple[list[str], list[str], int]:
    header: list[str] = []
    props: list[str] = []
    in_vertex = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            header.append(line)
            stripped = line.strip()
            if stripped.startswith("format ") and "ascii" not in stripped:
                raise ValueError(f"Only ASCII PLY is supported: {path}")
            if stripped.startswith("element vertex"):
                in_vertex = True
            elif stripped.startswith("element "):
                in_vertex = False
            elif in_vertex and stripped.startswith("property "):
                props.append(stripped.split()[-1])
            elif stripped == "end_header":
                break
    return header, props, len(header)


def remap_ply(input_ply: Path, output_ply: Path, objects: list[dict]) -> dict:
    header, props, header_lines = read_ply_header(input_ply)
    prop_idx = {name: i for i, name in enumerate(props)}
    required = {"object", "semantic", "red", "green", "blue"}
    if not required.issubset(prop_idx):
        raise ValueError(f"PLY missing required properties: {sorted(required - set(prop_idx))}")
    object_to_label = {
        object_number(str(obj.get("object_id"))): str(obj.get("semantic_label", "unknown"))
        for obj in objects
    }
    changed_vertices = 0
    total_vertices = 0
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    with input_ply.open("r", encoding="utf-8", errors="replace") as src, output_ply.open("w", encoding="utf-8") as dst:
        for line in header:
            dst.write(line)
        for _ in range(header_lines):
            next(src)
        for line in src:
            parts = line.split()
            if len(parts) < len(props):
                continue
            total_vertices += 1
            obj_id = int(float(parts[prop_idx["object"]]))
            label = object_to_label.get(obj_id)
            if label:
                semantic = str(LABEL_IDS.get(label, 0))
                if parts[prop_idx["semantic"]] != semantic:
                    changed_vertices += 1
                color = LABEL_COLORS.get(label, LABEL_COLORS["unknown"])
                parts[prop_idx["semantic"]] = semantic
                parts[prop_idx["red"]] = str(color[0])
                parts[prop_idx["green"]] = str(color[1])
                parts[prop_idx["blue"]] = str(color[2])
            dst.write(" ".join(parts) + "\n")
    return {"total_vertices": total_vertices, "changed_vertices": changed_vertices, "output_ply": str(output_ply)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--input-ply", type=Path, default=None)
    parser.add_argument("--output-ply", type=Path, default=None)
    args = parser.parse_args()

    original = read_objects(args.objects_jsonl)
    relabeled = []
    decisions = []
    for obj in original:
        row, decision = relabel_object(obj)
        relabeled.append(row)
        decisions.append(decision)
    write_objects(args.output_jsonl, relabeled)

    changed = [d for d in decisions if d["changed"]]
    report = {
        "objects": len(relabeled),
        "changed": len(changed),
        "changed_ratio": len(changed) / max(len(relabeled), 1),
        "old_label_counts": dict(Counter(str(o.get("semantic_label", "unknown")) for o in original)),
        "new_label_counts": dict(Counter(str(o.get("semantic_label", "unknown")) for o in relabeled)),
        "change_counts": dict(Counter(f"{d['old_label']}->{d['new_label']}" for d in changed)),
        "reason_counts": dict(Counter(d["reason"] for d in changed)),
        "changed_samples": changed[:50],
    }
    if args.input_ply and args.output_ply:
        report["ply"] = remap_ply(args.input_ply, args.output_ply, relabeled)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
