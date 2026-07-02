"""Object-level semantic evidence fusion.

This module is intentionally geometry-first and ownership-preserving.  It
turns object evidence into a semantic posterior, but it never creates object
boundaries and never promotes geometry buckets into semantic labels by itself.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from scripts.geometry_input_contract import is_geometry_only_row
from scripts.semantic_label_contract import LABEL_TO_SEMANTIC


SKIP_LABELS = {"unknown", "ignore", "sky", "water", "other"}
SURFACE_LABELS = {"floor", "ground", "road", "grass", "stair", "indoor_floor", "roof", "ceiling"}
HORIZONTAL_LABELS = {"floor", "ground", "road", "grass", "stair", "indoor_floor", "roof"}
UPPER_LABELS = {"ceiling", "roof"}
VERTICAL_LABELS = {"wall", "building", "railing", "pipe", "equipment"}
THIN_LABELS = {"railing", "pipe", "equipment", "tree"}
ROUGH_LABELS = {
    "car",
    "tree",
    "grass",
    "railing",
    "pipe",
    "equipment",
    "wall",
    "building",
    "floor",
    "road",
    "stair",
    "roof",
    "indoor_floor",
}
KNOWN_LABELS = set(LABEL_TO_SEMANTIC)


@dataclass(frozen=True)
class FusionParams:
    sam_weight: float = 1.0
    teacher_weight: float = 1.25
    scene_weight: float = 0.35
    min_total_weight: float = 3.0
    min_winner_ratio: float = 0.58
    min_scene_supported_ratio: float = 0.52
    allow_scene_only: bool = False


def params_from_args(args: argparse.Namespace) -> FusionParams:
    return FusionParams(
        sam_weight=float(args.sam_weight),
        teacher_weight=float(args.teacher_weight),
        scene_weight=float(args.scene_weight),
        min_total_weight=float(args.min_total_weight),
        min_winner_ratio=float(args.min_winner_ratio),
        min_scene_supported_ratio=float(args.min_scene_supported_ratio),
        allow_scene_only=bool(args.allow_scene_only),
    )


def normalize_label(label: str) -> str:
    value = str(label or "unknown").strip()
    if value == "ground":
        return "floor"
    if value == "ambiguous":
        return "unknown"
    return value if value in KNOWN_LABELS else "unknown"


def geometry_type(row: dict[str, Any]) -> str:
    return str(row.get("geometry_type") or row.get("object_type_geometry") or "unknown")


def current_semantic_label(row: dict[str, Any]) -> str:
    if is_geometry_only_row(row):
        return "unknown"
    return normalize_label(str(row.get("semantic_label") or row.get("dominant_label") or "unknown"))


def label_allowed(label: str, geom: str) -> bool:
    label = normalize_label(label)
    if label in SKIP_LABELS:
        return False
    if geom == "horizontal":
        return label in HORIZONTAL_LABELS
    if geom == "upper_surface":
        return label in UPPER_LABELS
    if geom == "vertical":
        return label in VERTICAL_LABELS
    if geom == "thin_linear":
        return label in THIN_LABELS
    if geom in {"rough_mixed", "mixed", "unknown"}:
        return label in ROUGH_LABELS or label in SURFACE_LABELS or label in VERTICAL_LABELS
    return True


def weighted_votes(values: Any, weight: float) -> Counter[str]:
    votes: Counter[str] = Counter()
    if not isinstance(values, dict):
        return votes
    for raw_label, raw_count in values.items():
        label = normalize_label(str(raw_label))
        if label in SKIP_LABELS:
            continue
        try:
            count = float(raw_count)
        except (TypeError, ValueError):
            continue
        if count <= 0:
            continue
        votes[label] += count * weight
    return votes


def scene_expected_votes(row: dict[str, Any], params: FusionParams) -> Counter[str]:
    scene = row.get("scene_prior") if isinstance(row.get("scene_prior"), dict) else {}
    return weighted_votes(scene.get("scene_expected_label_weights"), params.scene_weight)


def collect_evidence(row: dict[str, Any], params: FusionParams) -> dict[str, Counter[str]]:
    return {
        "sam": weighted_votes(row.get("semantic_votes"), params.sam_weight),
        "teacher": weighted_votes(row.get("teacher_allowed_votes") or row.get("teacher_semantic_votes"), params.teacher_weight),
        "scene": scene_expected_votes(row, params),
    }


def combine_allowed(
    evidence: dict[str, Counter[str]],
    geom: str,
) -> tuple[Counter[str], Counter[str]]:
    allowed: Counter[str] = Counter()
    vetoed: Counter[str] = Counter()
    for source_votes in evidence.values():
        for label, value in source_votes.items():
            if label_allowed(label, geom):
                allowed[label] += float(value)
            else:
                vetoed[label] += float(value)
    return allowed, vetoed


def has_non_scene_support(label: str, evidence: dict[str, Counter[str]]) -> bool:
    return evidence["sam"].get(label, 0.0) > 0 or evidence["teacher"].get(label, 0.0) > 0


def choose_label(row: dict[str, Any], params: FusionParams | None = None) -> dict[str, Any]:
    params = params or FusionParams()
    geom = geometry_type(row)
    original = current_semantic_label(row)
    evidence = collect_evidence(row, params)
    allowed, vetoed = combine_allowed(evidence, geom)
    total_allowed = float(sum(allowed.values()))
    conflict_flags: list[str] = []

    if vetoed:
        conflict_flags.append("geometry_vetoed_some_evidence")
    if total_allowed < params.min_total_weight:
        return {
            "semantic_label": original,
            "semantic_status": "kept_original_insufficient_evidence",
            "semantic_confidence": 0.0,
            "semantic_label_original": original,
            "semantic_evidence_scores": dict(allowed),
            "semantic_vetoed_scores": dict(vetoed),
            "conflict_flags": conflict_flags,
        }

    label, score = allowed.most_common(1)[0]
    ratio = float(score / max(total_allowed, 1e-9))
    if not params.allow_scene_only and not has_non_scene_support(label, evidence):
        return {
            "semantic_label": original,
            "semantic_status": "kept_original_scene_only_evidence",
            "semantic_confidence": ratio,
            "semantic_label_original": original,
            "semantic_evidence_scores": dict(allowed),
            "semantic_vetoed_scores": dict(vetoed),
            "conflict_flags": conflict_flags + ["scene_only_label_not_promoted"],
        }
    threshold = params.min_scene_supported_ratio if evidence["scene"].get(label, 0.0) > 0 else params.min_winner_ratio
    if ratio < threshold:
        return {
            "semantic_label": original,
            "semantic_status": "kept_original_ambiguous_evidence",
            "semantic_confidence": ratio,
            "semantic_label_original": original,
            "semantic_evidence_scores": dict(allowed),
            "semantic_vetoed_scores": dict(vetoed),
            "conflict_flags": conflict_flags + ["low_winner_ratio"],
        }
    return {
        "semantic_label": label,
        "semantic_status": "evidence_fusion_applied",
        "semantic_confidence": ratio,
        "semantic_label_original": original,
        "semantic_evidence_scores": dict(allowed),
        "semantic_vetoed_scores": dict(vetoed),
        "conflict_flags": conflict_flags,
    }


def apply_decision(row: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["semantic_label_original"] = decision["semantic_label_original"]
    out["semantic_label"] = decision["semantic_label"]
    out["semantic_id"] = int(LABEL_TO_SEMANTIC.get(decision["semantic_label"], 0))
    out["semantic_fusion_status"] = decision["semantic_status"]
    out["semantic_fusion_confidence"] = decision["semantic_confidence"]
    out["semantic_evidence_scores"] = decision["semantic_evidence_scores"]
    out["semantic_vetoed_scores"] = decision["semantic_vetoed_scores"]
    existing_flags = list(out.get("conflict_flags") or [])
    out["conflict_flags"] = sorted(set(existing_flags + list(decision.get("conflict_flags") or [])))
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    for row in rows:
        label_counts[str(row.get("semantic_label") or "unknown")] += 1
        status_counts[str(row.get("semantic_fusion_status") or "missing")] += 1
        for flag in row.get("conflict_flags") or []:
            flag_counts[str(flag)] += 1
    return {
        "object_count": len(rows),
        "label_counts": dict(label_counts),
        "status_counts": dict(status_counts),
        "conflict_flag_counts": dict(flag_counts),
    }
