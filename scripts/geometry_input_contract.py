"""Contract for geometry-only patch/object artifacts.

Geometry stages may emit viewer-compatible JSONL rows, but those rows must not
pretend that structural buckets are semantic labels.  Downstream semantic
fusion should treat these rows as ownership geometry plus evidence containers.
"""

from __future__ import annotations


GEOMETRY_ONLY_SEMANTIC_LABEL = "unknown"
GEOMETRY_ONLY_SEMANTIC_STATUS = "geometry_only_unlabeled"
GEOMETRY_ONLY_LABEL_POLICY = "geometry_is_not_semantic"


def geometry_only_semantic_fields(geometry_type: str) -> dict[str, str]:
    return {
        "semantic_label": GEOMETRY_ONLY_SEMANTIC_LABEL,
        "semantic_status": GEOMETRY_ONLY_SEMANTIC_STATUS,
        "label_policy": GEOMETRY_ONLY_LABEL_POLICY,
        "geometry_label": str(geometry_type or "unknown"),
    }


def is_geometry_only_row(row: dict) -> bool:
    return (
        row.get("semantic_status") == GEOMETRY_ONLY_SEMANTIC_STATUS
        and row.get("label_policy") == GEOMETRY_ONLY_LABEL_POLICY
    )
