from __future__ import annotations

from scripts.geometry_input_contract import (
    GEOMETRY_ONLY_LABEL_POLICY,
    GEOMETRY_ONLY_SEMANTIC_LABEL,
    GEOMETRY_ONLY_SEMANTIC_STATUS,
    geometry_only_semantic_fields,
    is_geometry_only_row,
)


def test_geometry_only_semantic_fields_do_not_promote_geometry_to_label() -> None:
    fields = geometry_only_semantic_fields("vertical")

    assert fields["semantic_label"] == GEOMETRY_ONLY_SEMANTIC_LABEL
    assert fields["semantic_label"] == "unknown"
    assert fields["geometry_label"] == "vertical"
    assert fields["semantic_status"] == GEOMETRY_ONLY_SEMANTIC_STATUS
    assert fields["label_policy"] == GEOMETRY_ONLY_LABEL_POLICY
    assert is_geometry_only_row(fields)


def test_geometry_only_row_requires_explicit_policy() -> None:
    assert not is_geometry_only_row({"semantic_label": "unknown", "geometry_label": "horizontal"})
