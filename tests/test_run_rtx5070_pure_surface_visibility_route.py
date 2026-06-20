from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_rtx5070_pure_surface_visibility_route.sh"


def test_pure_surface_visibility_route_is_remote_dry_run_by_default():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'REMOTE_HOST="${REMOTE_HOST:-scan-rtx5070}"' in text
    assert 'RUN="${RUN:-0}"' in text
    assert 'OVERWRITE="${OVERWRITE:-0}"' in text
    assert 'ssh "${REMOTE_HOST}"' in text


def test_pure_surface_visibility_route_reuses_safe_target_builder_then_attachment_fusion():
    text = SCRIPT.read_text(encoding="utf-8")

    safe_pos = text.index("scripts/run_parking_safe_semantic_prior_route.sh")
    structural_pos = text.index("python scripts/build_structural_region_field.py")
    attachment_pos = text.index("python scripts/classify_surface_attachment.py")
    fusion_pos = text.index("python scripts/fuse_targets_to_objects.py")
    viewer_pos = text.index("python scripts/export_frame_target_objects_for_viewer.py")

    assert safe_pos < structural_pos < attachment_pos < fusion_pos < viewer_pos
    assert "export BUILD_TARGETS=1" in text
    assert "export BUILD_OBJECTS=0" in text
    assert "--strict-surface-labels" in text
    assert "--fallback-zone-scan" in text


def test_pure_surface_visibility_route_keeps_structural_prior_nonsemantic_boundary():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "drivability structural field" in text
    assert "REMOTE_DRIVABILITY_PCD" in text
    assert "--drivability-pcd '${REMOTE_DRIVABILITY_PCD}'" in text
    assert "--structural-field '${STRUCTURAL_DIR}/structural_region_field.npz'" in text
    assert "surface_attachment_report.json" in text


def test_pure_surface_visibility_route_can_pull_review_artifacts_without_committing_them():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'PULL_RESULTS="${PULL_RESULTS:-0}"' in text
    assert 'server_parking_priority_s10/${OUT_SUFFIX}' in text
    assert 'rsync -az "${REMOTE_HOST}:${VIEWER_DIR}/"' in text

