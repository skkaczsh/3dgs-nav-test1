from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "refresh_rtx5070_parking_candidate_review.sh"


def test_refresh_script_chains_health_pull_manifest_validate():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "check_rtx5070_parking_runtime.py" in text
    assert "pull_rtx5070_parking_candidate_surface_route.sh" in text
    assert "build_rtx5070_parking_candidate_manifest.py" in text
    assert "validate_rtx5070_parking_candidate_manifest.py" in text
    assert "rtx5070_runtime_check.json" in text
    assert "validation.json" in text


def test_refresh_script_has_disk_safe_controls_and_viewer_output():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "DRY_RUN" in text
    assert "SKIP_PULL" in text
    assert "PULL_QA_CROPS" in text
    assert "viewer_url()" in text
    assert "viewer=$(viewer_url)" in text
