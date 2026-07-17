from pathlib import Path


def test_depth_aware_view_plan_runner_keeps_one_calibrated_contract() -> None:
    script = (Path(__file__).parents[1] / "scripts" / "run_superpoint_depth_aware_view_plan.sh").read_text(encoding="utf-8")
    assert "--global-view-plan-depth-aware" in script
    assert "--global-view-plan-prefilter" in script
    assert "OPENBLAS_NUM_THREADS=4" in script
    assert '"schema": "global-evidence-view-plan/v1"' in script
    assert "duplicate object ids across shards" in script
