from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_rtx5070_sync_anchor_solver.sh"


def test_sync_anchor_solver_runner_gates_production_with_readiness():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "scripts/check_sync_frame_map_readiness.py" in text
    assert "--frame-map-jsonl $(quote \"${REMOTE_OUTPUT}/solver/sync_smooth_paths.jsonl\")" in text
    assert "--solver-report $(quote \"${REMOTE_OUTPUT}/solver/sync_smooth_path_report.json\")" in text
    assert "sync_frame_map_readiness.json" in text
    assert "sync_frame_map_readiness.exit_code" in text
    assert "readiness_failed=1" in text
    assert "exit 3" in text


def test_sync_anchor_solver_runner_keeps_review_pack_on_readiness_failure():
    text = SCRIPT.read_text(encoding="utf-8")
    readiness_pos = text.index("scripts/check_sync_frame_map_readiness.py")
    review_pos = text.index("scripts/build_sync_anchor_review_pack.py")
    pull_pos = text.index("\"${SERVER}:${REMOTE_OUTPUT}/sync_frame_map_readiness.json\"")
    failure_pos = text.index("if [[ \"${readiness_code}\" != \"0\" ]]")

    assert readiness_pos < review_pos
    assert pull_pos < failure_pos
