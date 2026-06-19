from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_rtx5070_sync_anchor_solver.sh"


def test_sync_anchor_solver_runner_gates_production_with_readiness():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "scripts/check_sync_frame_map_readiness.py" in text
    assert "--time-mode $(quote \"${SOLVER_TIME_MODE}\")" in text
    assert "--absolute-intercept-source $(quote \"${SOLVER_ABSOLUTE_INTERCEPT_SOURCE}\")" in text
    assert 'SOLVER_ABSOLUTE_INTERCEPT_SOURCE="${SOLVER_ABSOLUTE_INTERCEPT_SOURCE:-anchors}"' in text
    assert "scripts/expand_sync_frame_map.py" in text
    assert "--frame-map-jsonl $(quote \"${REMOTE_OUTPUT}/expanded_frame_map.jsonl\")" in text
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


def test_sync_anchor_solver_runner_defaults_to_absprior_review():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'REVIEW_NAME="${REVIEW_NAME:-sync_anchor_review_priority_sky_penalty_timestamp_absprior_dot3_20260619}"' in text
    assert 'RUN_NAME="${RUN_NAME:-sync_anchor_constrained_timestamp_absprior_dot3_20260619}"' in text
    assert 'DOT_PX="${DOT_PX:-3}"' in text
    assert 'MAP_START="${MAP_START:-0}"' in text
    assert 'MAP_END="${MAP_END:-6180}"' in text
    assert 'MAP_STRIDE="${MAP_STRIDE:-10}"' in text
    assert 'VIDEO_FRAME_COUNT="${VIDEO_FRAME_COUNT:-6181}"' in text
    assert 'REMOTE_CANDIDATES="${REMOTE_CANDIDATES:-${REMOTE_WORK}/sync_calibration_sky_penalty_fullprobe_20260619/sync_candidates.jsonl}"' in text
