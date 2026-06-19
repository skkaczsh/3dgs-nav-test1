from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_rtx5070_sync_gated_parking_dataset.sh"


def test_gated_parking_dataset_requires_sync_readiness_and_frame_map():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'RUN="${RUN:-0}"' in text
    assert 'SYNC_RUN_NAME="${SYNC_RUN_NAME:-sync_anchor_constrained_timestamp_absprior_dot3_20260619}"' in text
    assert 'FRAME_MAP="${FRAME_MAP:-${REMOTE_SYNC_DIR}/expanded_frame_map.jsonl}"' in text
    assert 'READINESS_EXIT="${READINESS_EXIT:-${REMOTE_SYNC_DIR}/sync_frame_map_readiness.exit_code}"' in text
    assert 'readiness_not_passing=' in text
    assert 'missing_frame_map=' in text


def test_gated_parking_dataset_uses_frame_map_for_image_and_color_steps():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "--sync-mode frame-map" in text
    assert "--frame-map-jsonl $(quote \"${FRAME_MAP}\")" in text
    assert "--require-frame-map" in text
    assert "scripts/extract_undistorted_frames_jpeg.py" in text
    assert "scripts/colorize_lx_stream.py" in text


def test_gated_parking_dataset_runs_priority_and_safe_route_after_frames():
    text = SCRIPT.read_text(encoding="utf-8")

    extract_pos = text.index("scripts/extract_undistorted_frames_jpeg.py")
    priority_pos = text.index("scripts/segment_priority_classes.py")
    route_pos = text.index("scripts/run_parking_safe_semantic_prior_route.sh")

    assert extract_pos < priority_pos < route_pos
    assert "--batch-size $(quote \"${PRIORITY_BATCH_SIZE}\")" in text
    assert 'PRIORITY_BATCH_SIZE="${PRIORITY_BATCH_SIZE:-8}"' in text
    assert 'DO_SAFE_ROUTE="${DO_SAFE_ROUTE:-0}"' in text
