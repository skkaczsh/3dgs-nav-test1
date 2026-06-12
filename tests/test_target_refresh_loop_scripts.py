from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def test_target_refresh_loop_uses_label_records_delta_and_identity_preview():
    text = (SCRIPTS / "run_server_target_object_refresh_loop.sh").read_text(encoding="utf-8")

    assert "MIN_COMPLETION_DELTA" in text
    assert "RUN_ON_FIRST" in text
    assert "label_records.json" in text
    assert "run_server_target_object_fusion.sh" in text
    assert "SURFACE_VOXEL_SIZE" in text
    assert "FINE_VOXEL_SIZE" in text
    assert "relabel_objects_from_identity.py" in text
    assert "stride_ascii_ply.py" in text
    assert "LOCK_DIR" in text
    assert "skip: lock held" in text
    assert "state remains" in text


def test_remote_target_refresh_loop_starts_tmux_and_syncs_scripts():
    text = (SCRIPTS / "start_remote_scan_train_target_refresh_loop.sh").read_text(encoding="utf-8")

    assert "tmux has-session" in text
    assert "tmux new-session -d" in text
    assert "nohup bash -lc" in text
    assert "COPYFILE_DISABLE=1" in text
    assert "run_server_target_object_refresh_loop.sh" in text
    assert "MIN_COMPLETION_DELTA" in text
