from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def test_server_vlm_extra_loop_filters_stable_sam_and_limits_batch():
    text = (SCRIPTS / "run_server_vlm_extra_loop.sh").read_text(encoding="utf-8")

    assert "--min-sam-age-seconds" in text
    assert "MAX_ITEMS_PER_CYCLE" in text
    assert "invalid_sam_extra_candidates" in text
    assert "run_server_semantic_completion_sharded.sh" in text
    assert "sam2_prompt_v3_sky_label_merge_completion/semantic.png" in text
    assert "SOURCE_SAM_MASKS_DIR" in text
    assert 'export SAM_MASKS_DIR="${LINKED_SAM_DIR}"' in text
    assert 'export EXISTING_SAM_DIR="${SOURCE_SAM_MASKS_DIR}"' in text


def test_remote_vlm_extra_loop_starts_tmux_and_syncs_scripts():
    text = (SCRIPTS / "start_remote_scan_vlm_extra_loop.sh").read_text(encoding="utf-8")

    assert "tmux has-session" in text
    assert "tmux new-session -d" in text
    assert "nohup bash -lc" in text
    assert "already_running_pid" in text
    assert "COPYFILE_DISABLE=1" in text
    assert "run_server_vlm_extra_loop.sh" in text
