from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "start_rtx5070_parking_candidate_surface_route.sh"


def test_launcher_runs_healthcheck_before_tmux():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "check_rtx5070_parking_runtime.py" in text
    assert "--output" in text
    assert "CHECK_ONLY=1" in text
    assert "tmux has-session" in text
    assert "tmux new-session -d -s" in text


def test_launcher_has_safe_session_and_dry_run_controls():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "DRY_RUN" in text
    assert "RESTART" in text
    assert "session_exists" in text
    assert "set RESTART=1" in text
    assert "remote_log" in text
    assert "tail -f" in text
