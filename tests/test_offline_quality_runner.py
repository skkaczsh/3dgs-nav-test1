from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def test_offline_quality_runner_avoids_server_connectivity_checks():
    script = (SCRIPTS / "run_offline_quality_checks.sh").read_text(encoding="utf-8")

    assert "diagnose_server_connectivity.py" not in script
    assert "ssh " not in script
    assert "scp " not in script


def test_offline_quality_runner_covers_core_checks():
    script = (SCRIPTS / "run_offline_quality_checks.sh").read_text(encoding="utf-8")

    assert "py_compile scripts/*.py" in script
    assert "scan_sensitive_tokens.py" in script
    assert "audit_runner_dependencies.py" in script
    assert "verify_review_delivery_manifest.py" in script
    assert "OFFLINE_QA_REPORT" in script
    assert "offline_quality_latest.json" in script
    assert "pytest -q" in script
    assert "tests/test_offline_quality_runner.py" in script
    assert "tests/test_route_status_summary.py" in script
    assert "tests/test_scan_sensitive_tokens.py" in script
