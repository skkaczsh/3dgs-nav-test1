import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "scan_sensitive_tokens_for_test",
        SCRIPTS / "scan_sensitive_tokens.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_scan_detects_huggingface_token_without_exposing_secret(tmp_path: Path):
    module = load_module()
    token_file = tmp_path / "bad.txt"
    token_file.write_text("HF token: hf_" + "A" * 32 + "\n", encoding="utf-8")

    report = module.scan(tmp_path)

    assert report["finding_count"] == 1
    assert report["findings"][0] == {"path": "bad.txt", "line": 1, "kind": "huggingface_token"}
    assert "A" * 32 not in str(report)


def test_scan_ignores_git_directory(tmp_path: Path):
    module = load_module()
    hidden = tmp_path / ".git" / "config"
    hidden.parent.mkdir()
    hidden.write_text("hf_" + "B" * 32, encoding="utf-8")

    report = module.scan(tmp_path)

    assert report["finding_count"] == 0


def test_scan_skips_large_files_by_default(tmp_path: Path):
    module = load_module()
    large = tmp_path / "large.txt"
    large.write_text("x" * 200 + " hf_" + "C" * 32, encoding="utf-8")

    report = module.scan(tmp_path, max_file_bytes=100)

    assert report["finding_count"] == 0
    assert report["skipped_large_files"] == 1


def test_current_repo_has_no_high_risk_tokens():
    module = load_module()
    report = module.scan(ROOT)

    assert report["finding_count"] == 0
