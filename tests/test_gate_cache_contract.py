from pathlib import Path

from scripts.gate_cache_contract import resolve_relative_path, stale_gate_reasons


def test_resolve_relative_path_uses_repo_like_root() -> None:
    root = Path("/tmp/repo")

    assert resolve_relative_path("docs/gate.json", root) == root / "docs/gate.json"
    assert resolve_relative_path("/abs/gate.json", root) == Path("/abs/gate.json")
    assert resolve_relative_path("", root) is None


def test_stale_gate_reasons_reports_changed_cached_fields() -> None:
    cached = {"status": "pass", "candidate": "v2", "metrics": {"a": 1}, "reasons": []}
    recomputed = {"status": "fail", "candidate": "v2", "metrics": {"a": 2}, "reasons": ["x"]}

    assert stale_gate_reasons(cached, recomputed, prefix="patch_gate") == [
        "patch_gate_stale_status",
        "patch_gate_stale_metrics",
        "patch_gate_stale_reasons",
    ]


def test_stale_gate_reasons_ignores_missing_recomputed_gate() -> None:
    assert stale_gate_reasons({"status": "pass"}, None, prefix="gate") == []
