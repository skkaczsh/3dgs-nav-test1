from pathlib import Path


def test_retry_runner_uses_only_missing_baseline_objects() -> None:
    script = (Path(__file__).parents[1] / "scripts" / "run_official_superpoint_evidence_retry.sh").read_text(encoding="utf-8")
    assert 'seen = {json.loads(line)["object_id"]' in script
    assert 'missing = [row for row in objects if row["object_id"] not in seen]' in script
    assert "--view-selection projected" in script
    assert "--global-depth-map-dir" in script
    assert "--sky-mask-dir" in script
