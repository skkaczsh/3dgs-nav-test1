from pathlib import Path

from scripts.sample_official_superpoints import read_jsonl


def test_preselected_rows_keep_candidate_metadata(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    path.write_text('{"object_id": 3, "seed_candidate_score": 2.5}\n', encoding="utf-8")
    assert read_jsonl(path) == [{"object_id": 3, "seed_candidate_score": 2.5}]
