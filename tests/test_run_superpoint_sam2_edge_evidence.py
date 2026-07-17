from pathlib import Path


def test_edge_runner_uses_only_edge_ledger_for_sam_inputs() -> None:
    script = (Path(__file__).parents[1] / "scripts" / "run_superpoint_sam2_edge_evidence.sh").read_text()

    assert 'make_sam2_input_links.py' in script
    assert '--views-jsonl "$EDGE_EVIDENCE"' in script
    assert '--output-mode compressed_rle' in script
    assert '--skip-visuals' in script
    assert 'for extension in jpg jpeg png' in script
    assert 'build_superpoint_sam2_comask_edges.py' in script
    assert 'make_superpoint_sam2_edge_review.py' in script
