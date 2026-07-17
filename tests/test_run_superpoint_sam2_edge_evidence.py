from pathlib import Path


def test_runner_supports_exact_single_ledger_without_duplicate_evidence() -> None:
    script = (Path(__file__).parents[1] / "scripts" / "run_superpoint_sam2_edge_evidence.sh").read_text()

    assert 'make_sam2_input_links.py' in script
    assert '--evidence-jsonl' in script
    assert '--views-jsonl "${EVIDENCE_JSONL:-$EDGE_EVIDENCE}"' in script
    assert 'cp "$EVIDENCE_JSONL" "$COMBINED_EVIDENCE"' in script
    assert 'mutually exclusive with --edge-evidence/--direct-evidence' in script
    assert '--output-mode compressed_rle' in script
    assert '--skip-visuals' in script
    assert 'GPU_IDS="${GPU_IDS:-$GPU_ID}"' in script
    assert 'mapfile -t INPUT_IMAGES' in script
    assert 'index % ${#GPU_LIST[@]}' in script
    assert 'run_sam2_shard' in script
    assert 'sam2_runner_shard${shard}_gpu${gpu}' in script
    assert 'sam2_runner_shard*.stdout.jsonl' in script
    assert 'rm -rf "$SHARD_ROOT"' in script
    assert 'wait "$pid"' in script
    assert 'build_superpoint_sam2_comask_edges.py' in script
    assert 'make_superpoint_sam2_edge_review.py' in script
