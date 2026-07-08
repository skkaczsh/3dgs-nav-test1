from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "show_current_mainline.py"


def test_show_current_mainline_json_output() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--format", "json"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(result.stdout)
    assert data["dataset"] == "MT20260616-175807"
    assert data["dense_patch_baseline"]["id"] == "energy_attach_v4_contact_evidence"
    assert data["dense_object_baseline"]["id"] == "superpoint_graph_v4_nearbbox_s070_e120_20260708_183437"
    assert data["current_promotion_candidate"]["id"] == "superpoint_graph_v4_nearbbox_s070_e120_20260708_183437"
    assert data["current_promotion_candidate"]["qa_candidate_id"] == "superpoint_graph_v4_nearbbox_s070_e120_20260708_183437"
    assert data["current_qa_report"]["promotion_gate_status"] == "visual_qa_pending_not_promoted"
    assert data["current_qa_report"]["review_index_url"] == "/tools/semantic_viewer_index.html"
    assert data["production_input_allowlist"]["passed"] is True
    assert data["production_input_allowlist"]["allowed_count"] == 6
    assert data["state_consistency"]["passed"] is True
    assert data["state_consistency"]["dataset"] == "MT20260616-175807"
    assert any(item["path"] == "scripts/run_dense_patch_object_refinement_v7.py" for item in data["approved_runners"])
    assert any(item["path"] == "scripts/run_semantic_evidence_pipeline.py" for item in data["approved_runners"])
    assert any(item["path"] == "scripts/cluster_superpoint_graph.py" for item in data["approved_runners"])
    assert any(item["pattern"] == "frame_object_points_stride10.ply" for item in data["forbidden_inputs"])


def test_show_current_mainline_text_output() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "current dense patch baseline:" in result.stdout
    assert "energy_attach_v4_contact_evidence" in result.stdout
    assert "remote executable baseline:" in result.stdout
    assert "latest remote run:" in result.stdout
    assert "dense_patch_object_refinement_v9_mainline_fixdeps_20260702_2108" in result.stdout
    assert "promotion_status: diagnostic_not_promoted" in result.stdout
    assert "Keep v8 as the current visual-promotion candidate" not in result.stdout
    assert "current promotion candidate:" in result.stdout
    assert "superpoint_graph_v4_nearbbox_s070_e120_20260708_183437 [visual_qa_pending_not_promoted]" in result.stdout
    assert "qa_candidate_id: superpoint_graph_v4_nearbbox_s070_e120_20260708_183437" in result.stdout
    assert "runner: scripts/cluster_superpoint_graph.py" in result.stdout
    assert "remote_runner: scripts/run_scan_train_superpoint_graph.sh" in result.stdout
    assert "approved runners:" in result.stdout
    assert "scripts/run_semantic_evidence_pipeline.py [semantic_evidence]" in result.stdout
    assert "blocker:" in result.stdout
    assert "current QA / promotion gate:" in result.stdout
    assert "promotion_gate_status: visual_qa_pending_not_promoted" in result.stdout
    assert "review_index: /tools/semantic_viewer_index.html" in result.stdout
    assert "production input allowlist:" in result.stdout
    assert "allowed_count=6" in result.stdout
    assert "state consistency:" in result.stdout
    assert "dataset=MT20260616-175807" in result.stdout
    assert "forbidden inputs:" in result.stdout
    assert "frame_object_points_stride10.ply" in result.stdout


def test_show_current_mainline_resolves_default_paths_outside_repo_root() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--format", "json"],
        cwd=ROOT.parent,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(result.stdout)
    assert data["dataset"] == "MT20260616-175807"


def test_show_current_mainline_supports_json_alias() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(result.stdout)
    assert data["production_input_allowlist"]["allowed_count"] == 6
    assert data["state_consistency"]["passed"] is True
